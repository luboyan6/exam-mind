"""子图 A —— 学科辅导：并行检索（Fan-out/Fan-in）、回答生成、
幻觉评估与重试循环。

知识点提取由 supervisor 节点完成（合并以降低延迟），
因此本子图从 academic_router 开始，并行分发到
rag_retrieve 和 web_search。
"""

from __future__ import annotations

import asyncio
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import CONTEXT_CLEAR, TutorState
from src.rag.retriever import retrieve
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_retrieval, traced_search

logger = logging.getLogger(__name__)

MAX_RETRIES = get_setting("academic.max_retries", 2)


# ── 幻觉评估的结构化输出模型 ────────────────────────────────────
class HallucinationEvaluation(BaseModel):
    """由 LLM 评估的忠实度判断结果。"""

    is_faithful: bool = Field(
        description="回答是否基于检索上下文且未捷造内容",
    )
    reason: str = Field(
        description="评估判断的简要说明",
    )


def _last_human_query(state: TutorState) -> str:
    """提取最后一条 HumanMessage 的内容（对重试循环健壮）。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ── 节点 0: 学术路由（Fan-out 触发器） ─────────────────────────

@traced_node
async def academic_router(state: TutorState) -> dict:
    """并行 Fan-out 的路由节点。在重试路径上清空上下文。"""
    if state.get("retry_count", 0) > 0:
        return {"context": CONTEXT_CLEAR}
    return {}


# ── 节点 0b: 查询改写（仅重试路径） ─────────────────────────

@traced_node
async def rewrite_query(state: TutorState) -> dict:
    """根据幻觉反馈改写用户查询，以改善检索效果。"""
    original_query = _last_human_query(state)
    reason = state.get("hallucination_reason", "")

    llm = get_node_llm("supervisor")
    rewrite_prompt = load_prompt("rewrite_query").format(
        original_query=original_query,
        hallucination_reason=reason,
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content="你是一个查询改写助手。根据反馈改进用户的搜索查询。"),
            HumanMessage(content=rewrite_prompt),
        ])
        rewritten = response.content.strip()
    except Exception:
        logger.warning("Query rewrite failed, using original query")
        rewritten = original_query

    return {"rewritten_query": rewritten}


# ── 节点 1: RAG 检索（并行分支 A） ───────────────────────────

@traced_node
async def rag_retrieve(state: TutorState) -> dict:
    """使用 supervisor 节点提取的知识点查询 ChromaDB。"""
    rewritten = state.get("rewritten_query", "")
    keypoints = state.get("keypoints", [])
    subject = state.get("subject")

    if rewritten:
        query = rewritten
    elif keypoints:
        query = " ".join(keypoints)
    else:
        query = _last_human_query(state)

    subj = subject if subject != "other" else None

    with traced_retrieval(query=query, subject=subj) as span:
        result = await asyncio.to_thread(retrieve, query=query, subject=subj)
        span.set_attribute("rag.doc_count", len(result.get("docs", [])))
        span.set_attribute("rag.is_hit", result.get("is_hit", False))
        if result.get("docs"):
            span.set_attribute("rag.top_score", result["docs"][0].get("score", 0))

    docs = result["docs"]
    return {"context": [{"type": "rag", **doc} for doc in docs]}


# ── 节点 2: 网络搜索（并行分支 B） ─────────────────────────

_SEARCH_TIMEOUT = get_setting("academic.search_timeout", 15)


@traced_node
async def web_search(state: TutorState) -> dict:
    """Fan-out 网络搜索 —— 与 rag_retrieve 并行执行。"""
    rewritten = state.get("rewritten_query", "")
    query = rewritten if rewritten else _last_human_query(state)

    with traced_search(query=query, timeout=_SEARCH_TIMEOUT) as span:
        try:
            search_results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
                timeout=_SEARCH_TIMEOUT,
            )
            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute("search.timed_out", False)
        except asyncio.TimeoutError:
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", True)
        except Exception:
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", False)

    return {"context": [{"type": "web", **r} for r in search_results]}


# ── 节点 3: 回答生成 ──────────────────────────────────────────

def _format_retrieved(docs: list[dict]) -> str:
    if not docs:
        return "无相关参考资料。"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] 来源：{d.get('source', '未知')}（相关度：{d.get('score', 'N/A')}）\n{d.get('content', '')}")
    return "\n\n".join(parts)


def _format_search(results: list[dict]) -> str:
    if not results:
        return "无网络搜索结果。"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.get('title', '无标题')} ({r.get('url', '')})\n{r.get('content', '')}")
    return "\n\n".join(parts)


@traced_node
async def generate_answer(state: TutorState) -> dict:
    """基于合并后的上下文（RAG + 网络）通过 LLM 综合生成最终回答。"""
    llm = get_node_llm("academic")

    question = _last_human_query(state)

    # 按来源类型拆分合并后的上下文
    context = state.get("context", [])
    rag_docs = [c for c in context if c.get("type") == "rag"]
    web_results = [c for c in context if c.get("type") == "web"]

    temperature = get_setting("academic.temperature", 0.7)
    user_prompt = load_prompt("academic_answer").format(
        retrieved_context=_format_retrieved(rag_docs),
        search_context=_format_search(web_results),
        question=question,
    )

    fallback = get_fallback_llm(temperature=temperature)
    messages = [
        SystemMessage(content=load_prompt("academic_system")),
        HumanMessage(content=user_prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="generate_answer",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}


# ── 节点 4: 幻觉评估（反思循环） ───────────────────────────

@traced_node
async def evaluate_hallucination(state: TutorState) -> dict:
    """评估生成的回答是否超出检索上下文范围产生幻觉。

    使用结构化 LLM 输出判断忠实度。检测到幻觉时，
    增加 retry_count 以触发条件边进行重新检索。
    解析或模型失败时默认通过（安全降级）。
    """
    eval_temp = get_setting("academic.hallucination_eval_temperature", 0.0)
    llm = get_node_llm("academic", temperature=eval_temp)
    structured_primary = llm.with_structured_output(HallucinationEvaluation)

    fallback_llm = get_fallback_llm(temperature=eval_temp)
    structured_fallback = fallback_llm.with_structured_output(HallucinationEvaluation)

    # 提取生成的回答（最后一条消息）和原始问题
    answer = state["messages"][-1].content
    question = _last_human_query(state)

    # 从所有检索源构建上下文
    docs = state.get("context", [])
    context = "\n".join(d.get("content", "") for d in docs) if docs else ""

    eval_prompt = load_prompt("hallucination_eval").format(
        question=question, context=context, answer=answer,
    )

    retry_count = state.get("retry_count", 0)

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="evaluate_hallucination",
        temperature=eval_temp,
    ) as span:
        try:
            evaluation = await async_invoke_with_fallback(
                structured_primary,
                [
                    SystemMessage(content=load_prompt("hallucination_system")),
                    HumanMessage(content=eval_prompt),
                ],
                fallback=structured_fallback,
                span=span,
            )
            is_faithful = evaluation.is_faithful
        except Exception:
            logger.warning("Hallucination evaluation failed, defaulting to valid")
            is_faithful = True

    hallucination_detected = not is_faithful

    result: dict = {"hallucination_detected": hallucination_detected}
    if hallucination_detected:
        result["retry_count"] = retry_count + 1
        result["hallucination_reason"] = evaluation.reason

    return result


def should_retry_or_end(state: TutorState) -> str:
    """条件边：通过 academic_router 重试或路由到 END。

    检测到幻觉时允许最多 MAX_RETRIES 次重新检索。
    重试次数用尽后，无论如何都路由到 END。
    """
    if (
        state.get("hallucination_detected", False)
        and state.get("retry_count", 0) <= MAX_RETRIES
    ):
        return "retry"
    return "end"

