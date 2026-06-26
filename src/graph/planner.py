"""子图 B —— 学习规划：先搜索政策，再单次调用生成计划。

说明:
- 两步流程（搜索 → 生成）

未来路线图:
- [本地 RAG] 用包含省级教育考试院官方 PDF 的 VectorDB（ChromaDB）替代/增强网络搜索。
- [上下文过滤] 实现重排序阶段，优先官方 .gov.cn 域名内容。
- [省份路由] 自动注入用户省份上下文到搜索查询，
  以处理多样化的考试方案（如 3+1+2 vs. 3+3）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.rag.retriever import retrieve
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_search

logger = logging.getLogger(__name__)


# ── 节点 1: 搜索最新考试政策 ──────────────────────────────

# 搜索超时限制
_SEARCH_TIMEOUT = get_setting("planner.search_timeout", 15)


@traced_node
async def search_policy(state: TutorState) -> dict:
    """使用 DuckDuckGo 获取最新考试政策信息，15秒超时。"""
    year = datetime.now().year
    query = f"{year}年高考最新政策 考试时间安排 科目改革"

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

    return {"search_results": search_results}


# ── 节点: gather_intel（第2a阶段 —— 并行 Fan-out） ───────────

def _last_human_query(state: TutorState) -> str:
    """提取最后一条 HumanMessage 的内容。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


async def _gather_emotional_intel(state: TutorState) -> str:
    """调用 LLM 从对话历史中总结用户的情绪状态。"""
    llm = get_node_llm("emotional")
    fallback = get_fallback_llm(temperature=get_setting("emotional.temperature", 0.8))

    history_text = "\n".join(
        f"{'学生' if isinstance(m, HumanMessage) else '老师'}: {m.content}"
        for m in state["messages"]
        if hasattr(m, "content")
    )

    messages = [
        SystemMessage(content=load_prompt("gather_emotional_intel")),
        HumanMessage(content=history_text),
    ]

    try:
        with traced_llm_call(
            model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            node_name="gather_emotional_intel",
            temperature=get_setting("emotional.temperature", 0.8),
        ) as span:
            response = await async_invoke_with_fallback(
                llm, messages, fallback=fallback, span=span,
            )
        return response.content.strip()
    except Exception:
        logger.warning("Emotional intel LLM call failed, using fallback", exc_info=True)
        return "无法获取情绪分析，建议按常规方式安排计划。"


async def _gather_resource_intel(state: TutorState) -> str:
    """并行获取 RAG + 网络搜索结果，格式化为资源摘要。"""
    query = _last_human_query(state)
    subject = state.get("subject")
    subj = subject if subject and subject != "other" else None

    async def _rag():
        try:
            result = await asyncio.to_thread(retrieve, query=query, subject=subj)
            docs = result.get("docs", [])
            if not docs:
                return ""
            parts = [f"- {d.get('content', '')[:200]}" for d in docs[:3]]
            return "【知识库资源】\n" + "\n".join(parts)
        except Exception:
            logger.warning("RAG retrieval failed in gather_intel", exc_info=True)
            return ""

    async def _web():
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(web_search_fn, query),
                timeout=_SEARCH_TIMEOUT,
            )
            if not results:
                return ""
            parts = [f"- {r.get('title', '')}: {r.get('content', '')[:200]}" for r in results[:3]]
            return "【网络搜索】\n" + "\n".join(parts)
        except Exception:
            logger.warning("Web search failed in gather_intel", exc_info=True)
            return ""

    rag_text, web_text = await asyncio.gather(_rag(), _web())

    combined = "\n\n".join(part for part in [rag_text, web_text] if part)
    return combined if combined else "未获取到相关资源信息。"


@traced_node
async def gather_intel(state: TutorState) -> dict:
    """第2a阶段：并行收集情绪情报和资源情报。

    将 emotional_intel、resource_intel 和合并后的 intel_summary
    存入 TutorState，供对抗式规划子图使用。
    """
    emotional_intel, resource_intel = await asyncio.gather(
        _gather_emotional_intel(state),
        _gather_resource_intel(state),
    )

    intel_summary = f"【情绪分析】\n{emotional_intel}\n\n{resource_intel}"

    return {
        "emotional_intel": emotional_intel,
        "resource_intel": resource_intel,
        "intel_summary": intel_summary,
        # 初始化对抗式规划状态
        "adv_round": 0,
        "draft": "",
        "academic_verdict": "",
        "academic_reason": "",
        "emotional_verdict": "",
        "emotional_reason": "",
        "consensus": False,
        "revision_notes": "",
    }

