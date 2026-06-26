"""对抗式规划节点（扁平化到父图 —— AC-01）。

起草者 → [学术审查员 ∥ 情绪审查员] → 共识检查 → （循环/输出）

对抗式审查循环：起草者生成学习计划，
两个并行审查员（学术质量 + 情绪关怀）进行评估，
共识检查决定是否接受或要求修改。
安全阀（配置中的 max_rounds）在 N 次迭代后强制输出。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class ReviewVerdict(BaseModel):
    verdict: Literal["approve", "reject"]
    reason: str


class FeedbackClassification(BaseModel):
    """对用户反馈的学习计划进行分类。"""
    route: Literal["tweak", "rewrite"]
    reason: str


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _last_human_query(state: TutorState) -> str:
    """从状态消息中提取最后一条 HumanMessage 的内容。"""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# 节点定义
# ---------------------------------------------------------------------------


@traced_node
async def drafter_node(state: TutorState) -> dict[str, Any]:
    """根据情报和修改意见起草或重写学习计划。"""
    llm = get_node_llm("planner")
    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)

    user_request = _last_human_query(state)
    intel_summary = state.get("intel_summary", "")
    revision_notes = state.get("revision_notes", "")

    if revision_notes:
        # 重写路径：整合审查员反馈
        prompt_text = load_prompt("plan_rewrite").format(
            user_request=user_request,
            intel_summary=intel_summary,
            current_draft=state.get("draft", ""),
            revision_notes=revision_notes,
        )
    else:
        # 首次起草
        prompt_text = load_prompt("plan_drafter").format(
            user_request=user_request,
            intel_summary=intel_summary,
        )

    messages = [
        SystemMessage(content=load_prompt("plan_drafter_system")),
        HumanMessage(content=prompt_text),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="drafter_node",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {
        "draft": response.content,
        "adv_round": state.get("adv_round", 0) + 1,
    }


async def _run_reviewer(
    state: TutorState,
    *,
    system_prompt_name: str,
    node_name: str,
) -> ReviewVerdict:
    """学术审查和情绪审查的共享逻辑。"""
    reviewer_temp = get_setting("planner.reviewer_temperature", 0.0)
    llm = get_node_llm("planner", temperature=reviewer_temp)
    structured_primary = llm.with_structured_output(ReviewVerdict, method="json_mode")

    fallback_llm = get_fallback_llm(temperature=reviewer_temp)
    structured_fallback = fallback_llm.with_structured_output(ReviewVerdict, method="json_mode")

    review_prompt = (
        f"## 学习计划\n\n{state.get('draft', '')}\n\n"
        f"## 学生情况\n\n{state.get('intel_summary', '')}\n\n"
        f"请以 json 格式返回你的审查结论。"
    )
    messages = [
        SystemMessage(content=load_prompt(system_prompt_name)),
        HumanMessage(content=review_prompt),
    ]

    with traced_llm_call(
        model_name=get_setting("planner.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        node_name=node_name,
        temperature=reviewer_temp,
    ) as span:
        try:
            verdict = await async_invoke_with_fallback(
                structured_primary, messages,
                fallback=structured_fallback, span=span,
            )
            return verdict
        except Exception:
            logger.warning("Reviewer %s failed, defaulting to approve", node_name, exc_info=True)
            return ReviewVerdict(verdict="approve", reason="审查异常，默认通过")


@traced_node
async def reviewer_academic_node(state: TutorState) -> dict[str, Any]:
    """学术质量审查员。"""
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_academic_system",
        node_name="reviewer_academic",
    )
    return {"academic_verdict": verdict.verdict, "academic_reason": verdict.reason}


@traced_node
async def reviewer_emotional_node(state: TutorState) -> dict[str, Any]:
    """情绪关怀审查员。"""
    verdict = await _run_reviewer(
        state,
        system_prompt_name="plan_reviewer_emotional_system",
        node_name="reviewer_emotional",
    )
    return {"emotional_verdict": verdict.verdict, "emotional_reason": verdict.reason}


@traced_node
async def consensus_check_node(state: TutorState) -> dict[str, Any]:
    """检查两个审查员是否都通过，或在达到最大轮次时强制输出。"""
    current_round = state.get("adv_round", 0)
    max_rounds = get_setting("planner.adversarial_max_rounds", 3)
    academic = state.get("academic_verdict", "")
    emotional = state.get("emotional_verdict", "")

    both_approve = academic == "approve" and emotional == "approve"

    # 安全阀：达到最大轮次时强制通过
    if current_round >= max_rounds:
        if not both_approve:
            logger.warning(
                "Max rounds (%d) reached with unresolved rejections, forcing output",
                max_rounds,
            )
        return {"consensus": True, "revision_notes": ""}

    if both_approve:
        return {"consensus": True, "revision_notes": ""}

    # 收集拒绝理由，供修改使用（AC-03）
    notes_parts: list[str] = []
    if academic == "reject":
        reason = state.get("academic_reason", "未提供原因")
        notes_parts.append(f"[学术审查] {reason}")
    if emotional == "reject":
        reason = state.get("emotional_reason", "未提供原因")
        notes_parts.append(f"[情绪审查] {reason}")

    return {
        "consensus": False,
        "revision_notes": "; ".join(notes_parts) if notes_parts else "需要修改",
    }


@traced_node
async def adv_rewrite_node(state: TutorState) -> dict[str, Any]:
    """在将草稿发回起草者修改前重置审查结论。"""
    return {
        "academic_verdict": "",
        "academic_reason": "",
        "emotional_verdict": "",
        "emotional_reason": "",
    }


@traced_node
async def plan_output_node(state: TutorState) -> dict:
    """最终计划输出 —— 若有检查点管理器则中断以进行 HIL 审查。"""
    plan_text = state.get("draft", "")

    # HIL：暂停以进行人工审查。无检查点管理器时跳过（无状态模式）。
    try:
        user_response = interrupt(plan_text)
    except ValueError:
        # 无检查点管理器 —— 跳过 HIL，直接使用草稿
        logger.warning("interrupt() failed (no checkpointer?), skipping HIL review")
        user_response = plan_text

    # ── 用户提供了反馈（dict）→ 路由到 feedback_router ──
    if isinstance(user_response, dict) and user_response.get("action") == "feedback":
        return {
            "hil_action": "feedback",
            "hil_feedback": user_response.get("text", ""),
        }

    # ── 用户确认（string）→ 完成 ──
    final_plan = user_response if isinstance(user_response, str) and user_response else plan_text
    return {
        "plan": final_plan,
        "messages": [AIMessage(content=final_plan)],
        "hil_action": "confirm",
    }


@traced_node
async def feedback_router(state: TutorState) -> dict[str, Any]:
    """将用户的 HIL 反馈分类为 'tweak'（微调）或 'rewrite'（重写）。

    使用 supervisor 的快速模型进行快速分类。
    同时更新 hil_summary：将旧摘要 + 新反馈压缩为一个字符串。
    """
    llm = get_node_llm("supervisor")
    structured_llm = llm.with_structured_output(FeedbackClassification, method="json_mode")

    feedback = state.get("hil_feedback", "")
    draft = state.get("draft", "")
    old_summary = state.get("hil_summary", "")

    # ── 步骤 1: 反馈分类 ──
    classify_prompt = (
        f"学生对以下学习计划提出了修改意见。\n\n"
        f"## 当前计划（前500字）\n{draft[:500]}\n\n"
        f"## 学生反馈\n{feedback}\n\n"
        f"判断这个反馈需要的修改程度：\n"
        f"- tweak: 只需要局部微调（如调整某天科目、修改时间、增删某个小项）\n"
        f"- rewrite: 需要重新规划（如整体思路不对、完全不符合需求、需要换方向）\n\n"
        f"请以 json 格式返回你的分类结果。"
    )

    try:
        result = await structured_llm.ainvoke([
            SystemMessage(content="你是一个学习计划修改分类器。根据学生反馈判断需要微调还是重写。"),
            HumanMessage(content=classify_prompt),
        ])
        route = result.route
    except Exception:
        logger.warning("Feedback classification failed, defaulting to tweak")
        route = "tweak"

    # ── 步骤 2: 压缩摘要（覆写而非追加） ──
    if old_summary:
        new_summary = f"历史修改摘要: {old_summary[:200]}\n最新反馈: {feedback[:500]}"
    else:
        new_summary = f"用户反馈: {feedback[:500]}"

    if route == "rewrite":
        # 重写：清空对抗状态，将反馈视为新的方向
        return {
            "feedback_route": "rewrite",
            "hil_summary": new_summary,
            "revision_notes": feedback,
            "adv_round": 0,
            "draft": "",
            "academic_verdict": "",
            "academic_reason": "",
            "emotional_verdict": "",
            "emotional_reason": "",
            "consensus": False,
        }
    else:
        # 微调：保留草稿，将反馈传递给 plan_tweak
        return {
            "feedback_route": "tweak",
            "hil_summary": new_summary,
        }


@traced_node
async def plan_tweak_node(state: TutorState) -> dict[str, Any]:
    """根据用户反馈对计划进行定向微调。

    单次 LLM 调用 —— 小修改无需审查循环。
    """
    llm = get_node_llm("planner")
    temperature = get_setting("planner.temperature", 0.7)
    fallback = get_fallback_llm(temperature=temperature)

    draft = state.get("draft", "")
    feedback = state.get("hil_feedback", "")
    summary = state.get("hil_summary", "")

    prompt = (
        f"请根据学生的反馈对以下学习计划进行**局部微调**。\n"
        f"只修改学生提到的部分，保持其他内容不变。\n\n"
        f"## 当前计划\n{draft}\n\n"
        f"## 学生反馈\n{feedback}\n\n"
    )
    if summary:
        prompt += f"## 修改历史摘要\n{summary}\n\n"
    prompt += "请输出修改后的完整计划："

    messages = [
        SystemMessage(content=load_prompt("plan_drafter_system")),
        HumanMessage(content=prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="plan_tweak",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"draft": response.content}


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------


def should_output_or_revise(state: TutorState) -> str:
    """共识检查后的条件边：输出或修改。"""
    if state.get("consensus", False):
        return "output"
    return "revise"


def route_after_hil(state: TutorState) -> str:
    """计划输出后的条件边：确认 → 结束，反馈 → feedback_router。"""
    return "feedback" if state.get("hil_action") == "feedback" else "end"


def route_feedback(state: TutorState) -> str:
    """反馈路由后的条件边：微调或重写。"""
    return state.get("feedback_route", "tweak")

