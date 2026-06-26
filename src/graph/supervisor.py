"""Supervisor 节点 —— 基于 LLM 的意图分类、学科检测和知识点提取。

将路由分发与学科知识点提取合并为单次 LLM 调用，
消除了学科路径上多余的 API 往返。
使用结构化输出（Pydantic）代替手动 JSON 解析。
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.llm import get_node_llm
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


class SupervisorOutput(BaseModel):
    """Supervisor 意图分类的结构化输出模型。"""
    intent: Literal["academic", "planning", "emotional", "unknown"]
    keywords: list[str]
    confidence: float


_VALID_INTENTS = set(get_setting(
    "supervisor.valid_intents",
    ["academic", "planning", "emotional", "unknown"],
))


@traced_node
async def supervisor_node(state: TutorState) -> dict:
    """在一次 LLM 调用中完成意图分类、学科检测和知识点提取。

    使用 ``with_structured_output(SupervisorOutput)`` 确保解析可靠。

    返回:
        包含 ``intent``、``subject``、``keypoints`` 的状态更新字典。
    """
    llm = get_node_llm("supervisor")
    structured_llm = llm.with_structured_output(SupervisorOutput)

    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    temperature = get_setting("supervisor.temperature", 0.0)
    model_name = get_setting("supervisor.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    with traced_llm_call(
        model_name=model_name,
        node_name="supervisor",
        temperature=temperature,
    ):
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=load_prompt("supervisor_system")),
                HumanMessage(content=user_text),
            ])
            intent = result.intent
            subject = "other"
            keypoints = result.keywords
            # 根据结构化输出上下文检测学科
            if intent == "academic" and keypoints:
                query_lower = user_text.lower()
                math_keywords = {"数学", "函数", "方程", "几何", "代数", "概率", "向量",
                                 "导数", "积分", "椭圆", "双曲线", "抛物线", "三角"}
                chinese_keywords = {"语文", "作文", "文言文", "古诗", "阅读理解", "诗词",
                                    "鉴赏", "修辞", "散文", "小说"}
                if any(kw in query_lower for kw in math_keywords):
                    subject = "math"
                elif any(kw in query_lower for kw in chinese_keywords):
                    subject = "chinese"
        except Exception:
            logger.warning("Supervisor structured output failed, defaulting to academic")
            intent = "academic"
            subject = "other"
            keypoints = []

    if intent not in _VALID_INTENTS:
        intent = "academic"

    return {"intent": intent, "subject": subject, "keypoints": keypoints}


@traced_node
async def handle_unknown(state: TutorState) -> dict:
    """处理超纲问题，返回友好的引导回复。"""
    return {
        "messages": [AIMessage(
            content=(
                "抱歉，这个问题超出了我的辅导范围。我是你的高考辅导助手，"
                "可以帮你解答学科知识、制定学习计划、或者聊聊学习中的烦恼。"
                "请问有什么学习上的问题需要帮助吗？"
            ),
        )],
    }


def route_by_intent(state: TutorState) -> str:
    """条件边函数：根据意图路由到对应的子图。"""
    return state.get("intent", "academic")

