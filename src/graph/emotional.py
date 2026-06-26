"""情绪支持节点 —— 单次 LLM 调用，采用班主任人设。"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, SystemMessage

from src.config import get_setting, load_prompt
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node


@traced_node
async def emotional_response(state: TutorState) -> dict:
    """返回温暖且实用的情绪支持回复。"""
    llm = get_node_llm("emotional")

    history = [SystemMessage(content=load_prompt("emotional_system"))]
    for msg in state["messages"]:
        history.append(msg)

    temperature = get_setting("emotional.temperature", 0.8)
    fallback = get_fallback_llm(temperature=temperature)

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="emotional_response",
        temperature=temperature,
    ) as span:
        response = await async_invoke_with_fallback(
            llm, history, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}

