"""LLM 工厂与容灾调用逻辑。

提供弹性充足的 invoke_with_fallback()，捕获临时性 API 错误
（超时、502、限流等）并在备用模型上重试，
同时将故障转移事件记录在当前活动的 OpenTelemetry Span 上。
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

from src.config import get_setting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 可恢复错误类型 —— 触发自动容灾降级
# ---------------------------------------------------------------------------

_FALLBACK_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

try:
    import openai

    _FALLBACK_ERRORS = (
        TimeoutError,
        ConnectionError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.RateLimitError,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# LLM 工厂函数
# ---------------------------------------------------------------------------

def get_node_llm(node_name: str, **overrides) -> ChatOpenAI:
    """构建针对特定图节点配置的 ChatOpenAI 实例。

    从 ``settings.yaml`` 读取每个节点的 ``model``、``base_url``、
    ``api_key_env`` 和 ``temperature``。节点无显式配置时，
    回退到 ``DEEPSEEK_*`` 环境变量。
    """
    model = get_setting(f"{node_name}.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    api_key_env = get_setting(f"{node_name}.api_key_env", "DEEPSEEK_API_KEY")
    base_url = get_setting(f"{node_name}.base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    temperature = get_setting(f"{node_name}.temperature", 0.7)

    defaults = dict(
        model=model,
        api_key=os.getenv(api_key_env),
        base_url=base_url,
        temperature=temperature,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


def get_primary_llm(**overrides) -> ChatOpenAI:
    """根据 DEEPSEEK_* 环境变量构建主聊天模型。"""
    defaults = dict(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


def get_fallback_llm(**overrides) -> ChatOpenAI:
    """根据 FALLBACK_* 环境变量构建备用聊天模型。

    默认回退到主 API 配置，这样临时性错误（502、超时）
    可以在同一端点上获得第二次机会。可通过覆盖 ``FALLBACK_MODEL``、
    ``FALLBACK_API_KEY`` 和 ``FALLBACK_BASE_URL`` 指向本地 Ollama
    实例或其他云服务商。
    """
    defaults = dict(
        model=os.getenv("FALLBACK_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        api_key=os.getenv("FALLBACK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "not-configured",
        base_url=os.getenv("FALLBACK_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        temperature=0.7,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


# ---------------------------------------------------------------------------
# 弹性调用
# ---------------------------------------------------------------------------

def invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    """调用 *primary*；遇到可恢复错误时故障转移到 *fallback*。

    参数:
        primary: 主聊天模型实例。
        messages: 传给 ``invoke()`` 的消息列表。
        fallback: 可选的备用聊天模型。为 ``None`` 时错误直接抛出。
        span: 可选的 OTel Span，用于记录容灾元数据。

    返回:
        成功模型（主或备用）的 LLM 响应。

    抛出:
        未配置备用模型时抛出原始错误；
        两个模型都失败时抛出备用模型的错误。
    """
    try:
        response = primary.invoke(messages)
        if span is not None:
            span.set_attribute("llm.fallback_used", False)
        return response
    except _FALLBACK_ERRORS as exc:
        if fallback is None:
            raise

        logger.warning(
            "Primary LLM failed (%s: %s), falling back",
            type(exc).__name__,
            exc,
        )

        if span is not None:
            span.set_attribute("llm.fallback_used", True)
            span.set_attribute(
                "llm.fallback_model",
                getattr(fallback, "model_name", "unknown"),
            )
            span.add_event(
                "llm.fallback_triggered",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        return fallback.invoke(messages)


async def async_invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    """invoke_with_fallback 的异步版本；全程使用 ainvoke()。

    参数:
        primary: 主聊天模型（或结构化输出链）实例。
        messages: 传给 ``ainvoke()`` 的消息列表。
        fallback: 可选的备用聊天模型。为 ``None`` 时错误直接抛出。
        span: 可选的 OTel Span，用于记录容灾元数据。

    返回:
        成功模型（主或备用）的 LLM 响应。

    抛出:
        未配置备用模型时抛出原始错误；
        两个模型都失败时抛出备用模型的错误。
    """
    try:
        response = await primary.ainvoke(messages)
        if span is not None:
            span.set_attribute("llm.fallback_used", False)
        return response
    except _FALLBACK_ERRORS as exc:
        if fallback is None:
            raise

        logger.warning(
            "Primary LLM failed (%s: %s), falling back",
            type(exc).__name__,
            exc,
        )

        if span is not None:
            span.set_attribute("llm.fallback_used", True)
            span.set_attribute(
                "llm.fallback_model",
                getattr(fallback, "model_name", "unknown"),
            )
            span.add_event(
                "llm.fallback_triggered",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        return await fallback.ainvoke(messages)

