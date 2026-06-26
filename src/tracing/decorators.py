"""追踪装饰器 —— 对图节点和操作的无侵入式插桩。"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.tracing.collector import get_tracer


# ---------------------------------------------------------------------------
# @traced_node —— 包装 LangGraph 节点函数（同步和异步）
# ---------------------------------------------------------------------------

def traced_node(func: Callable) -> Callable:
    """装饰器：用 OpenTelemetry Span 包装 LangGraph 节点函数。

    同时支持同步和异步节点函数。
    记录：节点名称、执行耗时、输入/输出状态键，
    以及任何异常作为 Span 事件。
    """

    def _record_result(span, result):
        if isinstance(result, dict):
            span.set_attribute("graph.node.output_keys", str(list(result.keys())))

            if "intent" in result:
                span.set_attribute("graph.node.intent", result["intent"])
            if "subject" in result:
                span.set_attribute("graph.node.subject", result["subject"])
            if "keypoints" in result:
                span.set_attribute("graph.node.keypoint_count", len(result["keypoints"]))
            if "context" in result:
                span.set_attribute("graph.node.context_count", len(result["context"]))
            if "search_results" in result:
                span.set_attribute("graph.node.search_result_count", len(result["search_results"]))
            if "messages" in result:
                span.set_attribute("graph.node.message_count", len(result["messages"]))
            if "retry_count" in result:
                span.set_attribute("graph.node.retry_count", result["retry_count"])
            if "hallucination_detected" in result:
                span.set_attribute("graph.node.hallucination_detected", result["hallucination_detected"])

        span.set_status(StatusCode.OK)

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
            tracer = get_tracer()
            with tracer.start_as_current_span(
                f"graph.node.{func.__name__}",
                attributes={
                    "graph.node.name": func.__name__,
                    "graph.node.input_keys": str(list(state.keys())),
                },
            ) as span:
                try:
                    result = await func(state, *args, **kwargs)
                    _record_result(span, result)
                    return result
                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            f"graph.node.{func.__name__}",
            attributes={
                "graph.node.name": func.__name__,
                "graph.node.input_keys": str(list(state.keys())),
            },
        ) as span:
            try:
                result = func(state, *args, **kwargs)
                _record_result(span, result)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return sync_wrapper


# ---------------------------------------------------------------------------
# traced_llm_call —— LLM 调用的上下文管理器
# ---------------------------------------------------------------------------

@contextmanager
def traced_llm_call(
    model_name: str = "unknown",
    node_name: str = "unknown",
    temperature: float | None = None,
) -> Generator[trace.Span, None, None]:
    """为 LLM 调用创建子 Span 的上下文管理器。

    产出 Span 对象，供调用者设置额外属性（如 Token 计数）。
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"llm.invoke.{node_name}",
        attributes={
            "llm.model": model_name,
            "llm.node": node_name,
        },
    ) as span:
        if temperature is not None:
            span.set_attribute("llm.temperature", temperature)

        start = time.monotonic()
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            span.set_attribute("llm.latency_ms", round(latency_ms, 2))


# ---------------------------------------------------------------------------
# traced_retrieval —— RAG 检索的上下文管理器
# ---------------------------------------------------------------------------

@contextmanager
def traced_retrieval(
    query: str,
    subject: str | None = None,
    top_k: int = 5,
) -> Generator[trace.Span, None, None]:
    """RAG 检索的上下文管理器 Span。"""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "rag.retrieve",
        attributes={
            "rag.query": query[:200],
            "rag.subject": subject or "all",
            "rag.top_k": top_k,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


# ---------------------------------------------------------------------------
# traced_search —— 网络搜索的上下文管理器
# ---------------------------------------------------------------------------

@contextmanager
def traced_search(
    query: str,
    timeout: int = 15,
) -> Generator[trace.Span, None, None]:
    """网络搜索的上下文管理器 Span。"""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "web.search",
        attributes={
            "search.query": query[:200],
            "search.timeout_sec": timeout,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise

