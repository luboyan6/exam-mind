"""OpenTelemetry 采集器 —— TracerProvider 初始化，支持 OTLP + SQLite 降级。

超时策略
~~~~~~~~
当 OTLP 采集器（Jaeger）不可达时，gRPC 导出器会进入
内部指数退避重试循环。使用 SDK 默认配置（timeout=10s，最多 3 次重试），
单次 ``export()`` 调用可能阻塞约 33 秒。
由于 ``BatchSpanProcessor`` 在持有 GIL 的守护线程上运行该调用，
asyncio 事件循环会因饥饿而导致 SSE 响应停滞。

解决方案：在导出器上设置较短的 RPC ``timeout``，同时在处理器上设置较短的
``export_timeout_millis``，使得采集器不可达时静默丢弃追踪数据，
而不是阻塞应用。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None

# ── 超时参数（秒 / 毫秒） ────────────────────────────────────────
_OTLP_TIMEOUT_SEC = 5          # gRPC 截止时间 & SDK 总重试上限
_BATCH_EXPORT_TIMEOUT_MS = 8000  # 单次批量导出的最大等待时间
_BATCH_SCHEDULE_DELAY_MS = 5000  # 处理器刷新间隔
_SHUTDOWN_TIMEOUT_MS = 5000      # provider.shutdown() 的最大等待时间


def setup_tracing() -> TracerProvider | None:
    """初始化 OpenTelemetry TracerProvider 并配置导出器。

    从环境变量读取配置：
        OTEL_TRACING_ENABLED  -- "true"/"false" 开关（默认 "true"）
        OTEL_SERVICE_NAME     -- 资源服务名（默认 "exam-mind"）
        OTEL_TRACES_EXPORTER  -- "otlp"、"sqlite" 或 "none"（默认 "otlp"）
        OTEL_EXPORTER_OTLP_ENDPOINT -- gRPC 端点（默认 "localhost:4317"）
        OTEL_SQLITE_FALLBACK_PATH   -- SQLite 数据库路径（默认 "logs/traces.db"）

    返回:
        已配置的 TracerProvider，追踪禁用时返回 None。
    """
    global _tracer_provider

    enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower()
    if enabled != "true":
        logger.info("OpenTelemetry tracing is disabled (OTEL_TRACING_ENABLED=%s)", enabled)
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "exam-mind")
    exporter_type = os.getenv("OTEL_TRACES_EXPORTER", "otlp").lower()

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # 主导出器：OTLP → Jaeger
    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True,
                timeout=_OTLP_TIMEOUT_SEC,
            )
            provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter,
                    export_timeout_millis=_BATCH_EXPORT_TIMEOUT_MS,
                    schedule_delay_millis=_BATCH_SCHEDULE_DELAY_MS,
                )
            )
            logger.info("OTLP exporter configured -> %s (timeout=%ss)", endpoint, _OTLP_TIMEOUT_SEC)
        except Exception:
            logger.exception("Failed to configure OTLP exporter, continuing with fallback only")

    # SQLite 降级导出器（除非导出器为 "none"，否则始终添加）
    if exporter_type != "none":
        try:
            from src.tracing.sqlite_exporter import SQLiteSpanExporter

            db_path = os.getenv("OTEL_SQLITE_FALLBACK_PATH", "logs/traces.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            sqlite_exporter = SQLiteSpanExporter(db_path)
            provider.add_span_processor(BatchSpanProcessor(sqlite_exporter))
            logger.info("SQLite fallback exporter configured -> %s", db_path)
        except Exception:
            logger.exception("Failed to configure SQLite fallback exporter")

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    logger.info(
        "OpenTelemetry tracing initialized (service=%s, exporter=%s)",
        service_name,
        exporter_type,
    )
    return provider


def get_tracer(name: str = "exam_mind") -> trace.Tracer:
    """返回 Tracer 实例。即使追踪未初始化也可安全调用。"""
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """刷新待处理的 Span 并关闭 TracerProvider。

    使用有界超时，确保应用在退出时不会因等待不可达的采集器而永久阻塞。
    """
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush(timeout_millis=_SHUTDOWN_TIMEOUT_MS)
        except Exception:
            logger.warning("Timeout flushing remaining traces — dropping them")
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry tracing shut down")
        _tracer_provider = None

