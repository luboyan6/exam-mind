"""PostgreSQL 检查点管理器生命周期管理，用于 LangGraph 状态持久化。

使用 langgraph-checkpoint-postgres 的 AsyncPostgresSaver 跨会话持久化
对话状态，以 thread_id 为键。
"""

from __future__ import annotations

import logging
import os
import uuid

logger = logging.getLogger(__name__)


def get_db_uri() -> str | None:
    """从环境变量中读取 PostgreSQL 连接 URI。

    将 SQLAlchemy 风格的 scheme（如 ``postgresql+asyncpg://``）
    规范化为 psycopg 所需的 ``postgresql://``。

    返回:
        DB_URI 字符串，未配置时返回 None。
    """
    uri = os.getenv("DB_URI")
    if uri and uri.startswith("postgresql+"):
        uri = "postgresql" + uri[uri.index("://"):]
    return uri


def make_thread_config(thread_id: str | None = None) -> dict:
    """构建包含 thread_id 的 LangGraph 配置字典。

    参数:
        thread_id: 显式会话 ID。为 None 时自动生成新的 UUID。

    返回:
        格式为 ``{"configurable": {"thread_id": "..."}}`` 的配置字典。
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    return {"configurable": {"thread_id": thread_id}}

