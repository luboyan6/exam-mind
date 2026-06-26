"""跨模块复用的数据结构。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """来自前端的聊天请求。"""

    query: str = Field(max_length=4096)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    """恢复被人工介入（HIL）中断的图执行。"""

    thread_id: str
    edited_plan: str = Field(default="", max_length=16384)
    feedback: str | None = Field(default=None, max_length=4096)

