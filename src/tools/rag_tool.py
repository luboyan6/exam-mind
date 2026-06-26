"""RAG 检索封装为 LangChain 工具，供 LangGraph 节点调用。"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from src.rag.retriever import retrieve


@tool
def rag_retrieve(query: str, subject: Optional[str] = None) -> dict:
    """搜索本地 ExamMind 知识库（历年真题、考纲、笔记）。

    返回字典包含：
      - docs: 匹配的文档块列表，含 content、source、score
      - is_hit: 检索是否找到相关结果
    """
    return retrieve(query=query, subject=subject)

