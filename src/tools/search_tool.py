"""DuckDuckGo 网络搜索工具，用于在线补充检索。

提供懒加载单例搜索工具和便捷的 ``search()`` 函数，
将 DuckDuckGo 输出规范化为统一 schema
（``content``、``title``、``url``）供图节点消费。
"""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults

_search_tool: DuckDuckGoSearchResults | None = None


def get_search_tool() -> DuckDuckGoSearchResults:
    """懒加载单例 —— 避免跨图节点重复实例化。"""
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchResults(
            max_results=3,
            output_format="list",
        )
    return _search_tool


def search(query: str) -> list[dict]:
    """执行网络搜索并返回规范化结果。

    DuckDuckGo 返回 ``snippet``、``title``、``link`` 字段。
    本函数将其重映射为 ``content``、``title``、``url``，
    使下游消费者与具体服务商解耦。

    参数:
        query: 搜索查询字符串。

    返回:
        包含 ``content``、``title``、``url`` 键的字典列表。
        任何失败均返回空列表。
    """
    tool = get_search_tool()
    try:
        results = tool.invoke(query)
    except Exception:
        return []

    if isinstance(results, str):
        return [{"content": results, "title": "", "url": ""}]

    return [
        {
            "content": r.get("snippet", r.get("content", "")),
            "title": r.get("title", ""),
            "url": r.get("link", r.get("url", "")),
        }
        for r in results
    ]

