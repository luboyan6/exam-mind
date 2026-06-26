"""SiliconFlow BGE 重排序 API 封装。

调用 SiliconFlow 重排序端点对候选文档按查询重新打分。
API 失败时按原始顺序返回文档（优雅降级）。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.config import get_setting

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
_TIMEOUT = 15  # seconds


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """通过 SiliconFlow BGE 重排序器对 *documents* 按 *query* 重新排序。

    参数
    ----------
    query : str
        用户的搜索查询。
    documents : list[dict]
        候选文档列表。每个字典**必须**包含 ``"content"`` 键。
    top_n : int, 可选
        返回结果数。默认取 settings 中的 ``rag.reranker_top_n``
        （回退值: 5）。

    返回
    -------
    list[dict]
        按重排序相关度排序的 *top_n* 个文档，每个文档新增
        ``"rerank_score"`` 键。失败时按原始顺序返回截断到 *top_n* 的文档。
    """
    if not documents:
        return []

    if top_n is None:
        top_n = get_setting("rag.reranker_top_n", 5)

    api_key = os.getenv("SILICONFLOW_API_KEY")
    model = os.getenv(
        "RERANKER_MODEL",
        get_setting("rag.reranker_model", _DEFAULT_RERANKER_MODEL),
    )

    doc_texts = [d["content"] for d in documents]

    try:
        resp = httpx.post(
            _RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "query": query,
                "documents": doc_texts,
                "top_n": top_n,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("Reranker API call failed; returning original order", exc_info=True)
        return documents[:top_n]

    results: list[dict[str, Any]] = data.get("results", [])
    ranked: list[dict[str, Any]] = []
    for item in results:
        idx = item["index"]
        if 0 <= idx < len(documents):
            doc = {**documents[idx], "rerank_score": item["relevance_score"]}
            ranked.append(doc)

    return ranked[:top_n]

