"""混合检索：向量搜索 + BM25 关键词搜索 + BGE 重排序。"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import get_setting
from src.rag.indexer import load_index
from src.rag.reranker import rerank

logger = logging.getLogger(__name__)

# 保留常量，与现有测试向后兼容
RELEVANCE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# 单例（懒加载）
# ---------------------------------------------------------------------------

_vectorstore = None
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict[str, Any]] = []  # parallel list of doc dicts
_bm25_doc_count: int = 0  # ChromaDB doc count at last BM25 build


def _get_vectorstore():
    """懒加载单例 —— 避免每次查询都重新读取 ChromaDB。"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_index()
    return _vectorstore


def _build_bm25_index() -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """从 ChromaDB 中的所有文档构建 BM25 索引。

    返回 (bm25_index, corpus)，其中 corpus 是与索引平行的文档字典列表，
    包含 content、source、metadata 字段。
    同时更新 ``_bm25_doc_count`` 为当前 ChromaDB 集合大小。
    """
    global _bm25_doc_count
    try:
        vs = _get_vectorstore()
        collection = vs._collection
        data = collection.get(include=["documents", "metadatas"])

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        _bm25_doc_count = collection.count()

        if not documents:
            logger.warning("ChromaDB collection is empty; BM25 index will be empty")
            return None, []

        corpus: list[dict[str, Any]] = []
        tokenized: list[list[str]] = []

        for doc_text, meta in zip(documents, metadatas):
            if not doc_text:
                continue
            corpus.append({
                "content": doc_text,
                "source": (meta or {}).get("source_file", "unknown"),
                "metadata": meta or {},
            })
            tokenized.append(jieba.lcut(doc_text))

        if not tokenized:
            return None, []

        return BM25Okapi(tokenized), corpus

    except Exception:
        logger.warning("Failed to build BM25 index; keyword search disabled", exc_info=True)
        return None, []


def _get_bm25(force_rebuild: bool = False) -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """懒加载 BM25 单例，支持自动失效重建。

    以下情况会重建索引：
    - 尚未构建索引（首次调用）
    - ``force_rebuild`` 为 True
    - ChromaDB 文档数与缓存的 ``_bm25_doc_count`` 不一致
    """
    global _bm25_index, _bm25_corpus

    needs_build = _bm25_index is None and not _bm25_corpus

    if not needs_build and not force_rebuild:
        # 检查 ChromaDB 自上次构建后是否发生变化
        try:
            vs = _get_vectorstore()
            current_count = vs._collection.count()
            if current_count != _bm25_doc_count:
                needs_build = True
                logger.info(
                    "BM25 invalidation: doc count changed %d → %d, rebuilding",
                    _bm25_doc_count,
                    current_count,
                )
        except Exception:
            logger.warning("Failed to check ChromaDB doc count", exc_info=True)

    if needs_build or force_rebuild:
        _bm25_index, _bm25_corpus = _build_bm25_index()

    return _bm25_index, _bm25_corpus


def _bm25_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """在缓存语料库上运行 BM25 关键词搜索。"""
    bm25, corpus = _get_bm25()
    if bm25 is None or not corpus:
        return []

    tokens = jieba.lcut(query)
    scores = bm25.get_scores(tokens)

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results: list[dict[str, Any]] = []
    for idx, score in scored[:top_k]:
        if score <= 0:
            break
        doc = corpus[idx]
        results.append({
            "content": doc["content"],
            "source": doc["source"],
            "score": round(float(score), 4),
            "metadata": doc["metadata"],
        })
    return results


def _content_hash(text: str) -> str:
    """用于去重的 MD5 哈希。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _merge_and_dedup(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并向量 + BM25 结果，按内容哈希去重。"""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    # 向量结果优先（具有校准后的相关度分数）
    for doc in vector_results:
        h = _content_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    for doc in bm25_results:
        h = _content_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    return merged


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    subject: Optional[str] = None,
    year: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """混合检索：向量搜索 + BM25 + 重排序。

    返回
    -------
    dict，包含以下字段：
        docs  : list[dict]  — [{content, source, score, metadata}, ...]
        is_hit: bool         — 最高分是否 >= 相关度阈值
    """
    vector_top_k = get_setting("rag.vector_top_k", 10)
    bm25_top_k = get_setting("rag.bm25_top_k", 10)
    reranker_top_n = get_setting("rag.reranker_top_n", top_k)
    threshold = get_setting("rag.relevance_threshold", RELEVANCE_THRESHOLD)

    # --- 1. 向量搜索 ---
    vectorstore = _get_vectorstore()

    where_filter: dict | None = None
    conditions: list[dict] = []
    if subject:
        conditions.append({"subject": {"$eq": subject}})
    if year:
        conditions.append({"year": {"$eq": year}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=vector_top_k,
        filter=where_filter,
    )

    vector_docs: list[dict[str, Any]] = []
    for doc, score in results:
        vector_docs.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "unknown"),
            "score": round(score, 4),
            "metadata": doc.metadata,
        })

    # --- 2. BM25 关键词搜索 ---
    bm25_docs = _bm25_search(query, top_k=bm25_top_k)

    # --- 3. 合并 + 去重 ---
    merged = _merge_and_dedup(vector_docs, bm25_docs)

    # --- 4. 重排序 ---
    if merged:
        ranked = rerank(query, merged, top_n=reranker_top_n)
    else:
        ranked = []

    # --- 5. 判断是否命中 ---
    is_hit = False
    if ranked:
        # 优先使用 rerank_score，否则使用原始分数
        best_score = ranked[0].get("rerank_score", ranked[0].get("score", 0))
        is_hit = best_score >= threshold

    return {"docs": ranked, "is_hit": is_hit}

