"""ChromaDB 索引构建器，支持增量更新。

使用 SiliconFlow 的 OpenAI 兼容嵌入 API（BAAI/bge-m3）
代替本地 HuggingFace 模型，避免重度本地依赖。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import math

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

COLLECTION_NAME = "exam_docs"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def _l2_to_relevance(distance: float) -> float:
    """将 Chroma 的 L2 距离转换为 [0, 1] 范围的相关度分数。

    Chroma 默认距离度量为 L2（欧氏距离）。对于归一化嵌入，
    最大 L2 距离为 sqrt(2)。我们将 [0, sqrt(2)] 线性映射到
    [1, 0]，使分数越高表示相关性越强，且值始终在 [0, 1] 内。
    """
    return 1.0 - distance / math.sqrt(2)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_persist_dir(persist_directory: Optional[str] = None) -> str:
    """始终解析为锚定在项目根目录的绝对路径。"""
    rel = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_store/")
    path = Path(rel)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def _get_embedding(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    """创建基于 SiliconFlow 的 OpenAI 兼容嵌入客户端。

    参数:
        model_name: 覆盖嵌入模型标识符。
            回退到 ``EMBEDDING_MODEL`` 环境变量，再到 ``DEFAULT_EMBEDDING_MODEL``。

    返回:
        指向 SiliconFlow 的已配置 ``OpenAIEmbeddings`` 实例。
    """
    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base=os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ),
    )


def _content_id(doc: Document) -> str:
    """根据文本块内容生成确定性 ID —— 跨重复运行的真实去重。"""
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{doc.metadata.get('source_file', 'unknown')}_{digest}"


def build_index(
    documents: list[Document],
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """从 *documents* 创建（或更新）ChromaDB 集合。

    使用文本块内容的 md5 哈希作为去重 ID，确保重复运行安全。

    参数:
        documents: 待索引的 LangChain Document 对象列表。
        persist_directory: 覆盖 ChromaDB 持久化路径。
        embedding_model: 覆盖嵌入模型标识符。

    返回:
        已填充的 Chroma 向量存储实例。
    """
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding = _get_embedding(embedding_model)

    ids = [_content_id(doc) for doc in documents]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        ids=ids,
        relevance_score_fn=_l2_to_relevance,
    )
    return vectorstore


def load_index(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """从磁盘加载已有的 ChromaDB 集合。

    参数:
        persist_directory: 覆盖 ChromaDB 持久化路径。
        embedding_model: 覆盖嵌入模型标识符。

    返回:
        已加载的 Chroma 向量存储实例。
    """
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding = _get_embedding(embedding_model)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=persist_directory,
        relevance_score_fn=_l2_to_relevance,
    )

