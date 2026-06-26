"""文档加载器：PDF / Markdown / TXT → 分块 LangChain Document。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _guess_year(filename: str) -> Optional[str]:
    """尝试从文件名中提取 4 位数字年份。"""
    m = re.search(r"(20\d{2})", filename)
    return m.group(1) if m else None


def _read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_READERS = {
    ".pdf": _read_pdf,
    ".md": _read_text,
    ".txt": _read_text,
}


def load_documents(
    data_dir: str | Path,
    subject: str,
    doc_type: str = "exam",
    splitter=None,
) -> list[Document]:
    """加载 *data_dir* 下所有支持的文件并分块。

    每个文本块携带元数据 ``{subject, source_file, year, doc_type}``。

    参数
    ----------
    splitter : 可选
        具有 ``create_documents(texts, metadatas)`` 方法的文本分割器。
        为 *None*（默认）时使用内置的 ``RecursiveCharacterTextSplitter``。
        试卷文档请传入 ``SectionAwareSplitter``。
    """
    active_splitter = splitter if splitter is not None else _splitter

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []
    for filepath in sorted(data_dir.iterdir()):
        ext = filepath.suffix.lower()
        reader = _READERS.get(ext)
        if reader is None:
            continue

        raw_text = reader(filepath)
        if not raw_text.strip():
            continue

        metadata = {
            "subject": subject,
            "source_file": filepath.name,
            "year": _guess_year(filepath.name) or "unknown",
            "doc_type": doc_type,
        }

        chunks = active_splitter.create_documents(
            texts=[raw_text],
            metadatas=[metadata],
        )
        documents.extend(chunks)

    return documents

