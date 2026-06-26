"""面向中文试卷的节标题感知文档分割器（REQ-05）。

按节标题（如 "一、现代文阅读"、"四、写作"）分割试卷，
然后在每个节内进行字符级子分块。
每个生成的文本块携带 ``section_title`` 元数据字段。

设计：ADR-005 —— 独立模块，与
``RecursiveCharacterTextSplitter.create_documents()`` 接口兼容。
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 匹配顶层中文节标题：
#   一、现代文阅读  /  二．填空题  /  三.选择题
# 不匹配子节标记如 （一）或 (1)。
SECTION_PATTERN = re.compile(
    r"^([一二三四五六七八九十]+[、.．]\s*.+)",
    re.MULTILINE,
)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200


class SectionAwareSplitter:
    """按节标题分割文本，然后在每个节内进行子分块。"""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    # ------------------------------------------------------------------
    # 公开 API（与 RecursiveCharacterTextSplitter 兼容）
    # ------------------------------------------------------------------

    def create_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> list[Document]:
        """将 *texts* 分割为节标题感知的文本块。

        参数与 ``RecursiveCharacterTextSplitter.create_documents`` 一致，
        可在 ``load_documents`` 中作为直接替换使用。
        """
        all_chunks: list[Document] = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            base_meta = metadatas[i] if metadatas else {}
            sections = self._split_into_sections(text)
            for title, body in sections:
                if not body.strip():
                    continue
                chunk_meta = {**base_meta, "section_title": title}
                chunks = self._splitter.create_documents(
                    texts=[body],
                    metadatas=[chunk_meta],
                )
                all_chunks.extend(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """按节标题将 *text* 分割为 ``(title, body)`` 对。

        未找到标题时返回单个 ``("", full_text)`` 条目。
        第一个标题前的前言文本会拼接到第一个节的正文前。
        """
        matches = list(SECTION_PATTERN.finditer(text))

        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()

        for idx, match in enumerate(matches):
            title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            # 将前言拼接到第一个节
            if idx == 0 and preamble:
                body = preamble + "\n" + body

            sections.append((title, body))

        return sections

