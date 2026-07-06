"""
核心模块：文档加载器

支持多种文档格式的解析：
  - PDF: 使用 PyMuPDFLoader，保留页码和元数据
  - TXT/MD: 使用 TextLoader
  - 后续可扩展: DOCX, HTML, CSV, 图片OCR 等

设计要点：
  1. 文档清洗 - 去除乱码字符、多余空白、页眉页脚
  2. 元数据保留 - 页码、来源文件、加载时间
  3. 批量加载 - 支持加载整个目录
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    文本清洗：
    - 去除不可见控制字符（保留常见空白符）
    - 规范化空白（多空格→单空格，多换行→两个换行）
    - 去除 PDF 常见的乱码片段
    """
    # 去除不可见控制字符（保留 \n, \t, \r）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # 规范化空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除零宽字符
    text = re.sub(r"[​‌‍‎‏﻿]", "", text)
    return text.strip()


def load_pdf(file_path: str) -> List[Document]:
    """
    加载并清洗 PDF 文档

    使用 PyMuPDF 加载，保留每页元数据，并执行文本清洗。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

    logger.info(f"正在加载 PDF 文档: {file_path}")
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()

    # 清洗每个文档块
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source_file"] = os.path.basename(file_path)
        doc.metadata["loaded_at"] = datetime.now().isoformat()
        # 过滤空文档
    documents = [doc for doc in documents if len(doc.page_content) > 50]

    logger.info(f"PDF 加载完成，共 {len(documents)} 页")
    return documents


def load_text(file_path: str) -> List[Document]:
    """加载纯文本 / Markdown 文件"""
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents


def load_documents(
    file_path: Optional[str] = None, file_type: Optional[str] = None
) -> List[Document]:
    """
    统一文档加载入口

    Args:
        file_path: 文件路径，默认使用配置中的 PDF_PATH
        file_type: 文件类型（"pdf" / "txt"），默认根据后缀自动推断

    Returns:
        加载并清洗后的 Document 列表
    """
    path = file_path or settings.PDF_PATH
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if file_type is None:
        suffix = path_obj.suffix.lower()
    else:
        suffix = f".{file_type}"

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".txt", ".md", ".markdown"):
        return load_text(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")


if __name__ == "__main__":
    # 测试加载
    logging.basicConfig(level=logging.INFO)
    docs = load_documents()
    print(f"加载了 {len(docs)} 页文档")
    if docs:
        print(f"第一页前200字:\n{docs[0].page_content[:200]}")
        print(f"元数据: {docs[0].metadata}")