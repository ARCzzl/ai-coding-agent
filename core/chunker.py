
import logging
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


# ==================== 策略1: 递归字符分块 ====================

def recursive_split(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    递归字符分块 - 逐级尝试分隔符切分

    分隔符优先级: 段落 → 换行 → 句号 → 空格 → 字符

    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=。)", "(?<=！)", "(?<=？)", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=True,
    )

    chunks = splitter.split_documents(documents)

    # 过滤空块和太短的块
    chunks = [c for c in chunks if len(c.page_content.strip()) > 10]

    logger.info(
        f"递归分块完成: {len(documents)} 文档 → {len(chunks)} 块 "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ==================== 策略2: 语义分块 ====================

def semantic_split(
    documents: List[Document],
    embedding_model=None,
    percentile_threshold: float = 0.90,
) -> List[Document]:
    """
    语义分块 - 基于句子 embedding 相似度识别分块边界

    """
    from langchain_experimental.text_splitter import SemanticChunker

    if embedding_model is None:
        from langchain_ollama import OllamaEmbeddings

        embedding_model = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile_threshold,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"语义分块完成: {len(documents)} 文档 → {len(chunks)} 块"
    )
    return chunks



def parent_child_split(
    documents: List[Document],
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 256,
    parent_overlap: int = 128,
    child_overlap: int = 32,
) -> Tuple[List[Document], List[Document]]:
 
    import uuid

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        length_function=len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        length_function=len,
    )

    parent_chunks = parent_splitter.split_documents(documents)
    child_chunks_list = []

    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent.metadata["doc_id"] = parent_id

        # 将父块切分为子块，建立父子关联
        sub_chunks = child_splitter.split_documents([parent])
        for child in sub_chunks:
            child.metadata["parent_doc_id"] = parent_id
            # 保留父块的页面信息
            if "page" in parent.metadata:
                child.metadata["page"] = parent.metadata["page"]
            child_chunks_list.append(child)

    logger.info(
        f"父子分块完成: 父块={len(parent_chunks)}, 子块={len(child_chunks_list)}"
    )
    return parent_chunks, child_chunks_list


# ==================== 统一入口 ====================

def chunk_documents(
    documents: List[Document],
    strategy: str = "recursive",
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    统一分块入口

    Args:
        documents: 原始文档列表
        strategy: 分块策略
            - "recursive": 递归字符分块（默认，最稳定）
            - "semantic": 语义分块
            - "parent_child": 返回子块用于检索
        chunk_size: 块大小
        chunk_overlap: 块重叠

    Returns:
        分块后的 Document 列表
    """
    strategies = {
        "recursive": lambda: recursive_split(documents, chunk_size, chunk_overlap),
        "semantic": lambda: semantic_split(documents),
    }

    if strategy not in strategies:
        raise ValueError(
            f"不支持的分块策略: {strategy}，可选: {list(strategies.keys())}"
        )

    return strategies[strategy]()
