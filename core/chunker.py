"""
核心模块：智能分块器

提供多种分块策略，是 RAG 检索质量的关键环节：

策略1 - 递归字符分块 (RecursiveCharacterTextSplitter):
  适合大多数场景，按语义分隔符递归切分

策略2 - 语义分块 (SemanticChunker):
  基于 embedding 相似度判断分块边界，同一块的句子语义接近

策略3 - 父子文档分块 (Parent-Child / Small-to-Big):
  用小子块做检索，返回父块做上下文，兼顾检索精度与上下文完整性

面试加分点：理解不同分块策略的适用场景和 trade-off
"""

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

    这是最常用的分块策略，LangChain 默认推荐。
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

    原理：
    1. 将文本按句子拆分
    2. 计算相邻句子的 embedding 余弦相似度
    3. 相似度低于阈值的位置作为分块边界
    4. 保持语义连贯性，同一块的句子讨论同一主题

    优点：分块更符合语义逻辑，检索相关性更高
    缺点：计算成本高，需要 embedding 模型
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


# ==================== 策略3: 父子文档分块 ====================

def parent_child_split(
    documents: List[Document],
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 256,
    parent_overlap: int = 128,
    child_overlap: int = 32,
) -> Tuple[List[Document], List[Document]]:
    """
    父子文档分块 (Small-to-Big)

    策略说明：
    - 父块(paremt_chunks): 较大的上下文块，用于最终返回给 LLM
    - 子块(child_chunks): 较小的检索块，用于向量检索匹配

    流程：
    1. 检索时在子块中搜索（更精准的语义匹配）
    2. 返回对应的父块内容（更完整的上下文）

    为什么更好：
    - 小子块检索：与 query 的 embedding 更接近，检索更精准
    - 大父块返回：为 LLM 提供更充足的上下文，减少信息丢失

    论文参考：SmallToBig retrieval strategy (LangChain blog)
    """
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