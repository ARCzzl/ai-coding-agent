
import logging
from typing import List, Optional

import jieba
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.settings import settings

logger = logging.getLogger(__name__)


def chinese_tokenizer(text: str) -> List[str]:
    """中文分词器：使用 jieba 进行精确模式分词"""
    return [token for token in jieba.cut(text) if token.strip()]


# ==================== 稠密检索器 ====================

def create_dense_retriever(
    chunks: List[Document],
    embedding_model=None,
    top_k: int = None,
    persist_dir: str = None,
) -> BaseRetriever:
    """
    创建稠密检索器（向量检索）

    使用 ChromaDB 作为向量数据库，Ollama Embedding 生成向量。

    关键参数：
    - top_k: 召回数量，越大召回越全但噪音越多
    - embedding_model: 向量化模型，决定语义理解质量
    """
    top_k = top_k or settings.VECTOR_K
    persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR

    if embedding_model is None:
        embedding_model = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    # 持久化加载 or 新建
    import os

    chroma_sqlite = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(chroma_sqlite) and os.path.getsize(chroma_sqlite) > 0:
        logger.info(f"从 {persist_dir} 加载已有向量数据库")
        vector_db = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
    else:
        logger.info("创建新的向量数据库")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=persist_dir,
        )

    return vector_db.as_retriever(search_kwargs={"k": top_k})


# ==================== 稀疏检索器 ====================

def create_sparse_retriever(
    chunks: List[Document], top_k: int = None
) -> BM25Retriever:
    """
    创建稀疏检索器（BM25）

    """
    top_k = top_k or settings.BM25_K

    bm25 = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=chinese_tokenizer,
    )
    bm25.k = top_k
    logger.info(f"BM25 检索器创建完成，k={top_k}")
    return bm25


# ==================== 混合检索器 ====================

def create_hybrid_retriever(
    chunks: List[Document],
    bm25_weight: float = None,
    vector_weight: float = None,
) -> EnsembleRetriever:
    """
    创建混合检索器
    融合策略：加权线性组合 (weighted linear combination)
    - Dense (向量检索): 高权重(0.7) → 语义理解为主
    - Sparse (BM25):     低权重(0.3) → 关键词兜底

    """
    bm25_weight = bm25_weight or settings.BM25_WEIGHT
    vector_weight = vector_weight or settings.VECTOR_WEIGHT

    bm25_retriever = create_sparse_retriever(chunks)
    dense_retriever = create_dense_retriever(chunks)

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, vector_weight],
    )

    logger.info(
        f"混合检索器创建完成: BM25({bm25_weight}) + Vector({vector_weight})"
    )
    return ensemble


# ==================== 查询重写 ====================

def rewrite_query_multi_perspective(
    query: str, llm=None, num_queries: int = None
) -> List[str]:
    """
    多视角查询重写 (Multi-Query Rewriting)
    原理：
    - 用户问题可能只有一种表达方式
    - 文档可能用不同的措辞描述同一内容
    - 用 LLM 生成多个等价表述，分别检索，合并结果
    显著提高召回率，但增加了检索次数
    """
    num_queries = num_queries or settings.MULTI_QUERY_COUNT

    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.3,  # 低温度，保持改写一致性
        )

    rewrite_prompt = f"""你是一个查询改写助手。用户的原始问题是关于领克汽车的。

请从不同角度生成 {num_queries} 个语义等价但表达不同的查询：
- 可以换用不同的关键词和句式
- 可以补充相关的细节
- 保持原始意图不变

原始问题: {query}

请严格按照以下格式输出，每行一个查询，不要编号：
"""

    try:
        response = llm.invoke(rewrite_prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # 解析返回的多个查询
        rewritten = []
        for line in content.strip().split("\n"):
            line = line.strip()
            # 去除可能的编号前缀
            import re

            line = re.sub(r"^\d+[\.\)、]\s*", "", line)
            if line and line != query and len(line) > 3:
                rewritten.append(line)

        # 去重 + 保留原始查询
        seen = {query}
        unique = [query]
        for q in rewritten:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        logger.info(f"查询重写: '{query[:30]}...' → {len(unique)} 个变体")
        return unique[: num_queries + 1]

    except Exception as e:
        logger.warning(f"查询重写失败: {e}，使用原始查询")
        return [query]


def multi_query_retrieve(
    query: str, retriever: BaseRetriever, llm=None
) -> List[Document]:
    """
    多查询检索：生成多个查询变体 → 分别检索 → 去重合并

    这是提升召回率的经典技巧，尤其适合：
    - 用户问题简短、模糊
    - 文档内容使用了不同术语
    """
    queries = rewrite_query_multi_perspective(query, llm)

    all_docs = []
    seen_contents = set()

    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            # 基于内容去重
            fingerprint = doc.page_content[:100]
            if fingerprint not in seen_contents:
                seen_contents.add(fingerprint)
                all_docs.append(doc)

    logger.info(f"多查询检索: {len(queries)} 查询 → {len(all_docs)} 去重文档")
    return all_docs
