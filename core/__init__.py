"""
core 模块：RAG 系统的核心组件

包含：
- loader:   文档加载与清洗
- chunker:  智能分块策略
- retriever: 高级检索（稠密+稀疏+查询重写）
- reranker:  Cross-Encoder 重排序
- llm:      LLM 统一封装
- memory:   对话记忆管理
"""

from core.loader import load_documents, clean_text
from core.chunker import chunk_documents, recursive_split, semantic_split
from core.retriever import (
    create_dense_retriever,
    create_sparse_retriever,
    create_hybrid_retriever,
    rewrite_query_multi_perspective,
    multi_query_retrieve,
)
from core.reranker import create_reranker, build_compression_retriever
from core.llm import create_llm, stream_response
from core.memory import create_memory

__all__ = [
    "load_documents",
    "clean_text",
    "chunk_documents",
    "recursive_split",
    "semantic_split",
    "create_dense_retriever",
    "create_sparse_retriever",
    "create_hybrid_retriever",
    "rewrite_query_multi_perspective",
    "multi_query_retrieve",
    "create_reranker",
    "build_compression_retriever",
    "create_llm",
    "stream_response",
    "create_memory",
]