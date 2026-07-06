"""
核心模块：重排序器

初检索返回的文档虽然相关，但排序可能不精确。
重排序器使用更强的 Cross-Encoder 模型对候选文档进行精细排序。

Cross-Encoder vs Bi-Encoder:
- Bi-Encoder: query 和 doc 分别编码，速度快但交互弱 → 用于初检索
- Cross-Encoder: query 和 doc 联合编码，精度高但速度慢 → 用于重排序

典型的 RAG 检索流程:
    用户问题 → 初检索(Bi-Encoder, 召回100+) → 重排序(Cross-Encoder, 保留top3) → LLM生成
"""

import logging
from typing import List

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from config.settings import settings

logger = logging.getLogger(__name__)


def create_reranker(
    model_name: str = None, top_n: int = None
) -> CrossEncoderReranker:
    """
    创建 Cross-Encoder 重排序器

    Args:
        model_name: HuggingFace 模型名，推荐:
            - BAAI/bge-reranker-large (中文能力强)
            - BAAI/bge-reranker-v2-m3 (多语言)
            - maidalun1020/bce-reranker-base_v1 (中文专用)
        top_n: 重排序后保留的文档数

    注意：
    - 初检索召回 5-10 个候选，重排序后取 top 3
    - 这个 pipeline 比单纯增加初检索 K 值效果更好
    """
    model_name = model_name or settings.RERANKER_MODEL
    top_n = top_n or settings.RERANKER_TOP_N

    logger.info(f"加载 Reranker 模型: {model_name}")
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return reranker


def build_compression_retriever(
    base_retriever: BaseRetriever,
    reranker_model: str = None,
    top_n: int = None,
) -> ContextualCompressionRetriever:
    """
    构建带重排序的压缩检索器

    这是检索管道的最后一环：
    Hybrid Retriever (初检索) → Cross-Encoder Reranker (精排) → Top-N文档 → LLM

    Args:
        base_retriever: 基础检索器（混合检索器）
        reranker_model: 重排序模型
        top_n: 最终返回文档数
    """
    reranker = create_reranker(reranker_model, top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    logger.info(
        f"压缩检索器构建完成: 初检→Reranker({reranker_model})→Top{top_n}"
    )
    return compression_retriever


class FallbackRetriever(BaseRetriever):
    """
    带降级策略的检索器

    当主检索器返回空结果时，自动降级到备用检索策略：
    1. 尝试去掉重排序，直接用基础检索器
    2. 尝试只用 BM25 关键词检索
    3. 返回预先定义的兜底回答

    这体现了生产环境的鲁棒性设计思维。
    """

    primary_retriever: BaseRetriever
    fallback_retriever: BaseRetriever = None
    fallback_message: str = "抱歉，未找到相关信息，请换个方式提问。"

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 主检索
        docs = self.primary_retriever.invoke(query)

        if docs:
            return docs

        # 降级1：备用检索器
        if self.fallback_retriever:
            logger.warning(f"主检索返回空，降级到备用检索器: {query[:50]}")
            docs = self.fallback_retriever.invoke(query)
            if docs:
                return docs

        # 降级2：返回兜底消息
        logger.warning(f"所有检索策略均无结果: {query[:50]}")
        from langchain_core.documents import Document

        return [Document(page_content=self.fallback_message)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reranker = create_reranker()
    print(f"Reranker 创建成功: {reranker}")