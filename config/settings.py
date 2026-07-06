"""
配置管理模块

集中管理所有环境变量和配置参数，严禁硬编码 API Key。
使用方式:
    from config.settings import settings
    model = settings.LLM_MODEL
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # .env 不存在时尝试加载 .env.example 中的默认值
    load_dotenv(dotenv_path=None)  # 只从系统环境变量加载


class Settings:
    """全局配置单例"""

    # ========== LLM 配置 ==========
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.6-plus")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "1.1"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.8"))

    # ========== Embedding 配置 ==========
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ========== Reranker 配置 ==========
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "3"))

    # ========== 数据配置 ==========
    PDF_PATH: str = os.getenv("PDF_PATH", "./car_info.pdf")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "car_info")

    # ========== 分块配置 ==========
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "128"))
    SEMANTIC_CHUNKING_ENABLED: bool = (
        os.getenv("SEMANTIC_CHUNKING_ENABLED", "false").lower() == "true"
    )

    # ========== 检索配置 ==========
    BM25_K: int = int(os.getenv("BM25_K", "5"))
    VECTOR_K: int = int(os.getenv("VECTOR_K", "5"))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    MULTI_QUERY_COUNT: int = int(os.getenv("MULTI_QUERY_COUNT", "3"))

    # ========== API 配置 ==========
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

    # ========== 评估配置 ==========
    EVAL_DATASET_PATH: str = os.getenv(
        "EVAL_DATASET_PATH", "./data/eval_questions.json"
    )

    # ========== 日志配置 ==========
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        """校验关键配置项，启动时调用"""
        if not self.LLM_API_KEY:
            raise ValueError(
                "LLM_API_KEY 未配置！请在 .env 文件中设置，"
                "或设置环境变量 LLM_API_KEY。参考 .env.example"
            )
        if not Path(self.PDF_PATH).exists():
            raise FileNotFoundError(
                f"PDF 文件不存在: {self.PDF_PATH}。请将数据文件放入正确路径。"
            )

    def display(self) -> str:
        """打印配置摘要（隐藏敏感信息）"""
        return (
            f"LLM Model: {self.LLM_MODEL}\n"
            f"Embedding: {self.EMBEDDING_MODEL}\n"
            f"Reranker: {self.RERANKER_MODEL}\n"
            f"Chunk Size: {self.CHUNK_SIZE} | Overlap: {self.CHUNK_OVERLAP}\n"
            f"Vector K: {self.VECTOR_K} | BM25 K: {self.BM25_K}\n"
            f"BM25 Weight: {self.BM25_WEIGHT} | Vector Weight: {self.VECTOR_WEIGHT}"
        )


# 全局单例
settings = Settings()