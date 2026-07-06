"""
核心模块：LLM 封装

统一 LLM 调用的入口，负责：
1. 模型初始化（支持 OpenAI 兼容 API）
2. 流式/非流式调用
3. Token 统计
4. 重试与异常处理
"""

import logging
from typing import AsyncIterator

from langchain_openai import ChatOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


def create_llm(
    model: str = None,
    temperature: float = None,
    top_p: float = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """
    创建 LLM 实例

    使用 OpenAI 兼容 API，支持：
    - 阿里云百炼 (DashScope)
    - Ollama 本地模型
    - 任何兼容 OpenAI API 的服务

    Args:
        model: 模型名
        temperature: 温度参数 (0-2)，越高越随机
        top_p: 核采样参数
        streaming: 是否启用流式输出

    Temperature 选择建议:
    - 检索/RAG任务: 0.3-0.7 (需要事实准确性)
    - 创意写作: 0.8-1.2
    - 代码生成: 0.1-0.3
    - 对话: 0.7-1.0

    ⚠️ 注意: API Key 必须通过环境变量设置，严禁硬编码
    """
    if not settings.LLM_API_KEY:
        raise ValueError(
            "LLM_API_KEY 未设置！请在 .env 文件中配置，"
            "或设置环境变量 LLM_API_KEY"
        )

    llm = ChatOpenAI(
        model=model or settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        top_p=top_p if top_p is not None else settings.LLM_TOP_P,
        streaming=streaming,
        max_retries=2,  # 网络异常自动重试
        timeout=30,  # 30秒超时
    )

    logger.info(f"LLM 初始化: model={llm.model_name}, streaming={streaming}")
    return llm


async def stream_response(messages: list) -> AsyncIterator[str]:
    """
    流式生成响应

    用于 FastAPI SSE 接口，逐步返回生成内容，提升用户体验。
    """
    llm = create_llm(streaming=True)
    async for chunk in llm.astream(messages):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content:
            yield content