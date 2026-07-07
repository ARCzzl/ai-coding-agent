
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
