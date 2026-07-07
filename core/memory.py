
import logging
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)

from config.settings import settings

logger = logging.getLogger(__name__)


def create_buffer_memory(
    memory_key: str = "history",
    return_messages: bool = True,
) -> ConversationBufferMemory:
    """
    完整对话缓冲记忆

    """
    return ConversationBufferMemory(
        return_messages=return_messages,
        memory_key=memory_key,
    )


def create_window_memory(
    k: int = 6,
    memory_key: str = "history",
    return_messages: bool = True,
) -> ConversationBufferWindowMemory:
    """
    滑动窗口记忆

    return ConversationBufferWindowMemory(
        k=k,
        return_messages=return_messages,
        memory_key=memory_key,
    )


def create_summary_memory(
    llm=None,
    memory_key: str = "history",
    max_token_limit: int = 2000,
    return_messages: bool = True,
) -> ConversationSummaryMemory:
    """
    摘要记忆

    当对话变长时，用 LLM 对早期对话生成摘要，保留关键信息。
    结合"摘要 + 最近对话"，兼顾信息保留和 token 控制。

    """
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.3,  # 摘要需要准确，低温
        )

    return ConversationSummaryMemory(
        llm=llm,
        max_token_limit=max_token_limit,
        return_messages=return_messages,
        memory_key=memory_key,
    )


def create_memory(
    strategy: str = "buffer",
    **kwargs,
):
    """
    统一记忆创建入口

    Args:
        strategy: "buffer" | "window" | "summary"
        **kwargs: 传递给具体记忆类的参数
    """
    strategies = {
        "buffer": create_buffer_memory,
        "window": create_window_memory,
        "summary": create_summary_memory,
    }

    if strategy not in strategies:
        raise ValueError(
            f"不支持的记忆策略: {strategy}，可选: {list(strategies.keys())}"
        )

    logger.info(f"创建对话记忆: 策略={strategy}")
    return strategies[strategy](**kwargs)
