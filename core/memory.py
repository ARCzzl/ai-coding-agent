"""
核心模块：对话记忆管理

实现多种记忆策略，解决多轮对话中的上下文管理问题：

策略对比：
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│     策略          │   记忆范围    │    Token 消耗  │   适用场景    │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ BufferMemory     │ 全部对话     │ 线性增长     │ 短对话       │
│ WindowMemory     │ 最近 K 轮    │ 恒定         │ 长对话       │
│ SummaryMemory    │ 摘要 + 最近  │ 可控         │ 长对话+细节  │
│ TokenBufferMemory│ Token 限额   │ 上限控制     │ 生产环境     │
└──────────────────┴──────────────┴──────────────┴──────────────┘

面试要点：
- 理解"记忆"的本质是 prompt 中的上下文窗口管理
- 知道 token 限制对记忆策略的影响
- 了解 RAG 本身也是一种"外部记忆"
"""

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

    优点: 保留所有历史，信息完整
    缺点: Token 线性增长，长对话可能超出上下文窗口
    适用: 短对话（< 10 轮）
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

    只保留最近 K 轮对话，旧对话自动丢弃。
    优点: Token 消耗恒定
    缺点: 丢失早期的上下文信息
    适用: 客服场景、单次会话较长但不需要跨话题记忆
    """
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

    优点: 在 token 预算内保留更多有效信息
    缺点: 摘要过程有信息损失，依赖 LLM 摘要质量
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