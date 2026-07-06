
import logging
from typing import Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel

from agent.tools import get_all_tools, set_rag_retriever
from core.retriever import create_hybrid_retriever
from core.reranker import build_compression_retriever
from core.llm import create_llm
from core.memory import create_memory
from config.settings import settings

logger = logging.getLogger(__name__)

# ==================== 系统提示词 ====================

SYSTEM_PROMPT = """你是「领克汽车」的智能销售顾问，名字叫「小领」。

## 你的身份
- 你是领克品牌的专业销售顾问，热情、专业、耐心
- 你对领克全系车型了如指掌，能解答产品、购车、用车等各类问题

## 工作原则
1. **知识优先**: 回答产品问题时，优先使用 car_info_rag 工具检索知识库
2. **精准计算**: 购车贷款相关问题时，使用 calculate_car_loan 工具
3. **对比分析**: 用户在两款车之间犹豫时，使用 compare_lynkco_models 工具
4. **诚实透明**: 不确定的信息要明确告知用户，不编造数据
5. **自然引导**: 在回答结束时，可以自然地询问用户是否需要进一步了解

## 禁止行为
- 不要编造车型参数、价格等具体数据
- 不要替用户做购买决定，只提供客观信息和建议
- 不要透露你是 AI 或机器人，保持"销售顾问"的人设

## 回复风格
- 专业但亲和，可以适度使用 emoji 增加亲切感
- 先给出核心信息，再补充细节
- 如果检索结果不完整，诚实告知并建议换个方式提问

现在，开始接待你的客户吧！"""


# ==================== Agent 构建 ====================

class LynkCoAgent:
    """
    领克汽车智能销售 Agent

    封装了完整的 RAG + Agent 流程：
    1. 初始化阶段：加载文档、构建检索器
    2. 对话阶段：Agent 自主决定工具调用
    3. 记忆管理：多轮对话上下文保持

    使用示例:
        agent = LynkCoAgent()
        agent.initialize()
        response = agent.chat("领克03落地多少钱")
        print(response)
    """

    def __init__(self, memory_strategy: str = "buffer"):
        """
        Args:
            memory_strategy: 记忆策略
                - "buffer": 完整缓冲（默认，适合短对话）
                - "window": 滑动窗口（适合长对话）
                - "summary": 摘要记忆（适合超长对话）
        """
        self.memory_strategy = memory_strategy
        self.llm: Optional[BaseChatModel] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.memory = None
        self.is_initialized = False

        logger.info(f"LynkCoAgent 创建: memory={memory_strategy}")

    def initialize(self) -> None:
        """
        初始化 Agent：加载文档 → 构建检索器 → 创建 Agent

        这是整个系统的初始化入口，应在应用启动时调用一次。
        """
        logger.info("=" * 50)
        logger.info("开始初始化领克汽车智能销售 Agent...")
        logger.info(f"配置: {settings.display()}")
        logger.info("=" * 50)

        # 1. 加载文档 + 分块
        from core.loader import load_documents
        from core.chunker import chunk_documents

        logger.info("步骤 1/5: 加载文档...")
        documents = load_documents()
        logger.info(f"  → 加载了 {len(documents)} 页文档")

        logger.info("步骤 2/5: 文档分块...")
        chunks = chunk_documents(documents, strategy="recursive")
        logger.info(f"  → 生成了 {len(chunks)} 个文本块")

        # 2. 构建检索器链
        logger.info("步骤 3/5: 构建检索器链...")
        hybrid_retriever = create_hybrid_retriever(chunks)
        compression_retriever = build_compression_retriever(hybrid_retriever)
        set_rag_retriever(compression_retriever)
        logger.info("  → 混合检索器 + Reranker 就绪")

        # 3. 创建 LLM
        logger.info("步骤 4/5: 初始化 LLM...")
        self.llm = create_llm()
        logger.info(f"  → LLM: {self.llm.model_name}")

        # 4. 创建记忆
        logger.info("步骤 5/5: 创建对话记忆...")
        self.memory = create_memory(strategy=self.memory_strategy)

        # 5. 组装 Agent
        tools = get_all_tools()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=settings.LOG_LEVEL == "DEBUG",
            handle_parsing_errors=True,
            max_iterations=5,  # 防止无限循环
            return_intermediate_steps=False,
        )

        self.is_initialized = True
        logger.info("=" * 50)
        logger.info("✅ Agent 初始化完成！可以开始对话。")
        logger.info("=" * 50)

    def chat(self, message: str) -> str:
        """
        进行一轮对话

        Args:
            message: 用户输入

        Returns:
            Agent 的回复文本
        """
        if not self.is_initialized:
            raise RuntimeError("Agent 尚未初始化，请先调用 initialize()")

        try:
            result = self.agent_executor.invoke({"input": message})
            return result.get("output", "抱歉，我暂时无法回复，请稍后再试。")
        except Exception as e:
            logger.error(f"Agent 对话异常: {e}", exc_info=True)
            return f"抱歉，系统出了一点问题: {str(e)}"

    def reset_memory(self):
        """清除对话记忆，开始新会话"""
        if self.memory:
            self.memory.clear()
            logger.info("对话记忆已清除")

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if not self.memory:
            return {"status": "no memory"}

        messages = self.memory.load_memory_variables({})
        history = messages.get("history", [])
        return {
            "strategy": self.memory_strategy,
            "message_count": len(history) if isinstance(history, list) else 0,
        }


# ==================== 工厂函数 ====================

def create_agent(memory_strategy: str = "buffer") -> LynkCoAgent:
    """
    工厂函数：创建并初始化 Agent

    这是推荐的快捷入口：
        agent = create_agent()
        response = agent.chat("你好")
    """
    agent = LynkCoAgent(memory_strategy=memory_strategy)
    agent.initialize()
    return agent