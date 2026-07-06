

import os
import sys
from typing import List
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
PDF_PATH = "./car_info.pdf"  # 汽车信息PDF文件路径
CHROMA_PERSIST_DIR = "./chroma_db"  # 向量数据库持久化目录
EMBEDDING_MODEL = "qwen3-embedding:0.6b"  # Ollama 嵌入模型
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 服务地址
# 领克车型参考价
LYNKCO_PRICE_DB = {
    "领克01": 17.98,
    "领克02": 14.98,
    "领克03": 13.68,
    "领克05": 18.68,
    "领克06": 11.86,
    "领克09": 26.59,
    "领克01 EM-P": 19.98,
    "领克09 EM-P": 31.99,
}
#  加载文档并分块
def load_and_split_documents(pdf_path: str) -> List:
    """加载 PDF 文档并进行智能分块"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"未找到PDF文件: {pdf_path}，请确保文件存在后再运行。")
    loader = PyMuPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=。！？ )", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
# 构建混合检索器 + 重排序
def build_compression_retriever(chunks: List):
    """
    构建带重排序的混合检索器
    """
    def chinese_tokenizer(text: str) -> List[str]:
        return [token for token in jieba.cut(text) if token.strip()]
    bm25_retriever = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=chinese_tokenizer
    )
    bm25_retriever.k = 5  # 设置返回候选数
    embedding_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    # 如果向量数据库已存在，则直接加载；否则从文档创建
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vector_db = Chroma(
            collection_name="car_info",
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_model
        )
    else:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="car_info",
            persist_directory=CHROMA_PERSIST_DIR
        )
    db_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    # 混合检索器 (加权组合)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, db_retriever],
        weights=[0.3, 0.7]
    )
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=1)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )
    return compression_retriever
_rag_retriever = None
@tool
def calculate_lynkco_loan(car_model: str, down_payment_ratio: int, loan_years: int) -> str:
    """
    计算领克汽车分期购车的月供和总利息。
    """
    # 查找车型价格
    price = LYNKCO_PRICE_DB.get(car_model)
    if price is None:
        return f"未找到车型。"
    #  贷款参数设置（假设年利率 4.5%）
    annual_rate = 0.045
    down_payment = price * (down_payment_ratio / 100)
    loan_amount = price - down_payment
    if loan_amount <= 0:
        return f"首付比例 {down_payment_ratio}% 已经覆盖全款，无需贷款。"
    months = loan_years * 12
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        monthly_payment = loan_amount / months
    else:
        monthly_payment = loan_amount * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
    total_payment = monthly_payment * months
    total_interest = total_payment - loan_amount
    # 格式化输出（单位转换为万元，保留两位小数）
    result = (
        f"【{car_model} 贷款购车方案】\n"
        f"参考车价：{price:.2f} 万元\n"
        f"首付 {down_payment_ratio}%：{down_payment:.2f} 万元\n"
        f"贷款金额：{loan_amount:.2f} 万元\n"
        f"贷款年限：{loan_years} 年（年利率 4.5%）\n"
        f"每月还款：{monthly_payment:.2f} 元\n"
        f"总利息：{total_interest:.2f} 元\n"
        f"总花费（首付+贷款本息）：{down_payment + total_payment:.2f} 元"
    )
    return result
@tool
def car_info_rag(query: str) -> str:
    """
    领克汽车信息检索工具。
    """
    global _rag_retriever
    if _rag_retriever is None:
        return "检索器未初始化。"

    try:
        # 执行检索
        docs = _rag_retriever.invoke(query)
        if not docs:
            return "未找到相关的信息。"
        # 返回最相关文档的内容
        return docs[0].page_content
    except Exception as e:
        return f"检索错误: {str(e)}"
# 初始化全局 RAG 检索器
def initialize_rag_system():
    """初始化整个 RAG 系统（加载文档、构建检索器）"""
    global _rag_retriever
    chunks = load_and_split_documents(PDF_PATH)
    _rag_retriever = build_compression_retriever(chunks)
#  构建 Agent (支持多轮对话 + 工具调用)
def build_agent():
    """构建具备多轮记忆和工具调用的 Agent"""
    llm = ChatOpenAI(
        model="qwen3.6-plus",
        api_key="YOUR_API_KEY_PLACEHOLDER",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        top_p=0.8,
        temperature=1.1
    )
    tools = [car_info_rag, calculate_lynkco_loan]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是领克汽车销售顾问。需要具体资料时使用 car_info_rag 工具；计算贷款时使用 calculate_lynkco_loan 工具。"),
        MessagesPlaceholder(variable_name="history"),  # 多轮对话历史
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")  # Agent 中间推理步骤
    ])
    memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # 打印 Agent 思考过程，便于调试
        handle_parsing_errors=True
    )
    return agent_executor
# 主函数：交互式对话循环
while True:
    question = input("请输入问题:")
    if question == "exit":
        break
    res = agent_executor.invoke({"input":question})
    print(res["output"])
if __name__ == "__main__":
    main()