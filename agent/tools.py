"""
Agent 工具模块

定义了 Agent 可调用的所有工具函数：

1. car_info_rag:      领克汽车知识库检索
2. calculate_car_loan: 购车贷款计算器
3. compare_models:     车型对比（新增）

工具设计原则：
- 每个工具职责单一
- docstring 是 Agent 判断何时调用的唯一依据，必须清晰
- 返回值结构化，便于 Agent 理解
"""

import logging
from typing import Optional, List

from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

# ==================== 车型价格数据库 ====================

LYNKCO_PRICE_DB = {
    "领克01": 17.98,
    "领克02": 14.98,
    "领克03": 13.68,
    "领克05": 18.68,
    "领克06": 11.86,
    "领克09": 26.59,
    "领克01 EM-P": 19.98,
    "领克09 EM-P": 31.99,
    "领克03+": 19.88,
    "领克05+": 23.58,
}

# 车型简介，用于对比
LYNKCO_DESC = {
    "领克01": "紧凑型SUV，适合城市通勤和家庭使用",
    "领克02": "跨界SUV，运动风格，适合年轻用户",
    "领克03": "紧凑型轿车，操控性能优秀",
    "领克05": "轿跑SUV，造型运动，动力强劲",
    "领克06": "小型SUV，入门级，性价比高",
    "领克09": "中大型SUV，旗舰车型，7座可选",
    "领克01 EM-P": "插电混动SUV，省油环保",
    "领克09 EM-P": "旗舰插电混动SUV，豪华舒适",
    "领克03+": "高性能轿车，运动调校",
    "领克05+": "高性能轿跑SUV",
}


# ==================== 全局检索器引用 ====================
_rag_retriever: Optional[BaseRetriever] = None


def set_rag_retriever(retriever: BaseRetriever):
    """设置 RAG 检索器实例（在初始化时调用）"""
    global _rag_retriever
    _rag_retriever = retriever
    logger.info("RAG 检索器已注册到工具模块")


# ==================== 工具1: RAG 知识检索 ====================

@tool
def car_info_rag(query: str) -> str:
    """
    领克汽车专业知识检索工具。

    用于查询领克汽车的产品信息，包括：
    - 车型参数（尺寸、动力、油耗、配置等）
    - 车辆功能介绍（智能驾驶、安全配置、车机系统等）
    - 购车政策、保养维护、售后服务等

    当用户询问任何关于领克汽车的具体产品信息时，优先使用此工具。

    Args:
        query: 用户的查询问题，尽量保留原始表述
    """
    global _rag_retriever
    if _rag_retriever is None:
        return "⚠️ 知识库尚未初始化，请联系管理员。"

    try:
        docs = _rag_retriever.invoke(query)

        if not docs:
            return (
                "未在知识库中找到相关信息。\n"
                "建议：换个方式提问，或询问具体的车型参数。"
            )

        # 返回 top-3 文档，给 LLM 更充分的上下文
        results = []
        for i, doc in enumerate(docs[:3], 1):
            source = doc.metadata.get("source_file", "未知")
            page = doc.metadata.get("page", "N/A")
            results.append(
                f"[来源{i}] 页码{page} | {source}\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"RAG 检索异常: {e}")
        return f"检索服务暂时不可用: {str(e)}"


# ==================== 工具2: 贷款计算 ====================

@tool
def calculate_car_loan(
    car_model: str,
    down_payment_ratio: int = 30,
    loan_years: int = 3,
) -> str:
    """
    计算领克汽车分期购车方案。

    根据车型参考价、首付比例和贷款年限，计算月供和总利息。

    使用场景：
    - 用户询问"分期买领克03需要多少月供"
    - 用户想对比不同首付比例的方案
    - 用户询问落地价或购车总花费

    Args:
        car_model: 车型名称，如 "领克03"、"领克09 EM-P"等
        down_payment_ratio: 首付比例（%），默认30%，范围0-100
        loan_years: 贷款年限，默认3年，范围1-5年
    """
    # 参数校验
    if down_payment_ratio < 0 or down_payment_ratio > 100:
        return "首付比例应在 0-100% 之间。"
    if loan_years < 1 or loan_years > 5:
        return "贷款年限应在 1-5 年之间。"

    # 查找车型
    price = LYNKCO_PRICE_DB.get(car_model)
    if price is None:
        # 模糊匹配
        for name in LYNKCO_PRICE_DB:
            if car_model in name or name in car_model:
                car_model = name
                price = LYNKCO_PRICE_DB[name]
                break

    if price is None:
        available = "、".join(LYNKCO_PRICE_DB.keys())
        return (
            f"未找到车型「{car_model}」。\n"
            f"当前支持的车型: {available}\n"
            f"如车型名称不完全匹配，请尝试使用简称。"
        )

    # 贷款计算（等额本息）
    annual_rate = 0.045  # 年利率 4.5%
    down_payment = price * (down_payment_ratio / 100)
    loan_amount = price - down_payment

    if loan_amount <= 0:
        down_payment_display = f"{down_payment:.2f}"
        return (
            f"【{car_model} 全款购车】\n"
            f"参考车价：{price:.2f} 万元\n"
            f"首付 {down_payment_ratio}%（{down_payment_display} 万元）已覆盖全款，无需贷款。"
        )

    months = loan_years * 12
    monthly_rate = annual_rate / 12

    # 等额本息公式: M = P * r * (1+r)^n / ((1+r)^n - 1)
    monthly_payment = (
        loan_amount
        * monthly_rate
        * (1 + monthly_rate) ** months
        / ((1 + monthly_rate) ** months - 1)
    )

    monthly_payment_wy = monthly_payment
    total_payment = monthly_payment_wy * months
    total_interest = total_payment - loan_amount

    # 格式化输出
    result = (
        f"【{car_model} 贷款购车方案】\n"
        f"{'─' * 32}\n"
        f"💰 参考车价：{price:.2f} 万元\n"
        f"📌 首付 {down_payment_ratio}%：{down_payment:.2f} 万元\n"
        f"🏦 贷款金额：{loan_amount:.2f} 万元\n"
        f"📅 贷款年限：{loan_years} 年（年利率 {annual_rate * 100}%）\n"
        f"{'─' * 32}\n"
        f"💳 每月还款：{monthly_payment_wy:.2f} 元\n"
        f"📊 总利息：{total_interest:.2f} 元\n"
        f"💵 总花费：{down_payment + total_payment:.2f} 万元\n"
        f"{'─' * 32}\n"
        f"📝 注：以上为等额本息计算，实际以银行审批为准"
    )
    return result


# ==================== 工具3: 车型对比 ====================

@tool
def compare_lynkco_models(model_a: str, model_b: str) -> str:
    """
    对比两款领克车型的基本信息。

    用于用户在两款车之间犹豫时，给出参考对比。

    Args:
        model_a: 第一款车型名称
        model_b: 第二款车型名称
    """
    price_a = LYNKCO_PRICE_DB.get(model_a)
    price_b = LYNKCO_PRICE_DB.get(model_b)
    desc_a = LYNKCO_DESC.get(model_a, "暂无简介")
    desc_b = LYNKCO_DESC.get(model_b, "暂无简介")

    if not price_a:
        return f"未找到车型: {model_a}"
    if not price_b:
        return f"未找到车型: {model_b}"

    diff = abs(price_a - price_b)
    cheaper = model_a if price_a < price_b else model_b

    return (
        f"【{model_a} vs {model_b} 对比】\n"
        f"{'─' * 40}\n"
        f"🚗 {model_a}\n"
        f"   参考价: {price_a:.2f} 万元\n"
        f"   定位: {desc_a}\n"
        f"\n"
        f"🚗 {model_b}\n"
        f"   参考价: {price_b:.2f} 万元\n"
        f"   定位: {desc_b}\n"
        f"{'─' * 40}\n"
        f"💰 价差: {diff:.2f} 万元（{cheaper} 更实惠）\n"
        f"\n"
        f"📝 选择建议：\n"
        f"  - 预算优先 → 选择 {cheaper}\n"
        f"  - 配置/体验优先 → 建议实地试驾两款车后决定\n"
    )


# ==================== 获取所有工具 ====================

def get_all_tools() -> List:
    """获取所有可用工具"""
    return [car_info_rag, calculate_car_loan, compare_lynkco_models]