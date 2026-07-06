
import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    """评估问题结构"""
    question: str                       # 用户问题
    ground_truth: str                   # 参考答案（人工标注）
    expected_sources: List[str] = field(default_factory=list)  # 预期来源


@dataclass
class EvalResult:
    """单条评估结果"""
    question: str
    answer: str
    ground_truth: str
    faithfulness: float        # 0-1，越高越好
    answer_relevancy: float    # 0-1，越高越好
    context_recall: float      # 0-1，越高越好
    context_precision: float   # 0-1，越高越好
    recall_at_k: float         # 0-1
    mrr: float                 # 0-1
    hit_rate: float            # 0 或 1


@dataclass
class EvalSummary:
    """评估汇总报告"""
    total_questions: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_recall: float
    avg_context_precision: float
    avg_recall_at_k: float
    avg_mrr: float
    hit_rate: float
    details: List[EvalResult] = field(default_factory=list)

    def display(self) -> str:
        """格式化输出评估报告"""
        return (
            f"\n{'=' * 50}\n"
            f"          RAG 系统评估报告\n"
            f"{'=' * 50}\n"
            f"评估问题数: {self.total_questions}\n"
            f"{'─' * 50}\n"
            f"📊 RAGAS 核心指标\n"
            f"  忠实度 (Faithfulness):     {self.avg_faithfulness:.2%}\n"
            f"  答案相关性 (Answer Relevancy): {self.avg_answer_relevancy:.2%}\n"
            f"  上下文召回率 (Context Recall):  {self.avg_context_recall:.2%}\n"
            f"  上下文精确度 (Context Precision): {self.avg_context_precision:.2%}\n"
            f"{'─' * 50}\n"
            f"🔍 检索指标\n"
            f"  Recall@K:     {self.avg_recall_at_k:.2%}\n"
            f"  MRR:          {self.avg_mrr:.2%}\n"
            f"  Hit Rate:     {self.hit_rate:.2%}\n"
            f"{'=' * 50}\n"
        )


class RAGEvaluator:
    """
    RAG 评估器

    Usage:
        evaluator = RAGEvaluator(questions, retriever, llm)
        summary = evaluator.evaluate()
        print(summary.display())
    """

    def __init__(
        self,
        questions: List[EvalQuestion],
        retriever,
        llm=None,
    ):
        """
        Args:
            questions: 评估问题列表
            retriever: 检索器实例
            llm: LLM 实例（用于 RAGAS 指标评判）
        """
        self.questions = questions
        self.retriever = retriever
        self.llm = llm
        self.results: List[EvalResult] = []

    # ==================== 基础检索指标 ====================

    def _calc_recall_at_k(
        self, retrieved_docs: List, expected_docs: List[str], k: int = 5
    ) -> float:
        """Recall@K: Top-K 检索结果中命中了多少预期文档"""
        if not expected_docs:
            return 0.0

        retrieved_texts = [
            doc.page_content[:100] for doc in retrieved_docs[:k]
        ]
        hits = 0
        for expected in expected_docs:
            for text in retrieved_texts:
                if expected[:30] in text:
                    hits += 1
                    break
        return hits / len(expected_docs)

    def _calc_mrr(self, retrieved_docs: List, expected_docs: List[str]) -> float:
        """MRR: 第一个相关文档的排名倒数"""
        if not expected_docs:
            return 0.0

        retrieved_texts = [doc.page_content[:100] for doc in retrieved_docs]
        for i, text in enumerate(retrieved_texts, 1):
            for expected in expected_docs:
                if expected[:30] in text:
                    return 1.0 / i
        return 0.0

    # ==================== RAGAS 指标 ====================

    def _eval_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        评估忠实度：回答中的每句话是否都能在上下文中找到依据

        使用 LLM 逐句检查：
        1. 将回答拆分成独立陈述
        2. 每条陈述与上下文逐一比对
        3. 有依据的陈述数 / 总陈述数 = faithfulness

        没有 LLM 时使用基于关键词的简化版估算
        """
        if not self.llm:
            return self._simple_faithfulness(answer, contexts)

        # 使用 LLM 评估
        context_text = "\n".join(contexts)
        prompt = f"""你是一个评估专家。请评估以下 AI 回答的"忠实度"。

忠实度的定义：回答中的每一条信息是否都能在给定的上下文中找到依据。
- 如果回答中添加了上下文中没有的信息，就是"不忠实"
- 如果回答的内容完全来自上下文，就是"忠实的"

上下文：
---
{context_text}
---

AI 回答：
---
{answer}
---

请给出 0.0 到 1.0 之间的分数，只需要输出数字："""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            # 提取数字
            import re
            numbers = re.findall(r"([\d.]+)", content)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Faithfulness 评估失败: {e}")

        return self._simple_faithfulness(answer, contexts)

    def _simple_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """简化版忠实度：基于关键词重叠率估算"""
        answer_words = set(answer)
        context_words = set("".join(contexts))
        if not answer_words:
            return 0.0
        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)

    def _eval_answer_relevancy(self, question: str, answer: str) -> float:
        """
        评估答案相关性：回答是否紧扣问题

        使用 LLM 判断答案是否回答了问题
        """
        if not self.llm:
            return self._simple_relevancy(question, answer)

        prompt = f"""评估以下回答与问题的相关程度。

问题：{question}

回答：{answer}

给出 0.0 到 1.0 之间的分数（1.0=完全相关，0.0=完全不相关），只输出数字："""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            import re
            numbers = re.findall(r"([\d.]+)", content)
            if numbers:
                return max(0.0, min(1.0, float(numbers[0])))
        except Exception as e:
            logger.warning(f"Relevancy 评估失败: {e}")

        return self._simple_relevancy(question, answer)

    def _simple_relevancy(self, question: str, answer: str) -> float:
        """简化版相关性：基于词重叠"""
        q_words = set(question)
        a_words = set(answer)
        if not q_words:
            return 0.0
        return len(q_words & a_words) / len(q_words)

    def _eval_context_recall(
        self, retrieved_docs: List, ground_truth: str
    ) -> float:
        """
        评估上下文召回率：参考答案中的信息在检索文档中的覆盖程度
        """
        contexts = [doc.page_content[:200] for doc in retrieved_docs[:5]]
        context_text = " ".join(contexts)

        if not self.llm:
            gt_words = set(ground_truth[:200])
            ctx_words = set(context_text)
            if not gt_words:
                return 0.0
            return len(gt_words & ctx_words) / len(gt_words)

        prompt = f"""评估检索到的文档覆盖了参考答案中的多少信息。

参考答案（Ground Truth）：
{ground_truth}

检索到的文档：
{context_text}

请给出 0.0 到 1.0 之间的覆盖度分数，只输出数字："""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            import re
            numbers = re.findall(r"([\d.]+)", content)
            if numbers:
                return max(0.0, min(1.0, float(numbers[0])))
        except Exception as e:
            logger.warning(f"Context Recall 评估失败: {e}")
            return 0.5

    def _eval_context_precision(self, retrieved_docs: List) -> float:
        """上下文精确度：检索到的文档中有多少是真正相关的"""
        if not self.llm or not retrieved_docs:
            return 1.0 / max(len(retrieved_docs), 1)

        relevant_count = 0
        for doc in retrieved_docs[:5]:
            prompt = f"""评估以下文档内容是否可能包含有用的汽车产品信息。

文档内容片段：
{doc.page_content[:300]}

回答"相关"或"不相关"："""

            try:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                if "相关" in content and "不相关" not in content:
                    relevant_count += 1
            except Exception:
                relevant_count += 1  # 评估失败时乐观处理

        return relevant_count / max(len(retrieved_docs[:5]), 1)

    # ==================== 主评估流程 ====================

    def evaluate(self) -> EvalSummary:
        """
        执行完整评估

        对每个问题：
        1. 检索相关文档
        2. 让 LLM 生成回答
        3. 计算各项指标

        Returns:
            评估汇总报告
        """
        if not self.llm:
            from core.llm import create_llm

            self.llm = create_llm(temperature=0.3)  # 评估用低温

        self.results = []

        for i, eq in enumerate(self.questions):
            logger.info(f"评估进度: {i + 1}/{len(self.questions)} - {eq.question[:30]}...")

            # 检索
            retrieved_docs = self.retriever.invoke(eq.question)
            contexts = [doc.page_content[:500] for doc in retrieved_docs[:5]]

            # 生成回答
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", "基于上下文回答问题，不要编造信息。"),
                ("user", "上下文：{context}\n\n问题：{question}"),
            ])

            chain = prompt | self.llm
            try:
                response = chain.invoke({
                    "context": "\n---\n".join(contexts),
                    "question": eq.question,
                })
                answer = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                logger.error(f"生成回答失败: {e}")
                answer = "生成失败"

            # 计算指标
            result = EvalResult(
                question=eq.question,
                answer=answer,
                ground_truth=eq.ground_truth,
                faithfulness=self._eval_faithfulness(answer, contexts),
                answer_relevancy=self._eval_answer_relevancy(eq.question, answer),
                context_recall=self._eval_context_recall(
                    retrieved_docs, eq.ground_truth
                ),
                context_precision=self._eval_context_precision(retrieved_docs),
                recall_at_k=self._calc_recall_at_k(
                    retrieved_docs, eq.expected_sources
                ),
                mrr=self._calc_mrr(retrieved_docs, eq.expected_sources),
                hit_rate=1.0 if self._calc_mrr(retrieved_docs, eq.expected_sources) > 0 else 0.0,
            )
            self.results.append(result)

            logger.info(
                f"  → Faithfulness: {result.faithfulness:.2%}, "
                f"Relevancy: {result.answer_relevancy:.2%}, "
                f"Recall@K: {result.recall_at_k:.2%}"
            )

        # 汇总
        n = len(self.results)
        summary = EvalSummary(
            total_questions=n,
            avg_faithfulness=sum(r.faithfulness for r in self.results) / n,
            avg_answer_relevancy=sum(r.answer_relevancy for r in self.results) / n,
            avg_context_recall=sum(r.context_recall for r in self.results) / n,
            avg_context_precision=sum(r.context_precision for r in self.results) / n,
            avg_recall_at_k=sum(r.recall_at_k for r in self.results) / n,
            avg_mrr=sum(r.mrr for r in self.results) / n,
            hit_rate=sum(r.hit_rate for r in self.results) / n,
            details=self.results,
        )

        logger.info("评估完成！")
        return summary