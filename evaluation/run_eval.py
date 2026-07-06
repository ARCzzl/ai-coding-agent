"""
评估执行脚本

定义评估问题集并执行完整的 RAG 系统评估。

这个脚本体现了"实验驱动开发"的思想：
- 不是拍脑袋调参数，而是用指标衡量效果
- 每次修改策略后重新评估，对比改进幅度
"""

import logging
from typing import List
import json
import os
from pathlib import Path

from evaluation.metrics import EvalQuestion, RAGEvaluator, EvalSummary
from config.settings import settings

logger = logging.getLogger(__name__)


def load_eval_questions(file_path: str = None) -> List[EvalQuestion]:
    """
    加载评估问题集

    优先从 JSON 文件加载，如果文件不存在则使用内置问题集。
    内置问题集覆盖了 RAG 系统的典型使用场景：
    - 精确查询：具体参数
    - 开放性问题：选购建议
    - 多文档关联：需要跨段落整合信息
    """
    file_path = file_path or settings.EVAL_DATASET_PATH

    if os.path.exists(file_path):
        logger.info(f"从文件加载评估问题: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            EvalQuestion(**item)
            for item in data
        ]

    # 内置评估问题集
    logger.info("使用内置评估问题集")
    return [
        EvalQuestion(
            question="领克03的发动机参数是什么？",
            ground_truth="领克03提供多种动力总成：1.5TD 三缸涡轮增压发动机（最大功率132kW，最大扭矩265N·m）和 2.0TD 四缸涡轮增压发动机（最大功率140kW/187kW，最大扭矩300N·m/350N·m）",
            expected_sources=["发动机", "动力", "功率", "扭矩"],
        ),
        EvalQuestion(
            question="领克01有哪些安全配置？",
            ground_truth="领克01配备City Safety城市安全系统、Pilot Assist领航辅助系统、Lane Keeping Aid车道保持辅助、BLIS盲点监测、360°全景影像、ACC自适应巡航、AEB主动刹车等多项主被动安全配置",
            expected_sources=["安全配置", "City Safety", "辅助系统"],
        ),
        EvalQuestion(
            question="领克09和领克05有什么区别？",
            ground_truth="领克09是中大型SUV（旗舰7座），领克05是紧凑型轿跑SUV；09尺寸更大、配置更高、价格更贵（26.59万起 vs 18.68万起）；09提供7座和6座版本，05为5座轿跑造型",
            expected_sources=["领克09", "领克05", "SUV", "尺寸"],
        ),
        EvalQuestion(
            question="领克06适合什么样的用户？",
            ground_truth="领克06适合年轻首购用户和预算有限的消费者，起售价11.86万元，定位小型SUV，造型时尚动感，空间适用，配置丰富，性价比高",
            expected_sources=["领克06", "11.86万", "小型SUV"],
        ),
        EvalQuestion(
            question="领克品牌的售后服务政策是怎样的？",
            ground_truth="领克提供终身免费质保、终身免费道路救援、终身免费数据流量三项终身免费服务；此外还有首保免费、定期保养提醒等服务",
            expected_sources=["售后", "质保", "免费", "保障"],
        ),
    ]


def run_full_evaluation(retriever, llm=None) -> EvalSummary:
    """
    执行完整评估

    Args:
        retriever: 检索器实例
        llm: LLM 实例（可选）
    """
    questions = load_eval_questions()
    logger.info(f"加载了 {len(questions)} 个评估问题")

    evaluator = RAGEvaluator(
        questions=questions,
        retriever=retriever,
        llm=llm,
    )

    summary = evaluator.evaluate()

    # 保存评估结果
    output_dir = Path("./evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"

    results_data = {
        "timestamp": timestamp,
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "vector_k": settings.VECTOR_K,
            "bm25_k": settings.BM25_K,
            "reranker_top_n": settings.RERANKER_TOP_N,
        },
        "summary": {
            "total_questions": summary.total_questions,
            "avg_faithfulness": summary.avg_faithfulness,
            "avg_answer_relevancy": summary.avg_answer_relevancy,
            "avg_context_recall": summary.avg_context_recall,
            "avg_context_precision": summary.avg_context_precision,
            "avg_recall_at_k": summary.avg_recall_at_k,
            "avg_mrr": summary.avg_mrr,
            "hit_rate": summary.hit_rate,
        },
        "details": [
            {
                "question": r.question,
                "answer": r.answer,
                "ground_truth": r.ground_truth,
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "context_recall": r.context_recall,
                "context_precision": r.context_precision,
                "recall_at_k": r.recall_at_k,
                "mrr": r.mrr,
            }
            for r in summary.details
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    logger.info(f"评估结果已保存到: {output_file}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 用法示例（需要先初始化检索器）
    print(load_eval_questions())