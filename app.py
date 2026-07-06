"""
 领克汽车智能销售顾问 — CLI 交互入口
"""
import os
import argparse
import logging
import sys
from pathlib import Path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings


def setup_logging(level: str = None):
    """配置日志"""
    level = level or settings.LOG_LEVEL
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def interactive_chat():
    """
    交互式对话模式

    启动一个命令行对话循环，用户可以持续与 Agent 对话。
    输入 'exit' 退出，输入 'reset' 清除记忆，输入 'stats' 查看统计。
    """
    from agent.agent import create_agent

    logger = logging.getLogger("cli")
    logger.info("正在初始化 Agent，请稍候...")

    try:
        agent = create_agent()
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}")
        logger.error(
            "请检查:\n"
            "  1. .env 文件中 LLM_API_KEY 是否配置\n"
            "  2. Ollama 服务是否已启动\n"
            "  3. car_info.pdf 文件是否存在"
        )
        sys.exit(1)

    print("\n" + "=" * 50)
    print("   🚗 领克汽车智能销售顾问 — AI 助手「小领」")
    print("=" * 50)
    print("  输入 'exit' 退出 | 'reset' 清除记忆 | 'stats' 查看状态")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("👤 您: ").strip()

            if not question:
                continue

            if question.lower() == "exit":
                print("\n感谢咨询领克汽车，再见！")
                break

            if question.lower() == "reset":
                agent.reset_memory()
                print("对话记忆已清除\n")
                continue

            if question.lower() == "stats":
                stats = agent.get_memory_stats()
                print(f" 对话统计: {stats}\n")
                continue

            # 调用 Agent
            response = agent.chat(question)
            print(f"\n小领: {response}\n")

        except KeyboardInterrupt:
            print("\n\n 再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


def run_evaluation():
    """运行 RAG 系统评估"""
    from agent.agent import create_agent
    from evaluation.run_eval import run_full_evaluation

    logger = logging.getLogger("eval")
    logger.info("初始化 Agent 用于评估...")

    agent = create_agent()

    # 获取底层检索器
    from agent.tools import _rag_retriever

    logger.info("开始运行 RAG 评估...")
    summary = run_full_evaluation(_rag_retriever, agent.llm)

    print(summary.display())

    return summary


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="领克汽车智能销售顾问 RAG 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python app.py              # 启动交互式对话
  python app.py --eval       # 运行系统评估
  python app.py --log DEBUG  # 启用调试日志
        """,
    )
    parser.add_argument(
        "--eval", action="store_true", help="运行 RAG 系统评估"
    )
    parser.add_argument(
        "--log", default="INFO", help="日志级别 (DEBUG/INFO/WARNING/ERROR)"
    )

    args = parser.parse_args()
    setup_logging(args.log)

    # 启动前检查
    try:
        settings.validate() 
    except (ValueError, FileNotFoundError) as e:
        print(f"配置错误: {e}")
        print("请参考 .env.example 文件配置 .env")
        sys.exit(1)

    if args.eval:
        run_evaluation()
    else:
        interactive_chat()


if __name__ == "__main__":
    main()