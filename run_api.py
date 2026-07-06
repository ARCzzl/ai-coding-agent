"""
============================================================
 领克汽车智能销售顾问 — API 服务入口
============================================================

使用方式:
    python run_api.py                     # 默认端口 8000
    python run_api.py --port 8080         # 指定端口
    python run_api.py --host 127.0.0.1    # 仅本地访问

前端接入示例:
    fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: '领克03多少钱'})
    }).then(r => r.json()).then(console.log)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="领克汽车智能销售顾问 API 服务")
    parser.add_argument("--host", default=settings.API_HOST, help="绑定地址")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="绑定端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    # 启动前校验
    try:
        settings.validate()
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ 配置错误: {e}")
        print("请参考 .env.example 文件配置 .env")
        sys.exit(1)

    print(f"\n🚀 启动领克汽车智能销售顾问 API 服务")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    print(f"  模型: {settings.LLM_MODEL}\n")

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()