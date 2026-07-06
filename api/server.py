
import logging
import asyncio
from typing import Optional, AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)

# ==================== 数据模型 ====================


class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户输入消息", min_length=1)
    session_id: Optional[str] = Field(None, description="会话ID")


class ChatResponse(BaseModel):
    """对话响应"""
    reply: str = Field(..., description="Agent回复")
    session_id: Optional[str] = None
    model: str = settings.LLM_MODEL


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    model: str = settings.LLM_MODEL
    embedding: str = settings.EMBEDDING_MODEL


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="领克汽车智能销售顾问 API",
    description="基于 RAG + Agent 的领克汽车智能问答系统",
    version="2.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局状态 ====================

_agent = None


def get_agent():
    """获取全局 Agent 实例（懒加载）"""
    global _agent
    if _agent is None:
        from agent.agent import create_agent

        logger.info("首次请求，初始化 Agent...")
        _agent = create_agent()
    return _agent


# ==================== API 端点 ====================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通对话接口

    请求体:
        { "message": "领克03多少钱", "session_id": "optional" }
    """
    try:
        agent = get_agent()
        reply = await asyncio.to_thread(agent.chat, request.message)
        return ChatResponse(
            reply=reply,
            session_id=request.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Agent 未就绪: {e}")
    except Exception as e:
        logger.error(f"对话异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口 (SSE)

    使用 Server-Sent Events 逐步返回生成内容。
    前端接入示例:
        const eventSource = new EventSource('/chat/stream');
        eventSource.onmessage = (e) => console.log(e.data);
    """

    async def generate() -> AsyncIterator[str]:
        try:
            agent = get_agent()
            reply = await asyncio.to_thread(agent.chat, request.message)

            # 模拟流式输出（逐字发送）
            chunk_size = 5  # 每次发送5个字符
            for i in range(0, len(reply), chunk_size):
                chunk = reply[i : i + chunk_size]
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.02)  # 模拟打字效果

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式对话异常: {e}", exc_info=True)
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/memory")
async def clear_memory():
    """清除对话记忆"""
    agent = get_agent()
    agent.reset_memory()
    return {"message": "对话记忆已清除"}


@app.get("/memory/stats")
async def memory_stats():
    """获取记忆统计信息"""
    agent = get_agent()
    return agent.get_memory_stats()


@app.get("/")
async def root():
    """API 文档重定向"""
    return {
        "name": "领克汽车智能销售顾问 API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "health": "GET /health",
            "clear_memory": "DELETE /memory",
            "memory_stats": "GET /memory/stats",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )