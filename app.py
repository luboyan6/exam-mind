"""ExamMind — 面向考试备考场景的多智能体 AI 辅导系统。"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv(Path(__file__).parent / ".env")

from src.database.checkpointer import get_db_uri, make_thread_config
from src.graph.builder import get_compiled_graph
from src.schemas import ChatRequest, ResumeRequest
from src.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理异步资源：链路追踪、PostgreSQL 检查点、图实例。"""
    setup_tracing()

    async with AsyncExitStack() as stack:
        checkpointer = None
        db_uri = get_db_uri()

        if db_uri:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(db_uri)
                )
                await checkpointer.setup()
                logger.info("PostgreSQL checkpointer initialized")
            except Exception:
                logger.exception(
                    "Failed to initialize PostgreSQL checkpointer, running stateless"
                )
                checkpointer = None
        else:
            logger.info("DB_URI not set, running without persistent state")

        app.state.graph = get_compiled_graph(checkpointer=checkpointer)
        yield

    shutdown_tracing()


app = FastAPI(title="ExamMind API", lifespan=lifespan)

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_NODES = {"generate_answer", "drafter", "plan_tweak", "emotional_response"}

# 非流式节点集合 —— 这些节点的最终 AIMessage 内容会以 "text" SSE 事件推送
TEXT_EMIT_NODES = {"plan_output", "handle_unknown"}

# 所有需要广播生命周期（启动/结束）事件到前端的图节点
GRAPH_NODES = {
    "supervisor",
    "academic_router",
    "rag_retrieve",
    "web_search",
    "generate_answer",
    "evaluate_hallucination",
    "rewrite_query",
    "search_policy",
    "gather_intel",
    "drafter",
    "reviewer_academic",
    "reviewer_emotional",
    "consensus_check",
    "adv_rewrite",
    "plan_output",
    "feedback_router",
    "plan_tweak",
    "emotional_response",
    "handle_unknown",
}


async def _stream_graph_events(
    graph,
    input_data,
    config: dict,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """/stream 和 /resume 共用的 SSE 事件流处理逻辑。

    解析 astream_events 并产出节点生命周期、Token 流式输出、
    用量统计及中断等 SSE 载荷。
    """
    node_start_times: dict[str, float] = {}

    try:
        async for event in graph.astream_events(input_data, config=config, version="v2"):
            event_type = event["event"]

            # ── 节点生命周期事件 ──────────────────────────────────────────
            if event_type in ("on_chain_start", "on_chain_end"):
                node_name = event.get("name")
                meta_node = event.get("metadata", {}).get("langgraph_node")
                # 仅广播顶层图节点（name 与 metadata 匹配），
                # 忽略内部子链（如 RunnableSequence 等）
                if node_name and node_name == meta_node and node_name in GRAPH_NODES:
                    if event_type == "on_chain_start":
                        node_start_times[node_name] = time.monotonic()
                        payload = json.dumps(
                            {"type": "node_event", "status": "start", "node": node_name},
                            ensure_ascii=False,
                        )
                    else:
                        duration_ms = None
                        start_t = node_start_times.pop(node_name, None)
                        if start_t is not None:
                            duration_ms = round((time.monotonic() - start_t) * 1000)

                        error = None
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and output.get("error"):
                            error = str(output["error"])

                        payload = json.dumps(
                            {
                                "type": "node_event",
                                "status": "end",
                                "node": node_name,
                                "duration_ms": duration_ms,
                                "error": error,
                            },
                            ensure_ascii=False,
                        )
                    yield f"data: {payload}\n\n"

                    # 为非流式节点推送 "text" 事件（完整输出）
                    if event_type == "on_chain_end" and node_name in TEXT_EMIT_NODES:
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict):
                            for msg in output.get("messages", []):
                                if hasattr(msg, "content") and msg.content:
                                    text_payload = json.dumps(
                                        {"type": "text", "content": msg.content, "node": node_name},
                                        ensure_ascii=False,
                                    )
                                    yield f"data: {text_payload}\n\n"

            # ── Token 流式输出 ─────────────────────────────────────────────
            elif event_type == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name in ALLOWED_NODES:
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        payload = json.dumps(
                            {"type": "token", "content": chunk.content},
                            ensure_ascii=False,
                        )
                        yield f"data: {payload}\n\n"

            # ── Token 用量事件 ─────────────────────────────────────────────
            elif event_type == "on_chat_model_end":
                node_name = event.get("metadata", {}).get("langgraph_node")
                output = event.get("data", {}).get("output")
                usage = getattr(output, "usage_metadata", None)
                if usage and node_name:
                    payload = json.dumps(
                        {
                            "type": "usage",
                            "node": node_name,
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
    except Exception as e:
        logger.exception("Unhandled error in graph streaming")
        error_payload = json.dumps(
            {"type": "error", "message": str(e)},
            ensure_ascii=False,
        )
        yield f"data: {error_payload}\n\n"
        return

    # ── 流结束后检查是否存在中断（HIL 挂起状态） ──────────────────
    state_snapshot = await graph.aget_state(config)
    if state_snapshot.next:
        for task in state_snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                draft = task.interrupts[0].value
                payload = json.dumps(
                    {"type": "interrupt", "draft": draft, "thread_id": thread_id},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
                return

    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


async def generate_sse(
    query: str,
    graph,
    thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """将 LangGraph 事件以 Server-Sent Events（SSE）形式流式输出。
    
    产出的 SSE 载荷类型包括：
    
    * ``{"type": "thread_id", "thread_id": "..."}``  
      — 流开始时发出一次，供前端调用 /resume 使用。
    * ``{"type": "node_event", "status": "start"|"end", "node": "<name>"}``  
      — 图节点开始或结束执行时发出。
    * ``{"type": "token", "content": "<text>"}``  
      — 允许的 LLM 节点每次产出 Token 时发出。
    * ``{"type": "interrupt", "draft": "...", "thread_id": "..."}``  
      — 图因人工审查（HIL）暂停时发出。
    
    参数:
        query: 用户输入的待处理文本。
        graph: 已编译的 LangGraph 实例（来自 app.state）。
        thread_id: 可选的会话 ID，用于多轮对话记忆；为 None 时自动生成。
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = make_thread_config(thread_id)
    state_input = {"messages": [HumanMessage(content=query)]}

    # 推送 thread_id，以便前端用于 /resume 请求
    yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

    async for chunk in _stream_graph_events(graph, state_input, config, thread_id):
        yield chunk


async def generate_resume_sse(
    edited_plan: str,
    feedback: str | None,
    graph,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """恢复被中断的图执行，并以 SSE 形式流式输出后续事件。

    参数:
        edited_plan: 用户编辑后的计划文本，用于恢复执行。
        feedback: 可选的反馈文本，用于 AI 驱动的计划修订。
        graph: 已编译的 LangGraph 实例（来自 app.state）。
        thread_id: 会话 ID，标识被中断的图状态。
    """
    config = make_thread_config(thread_id)

    if feedback:
        resume_value = {"action": "feedback", "text": feedback}
    else:
        resume_value = edited_plan

    resume_input = Command(resume=resume_value)

    async for chunk in _stream_graph_events(graph, resume_input, config, thread_id):
        yield chunk


@app.post("/stream")
async def stream_endpoint(chat: ChatRequest, request: Request):
    return StreamingResponse(
        generate_sse(chat.query, request.app.state.graph, thread_id=chat.thread_id),
        media_type="text/event-stream",
    )


@app.post("/resume")
async def resume_endpoint(req: ResumeRequest, request: Request):
    return StreamingResponse(
        generate_resume_sse(req.edited_plan, req.feedback, request.app.state.graph, req.thread_id),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

