"""图构建模块 —— 组装 Supervisor + 三条分支并编译。"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.academic import (
    academic_router,
    evaluate_hallucination,
    generate_answer,
    rag_retrieve,
    rewrite_query,
    should_retry_or_end,
    web_search,
)
from src.graph.emotional import emotional_response
from src.graph.plan_adversarial import (
    adv_rewrite_node,
    consensus_check_node,
    drafter_node,
    feedback_router,
    plan_output_node,
    plan_tweak_node,
    reviewer_academic_node,
    reviewer_emotional_node,
    route_after_hil,
    route_feedback,
    should_output_or_revise,
)
from src.graph.planner import gather_intel, search_policy
from src.graph.state import TutorState
from src.graph.supervisor import handle_unknown, route_by_intent, supervisor_node


def build_graph() -> StateGraph:
    """构建完整的 LangGraph StateGraph（未编译状态）。"""

    # 构建图实例
    graph = StateGraph(TutorState)

    # ── 节点注册 ──────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)

    # 子图 A —— 学科辅导（并行检索 + 回答生成）
    graph.add_node("academic_router", academic_router)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_hallucination", evaluate_hallucination)
    graph.add_node("rewrite_query", rewrite_query)

    # 学习规划（情报收集 → 扁平化对抗式规划）
    graph.add_node("search_policy", search_policy)
    graph.add_node("gather_intel", gather_intel)
    graph.add_node("drafter", drafter_node)
    graph.add_node("reviewer_academic", reviewer_academic_node)
    graph.add_node("reviewer_emotional", reviewer_emotional_node)
    graph.add_node("consensus_check", consensus_check_node)
    graph.add_node("adv_rewrite", adv_rewrite_node)
    graph.add_node("plan_output", plan_output_node)
    graph.add_node("feedback_router", feedback_router)
    graph.add_node("plan_tweak", plan_tweak_node)

    # 情绪支持
    graph.add_node("emotional_response", emotional_response)

    # 未知意图 / 超纲问题
    graph.add_node("handle_unknown", handle_unknown)

    # ── 边（流转关系） ──────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # 条件分支边
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,    # 根据意图分类路由
        {
            "academic": "academic_router",
            "planning": "search_policy",
            "emotional": "emotional_response",
            "unknown": "handle_unknown",
        },
    )

    # 学科辅导流程 —— Fan-out/Fan-in 并行检索
    graph.add_edge("academic_router", "rag_retrieve")
    graph.add_edge("academic_router", "web_search")

    # Fan-in: 两路汇聚到 generate_answer
    graph.add_edge("rag_retrieve", "generate_answer")
    graph.add_edge("web_search", "generate_answer")

    # 幻觉评估 + 重试循环
    graph.add_edge("generate_answer", "evaluate_hallucination")
    graph.add_conditional_edges(
        "evaluate_hallucination",
        should_retry_or_end,
        {
            "retry": "rewrite_query",
            "end": END,
        },
    )
    graph.add_edge("rewrite_query", "academic_router")

    # 规划流程: 政策搜索 → 情报收集 → 对抗循环 → 计划输出 → 结束
    graph.add_edge("search_policy", "gather_intel")
    graph.add_edge("gather_intel", "drafter")
    graph.add_edge("drafter", "reviewer_academic")
    graph.add_edge("drafter", "reviewer_emotional")
    graph.add_edge("reviewer_academic", "consensus_check")
    graph.add_edge("reviewer_emotional", "consensus_check")
    graph.add_conditional_edges(
        "consensus_check",
        should_output_or_revise,
        {
            "output": "plan_output",
            "revise": "adv_rewrite",
        },
    )
    graph.add_edge("adv_rewrite", "drafter")
    graph.add_conditional_edges(
        "plan_output",
        route_after_hil,
        {"end": END, "feedback": "feedback_router"},
    )
    graph.add_conditional_edges(
        "feedback_router",
        route_feedback,
        {"tweak": "plan_tweak", "rewrite": "drafter"},
    )
    graph.add_edge("plan_tweak", "plan_output")

    # 情绪支持 —— 直接结束
    graph.add_edge("emotional_response", END)

    # 未知意图 —— 直接结束
    graph.add_edge("handle_unknown", END)

    return graph


def get_compiled_graph(checkpointer=None):
    """构建并编译图实例，可直接调用。

    参数:
        checkpointer: 可选的 LangGraph 检查点管理器，用于状态持久化。
                      提供后，图会按 thread_id 保存/恢复状态。
    """
    return build_graph().compile(checkpointer=checkpointer)

