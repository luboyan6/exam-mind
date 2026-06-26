"""TutorState: 在 LangGraph 所有节点之间流转的共享状态对象，是系统唯一的数据源。"""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# 哨兵值：节点返回此值表示“清空所有上下文”
CONTEXT_CLEAR: list[dict] = [{"__clear__": True}]


def context_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """合并来自 Fan-out 分支的上下文列表。

    传入 CONTEXT_CLEAR 时将上下文重置为空（用于重试路径）。
    普通更新采用追加方式（与 operator.add 行为一致）。
    """
    if update and update[0].get("__clear__"):
        return []
    return existing + update


class TutorState(TypedDict):
    messages: Annotated[list, add_messages]                             # 对话历史
    intent: Literal["academic", "planning", "emotional", "unknown"]    # 用户意图分类
    subject: str                                                        # 当前讨论的学科/主题
    keypoints: list[str]                                                # 知识点列表
    context: Annotated[list[dict], context_reducer]                    # 合并后的检索上下文（Fan-in）
    search_results: list[dict]                                          # 规划器搜索结果
    plan: str                                                           # 生成的学习计划
    retry_count: int                                                    # 幻觉重试计数器
    hallucination_detected: bool                                        # 是否检测到幻觉
    rewritten_query: str                                                # 重试时改写后的查询
    hallucination_reason: str                                           # 幻觉评估给出的原因
    emotional_intel: str                                                # 情绪状态摘要（gather_intel 阶段）
    resource_intel: str                                                 # 资源情报摘要（gather_intel 阶段）
    intel_summary: str                                                  # 合并情报，供对抗式规划器使用
    # ── 对抗式规划（扁平化子图） ─────────────────────────────
    draft: str                                                          # 当前计划草稿文本
    academic_verdict: str                                               # 学术审查结论: "approve" / "reject"
    academic_reason: str                                                # 学术审查理由
    emotional_verdict: str                                              # 情绪审查结论: "approve" / "reject"
    emotional_reason: str                                               # 情绪审查理由
    adv_round: int                                                      # 当前审查轮次
    consensus: bool                                                     # 两个审查员是否全票通过
    revision_notes: str                                                 # 合并后的审查反馈，供起草者修改用
    # ── HIL 人工反馈循环 ──────────────────────────────────────
    hil_action: str                                                     # "confirm"（确认）或 "feedback"（反馈）
    hil_feedback: str                                                   # 用户原始反馈文本
    hil_summary: str                                                    # 所有历史反馈轮次的压缩摘要（覆写而非追加）
    feedback_route: str                                                 # "tweak"（微调）或 "rewrite"（重写）

