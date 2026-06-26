# ExamMind 项目介绍

## 一、项目概述

**ExamMind**（考试心灵）是一套面向考试备考场景的**生产级多智能体AI辅导系统**。项目以高考备考为核心场景，通过多智能体协作架构，为学生提供学科答疑、学习规划、情绪疏导三位一体的智能辅导服务。

当前版本：**v0.4.0**

> 项目持续迭代中，欢迎参与讨论与贡献。

---

## 二、核心功能

### 2.1 学科答疑（Academic）
- **混合RAG检索**：向量检索 + BM25关键词检索 + BGE精排序，实现多路并行召回
- **幻觉评估与自动重试**：生成回答后自动评估忠实度，检测到幻觉时自动改写查询并重新检索，最多重试2次
- **网络搜索补充**：DuckDuckGo实时搜索作为知识库补充
- **LLM容灾机制**：主API异常时自动切换备用模型

### 2.2 学习规划（Planning）
- **对抗式多智能体博弈**：
  - **起草者（Drafter）**：生成学习计划草案
  - **学术审查员**：评估计划的学术质量
  - **情绪审查员**：评估计划的情绪关怀 adequacy
  - **共识检查**：双审查员全票通过才放行，否则打回重写
- **人工介入（HIL）**：对抗收敛后图执行挂起，用户可直接编辑计划或提供反馈
- **反馈路由**：智能判定反馈属于"微调"还是"重写"，微调快速局部修改，重写清空草稿走完整对抗循环
- **单摘要防膨胀**：多轮反馈只保留一条压缩摘要，杜绝上下文无限增长

### 2.3 情绪疏导（Emotional）
- 以资深班主任人设，基于对话历史给出兼具温度与实用性的情绪支持建议

### 2.4 意图路由（Supervisor）
- Qwen2.5-7B轻量级模型进行意图分类
- 单次LLM调用完成：意图分类 + 学科检测 + 知识点提取
- 支持四类意图：学科答疑、学习规划、情绪疏导、未知意图

---

## 三、系统架构

### 3.1 架构概览

```
                    ┌─────────────────────────────────────────┐
                    │              用户输入                      │
                    └─────────────────┬───────────────────────┘
                                      ▼
                              ┌──────────────┐
                              │   Supervisor  │
                              │  （意图路由）  │
                              └──────┬───────┘
                                     │
            ┌────────────┬──────────┼──────────┬────────────┐
            ▼            ▼          ▼          ▼            ▼
      ┌─────────┐ ┌────────┐ ┌────────┐  ┌──────────┐ ┌──────────┐
      │Academic │ │Planning│ │Emotional│  │ Unknown  │ │ ...      │
      │学科答疑  │ │学习规划 │ │情绪疏导 │  │ 未知意图  │ │          │
      └────┬────┘ └────┬───┘ └────┬───┘  └────┬─────┘ └──────────┘
           │           │          │           │
      ┌────┴────┐ ┌───┴───┐     │      ┌────┴────┐
      │Fan-out/ │ │对抗式 │     │      │直接响应 │
      │Fan-in  │ │博弈   │     │      │         │
      │RAG检索 │ │HIL   │     │      │         │
      │幻觉检测│ │      │     │      │         │
      └─────────┘ └───────┘     │      └─────────┘
                                  │
                             ┌────┴────┐
                             │  END   │
                             └─────────┘
```

### 3.2 技术架构分层

| 层级 | 技术组件 | 说明 |
|------|---------|------|
| **前端** | Next.js 16 + Tailwind CSS + React Flow | 响应式聊天UI、SSE消费、交互式DAG可视化 |
| **后端API** | FastAPI + Uvicorn | SSE流式端点、CORS、OTel自动埋点 |
| **编排引擎** | LangGraph | StateGraph + interrupt() HIL + 条件边 + Fan-out/Fan-in |
| **路由LLM** | Qwen2.5-7B（SiliconFlow） | 轻量意图分类 + 反馈路由 |
| **生成LLM** | DeepSeek-V3 | 学科解答、学习计划、情绪支持 |
| **LLM容灾** | Qwen2.5-7B（SiliconFlow） | 跨厂商故障转移 |
| **向量数据库** | ChromaDB | 本地知识库检索（L2→相关度归一化） |
| **文本嵌入** | BAAI/bge-m3（SiliconFlow） | RAG向量化 |
| **关键词检索** | rank-bm25 + jieba | 中文感知BM25检索 |
| **重排序** | BAAI/bge-reranker-v2-m3 | 合并候选集精排 |
| **网络搜索** | DuckDuckGo | 学习规划及学科问答的在线补充 |
| **状态持久化** | PostgreSQL（psycopg） | LangGraph Checkpointer + HIL中断恢复 |
| **可观测性** | OpenTelemetry + Jaeger + SQLite | 全链路分布式追踪 |
| **配置管理** | YAML + XML | 运行参数与提示词模板 |

---

## 四、LangGraph 工作流详解

### 4.1 学科辅导流程（Academic）

```
supervisor ──> academic_router ──┬──> rag_retrieve ──┐
                                 │                    ├──> generate_answer ──> evaluate_hallucination ──┬──> END
                                 └──> web_search ─────┘                              │                      │
                                                                                      └──> rewrite_query ──> (retry)
```

1. **Fan-out并行检索**：`academic_router` 同时触发 `rag_retrieve`（向量+BM25混合检索）和 `web_search`（网络搜索）
2. **Fan-in汇聚生成**：两路结果汇聚到 `generate_answer`，LLM综合生成最终回答
3. **幻觉评估闭环**：`evaluate_hallucination` 检测回答是否基于检索上下文，检测到幻觉时通过 `rewrite_query` 改写查询并重试

### 4.2 学习规划流程（Planning）

```
supervisor ──> search_policy ──> gather_intel ──> drafter
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                      reviewer_academic      reviewer_emotional       consensus_check
                              │                     │                     │
                              └─────────────────────┘                     ├──> adv_rewrite ──> (retry)
                                                                          └──> plan_output ──> HIL中断
                                                                                                  │
                                                                                    ┌─────────────┴─────────────┐
                                                                                    ▼                           ▼
                                                                               feedback_router              END
                                                                                    │
                                                                      ┌───────────┴───────────┐
                                                                      ▼                       ▼
                                                                  plan_tweak              drafter (rewrite)
                                                                      │
                                                                 plan_output
```

1. **情报收集**：并行收集情绪情报和资源情报（RAG+网络搜索）
2. **对抗式起草**：起草者生成学习计划
3. **双审查员并行评审**：学术审查员 + 情绪审查员同时评估
4. **共识检查**：全票通过输出，否则打回重写（最多3轮）
5. **HIL人工审批**：计划输出时中断，等待用户确认或反馈
6. **反馈路由**：微调快速局部修改，重写走完整对抗循环

### 4.3 状态定义（TutorState）

```python
class TutorState(TypedDict):
    messages: list                    # 对话历史
    intent: Literal["academic", "planning", "emotional", "unknown"]  # 用户意图
    subject: str                      # 当前学科（math/chinese/other）
    keypoints: list[str]             # 知识点列表
    context: list[dict]              # 检索上下文（Fan-in合并）
    search_results: list[dict]       # 搜索结果
    plan: str                        # 生成的学习计划
    retry_count: int                 # 幻觉重试计数
    hallucination_detected: bool     # 是否检测到幻觉
    # ... 对抗式规划字段
    draft: str                       # 当前计划草稿
    academic_verdict: str            # 学术审查结论
    emotional_verdict: str         # 情绪审查结论
    adv_round: int                   # 审查轮次
    consensus: bool                  # 是否全票通过
    # ... HIL反馈字段
    hil_action: str                  # "confirm" 或 "feedback"
    hil_feedback: str               # 用户反馈文本
    hil_summary: str                # 压缩摘要
    feedback_route: str             # "tweak" 或 "rewrite"
```

---

## 五、关键技术特性

### 5.1 混合RAG检索
- **向量检索**：ChromaDB + BAAI/bge-m3嵌入，支持学科和年份过滤
- **BM25关键词检索**：rank-bm25 + jieba中文分词，自动失效重建
- **BGE重排序**：BAAI/bge-reranker-v2-m3对合并候选集精排
- **智能合并**：按内容哈希去重，rerank_score与原始分数双重阈值判断

### 5.2 LLM容灾机制
```python
def async_invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    try:
        return await primary.ainvoke(messages)
    except (TimeoutError, ConnectionError, APITimeoutError, ...):
        if fallback:
            return await fallback.ainvoke(messages)
        raise
```
- 捕获超时、连接错误、502、限流等可恢复错误
- 自动切换备用模型（默认Qwen2.5-7B）
- OpenTelemetry记录容灾事件

### 5.3 全链路追踪
- `@traced_node` 装饰器自动追踪所有图节点
- `@traced_llm_call` 追踪LLM调用（模型、温度、Token用量）
- `@traced_retrieval` / `@traced_search` 追踪检索和搜索
- OpenTelemetry导出至 Jaeger UI + SQLite 双通道

### 5.4 SSE实时流
- `/stream` 端点：启动新对话流
- `/resume` 端点：恢复HIL中断的图执行
- 事件类型：thread_id、node_event、token、text、usage、interrupt、done、error

### 5.5 配置驱动
- `config/settings.yaml`：运行参数（温度、超时、重试上限）
- `config/prompts/*.xml`：XML提示词模板，支持动态加载和格式化
- 所有LLM节点行为可通过配置调整，无需修改代码

---

## 六、项目结构

```
ExamMind/
├── app.py                        # FastAPI SSE端点 + lifespan管理
├── Dockerfile                     # 多阶段构建（前端 + 后端）
├── docker-compose.yml             # 一键部署（后端 + PostgreSQL + Jaeger）
├── pyproject.toml                 # Python依赖管理
├── config/
│   ├── settings.yaml              # 运行参数配置
│   └── prompts/                   # XML提示词模板
├── src/
│   ├── graph/
│   │   ├── builder.py             # 图构建与编译（19个节点）
│   │   ├── state.py               # TutorState状态定义（26个字段）
│   │   ├── supervisor.py          # 意图路由 + 关键词提取
│   │   ├── academic.py            # 并行检索、答案生成、幻觉评估
│   │   ├── planner.py             # 政策搜索 + 情报收集
│   │   ├── plan_adversarial.py    # 对抗式起草/审查 + HIL反馈路由
│   │   ├── emotional.py           # 情绪支持
│   │   └── llm.py                 # 统一LLM工厂 + 容灾降级
│   ├── rag/                       # 混合检索：向量 + BM25 + Reranker
│   │   ├── indexer.py             # ChromaDB索引管理
│   │   ├── retriever.py           # 混合检索主逻辑
│   │   ├── reranker.py            # BGE重排序
│   │   ├── loader.py              # 文档加载
│   │   └── section_splitter.py    # 章节拆分
│   ├── config/                    # YAML配置加载 + XML提示词缓存
│   │   └── config_manager.py
│   ├── database/                  # PostgreSQL Checkpointer管理
│   │   └── checkpointer.py
│   ├── tracing/                   # OTel初始化、@traced_node、SQLite导出
│   │   ├── collector.py
│   │   ├── decorators.py
│   │   └── sqlite_exporter.py
│   ├── tools/                     # 工具集
│   │   ├── rag_tool.py
│   │   └── search_tool.py
│   └── schemas.py                 # Pydantic请求模型
├── frontend/                      # Next.js前端
│   ├── app/page.tsx               # 主页面：SSE消费、HIL反馈
│   └── components/
│       ├── chat-area.tsx          # 消息气泡 + Markdown渲染
│       ├── plan-review.tsx        # HIL计划审阅（编辑/反馈/导出）
│       ├── right-panel.tsx        # 交互式DAG + 节点轨迹 + 日志
│       └── left-sidebar.tsx       # 对话历史
├── data/                          # 试卷数据（语文、数学）
├── scripts/                       # 索引构建脚本
│   └── build_index.py
├── tests/                         # 测试套件（全部Mock）
└── docs/                          # 架构文档
    └── architecture/
```

---

## 七、部署方式

### 7.1 Docker Compose（推荐）

```bash
git clone <repo-url>
cd ExamMind

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 SILICONFLOW_API_KEY

# 启动（后端 + 前端 + PostgreSQL）
docker compose up -d

# 可选：启用Jaeger追踪
docker compose --profile observability up -d
```

### 7.2 本地开发

```bash
# 后端
conda create -n exam_mind python=3.11 -y
conda activate exam_mind
pip install -e ".[dev]"

# 构建知识库
python scripts/build_index.py

# 启动后端
uvicorn app:app --reload --port 8000

# 前端
cd frontend
npm install
npm run dev
```

---

## 八、测试

```bash
# 单元测试（无需在线API）
OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short

# 前端构建检查
cd frontend && npm run build
```

---

## 九、技术选型亮点

1. **LangGraph深度实践**：完整利用StateGraph、条件边、Fan-out/Fan-in、interrupt() HIL等高级特性
2. **多智能体协作模式**：对抗式博弈（起草者 vs 双审查员）确保输出质量
3. **生产级容错**：LLM容灾、超时降级、安全阀机制、优雅错误处理
4. **可观测性优先**：全链路OpenTelemetry追踪，Jaeger + SQLite双通道导出
5. **配置驱动设计**：YAML参数 + XML提示词，行为调整无需改代码
6. **状态持久化**：PostgreSQL Checkpointer支持多轮对话记忆和HIL中断恢复

---

## 十、开源协议

本项目基于 [MIT License](./LICENSE) 开源。
