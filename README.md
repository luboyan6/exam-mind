# ExamMind — 智能备考辅导系统

<p align="center">
  <a href="README_en.md">English</a> ·
  <a href="docs/architecture/v0.3.0/diagram_design.md">架构图</a> ·
  <a href="CHANGELOG.md">更新日志</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v0.4.0-orange?style=flat-square" alt="version" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="python" />
  <img src="https://img.shields.io/badge/langgraph-v1.1.1-7C3AED?style=flat-square" alt="langgraph" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license" />
</p>

## 简介

ExamMind 是一套面向考试备考场景的**生产级多智能体 AI 辅导系统**。

基于 **LangGraph + FastAPI + Next.js**，通过 deepseek-v4flash 路由 Agent 将用户问题分发给学科答疑、学习规划、情绪疏导三大专项 Agent，并具备完整的链路追踪与故障恢复能力。

---

## 核心能力

- **学科答疑**：混合 RAG（向量 + BM25 + BGE 重排），并行召回，幻觉评估 + 自动重试
- **学习规划**：对抗式多智能体博弈，起草 → 双审查 → 全票通过，支持人工反馈
- **情绪疏导**：资深班主任人设，提供温暖且实用的建议
- **意图路由**：deepseek-v4flash 低延迟精准分类
- **LLM 容灾**：主 API 异常时自动切换备用模型
- **全链路追踪**：OpenTelemetry + Jaeger + SQLite
- **状态持久化**：PostgreSQL Checkpointer，无库时自动降级

---

## 技术栈

| 层级 | 技术 |
| ---- | ---- |
| 前端 | Next.js 16 + Tailwind CSS 4 + React Flow |
| 后端 API | FastAPI + Uvicorn |
| 编排 | LangGraph |
| 路由 LLM | deepseek-v4flash（SiliconFlow） |
| 生成 LLM | DeepSeek-V4-pro |
| 向量检索 | ChromaDB + BAAI/bge-m3 |
| 关键词检索 | rank-bm25 + jieba |
| 状态持久化 | PostgreSQL |
| 可观测性 | OpenTelemetry + Jaeger + SQLite |

---

## 快速上手

### Docker Compose（推荐）

```bash
git clone https://github.com/luboyan6/exam-mind.git
cd exam-mind

cp .env.example .env
# 填入 DEEPSEEK_API_KEY 和 SILICONFLOW_API_KEY

docker compose up -d
```

- 前端：`http://localhost:3000`
- 后端：`http://localhost:8000`
- Jaeger：`http://localhost:16686`

### 本地开发

```bash
conda create -n exam_mind python=3.11 -y
conda activate exam_mind
pip install -e ".[dev]"

cp .env.example .env
python scripts/build_index.py

# 终端 1
uvicorn app:app --reload --port 8000

# 终端 2
cd frontend && npm install && npm run dev
```

---

## 项目结构

```
exam-mind/
├── app.py                  # FastAPI 入口
├── src/graph/              # LangGraph 工作流
├── src/rag/                # 混合检索
├── src/tracing/            # 链路追踪
├── frontend/               # Next.js 前端
├── config/prompts/         # XML 提示词模板
├── data/                   # 试卷数据
└── tests/                  # 测试套件
```

---

## 测试

```bash
# 单元测试（无需在线 API）
OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short

# 前端构建
cd frontend && npm run build
```

---

## Contributors

- [@luboyan6](https://github.com/luboyan6)

---

## 开源协议

[MIT License](./LICENSE)