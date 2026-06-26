# ExamMind

<p align="center">
  <a href="README.md">中文 README</a> ·
  <a href="docs/architecture/v0.3.0/diagram_design.md">Architecture Diagrams</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v0.4.0-orange?style=flat-square" alt="version" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="python" />
  <img src="https://img.shields.io/badge/langgraph-v1.1.1-7C3AED?style=flat-square" alt="langgraph" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license" />
</p>

## Overview

ExamMind is a **production-grade multi-agent AI tutoring platform** purpose-built for exam preparation. It leverages **LangGraph** for stateful workflow orchestration, **FastAPI** for SSE-based real-time streaming, and **Next.js** for a responsive frontend experience. A lightweight Qwen2.5-7B supervisor agent intelligently routes user inquiries across three specialized assistants — subject Q&A, study planning, and emotional support — each backed by comprehensive observability and fault-tolerance mechanisms.

---

## Key Capabilities

- **Subject Q&A** — Hybrid RAG pipeline combining vector search, BM25 keyword retrieval, and BGE reranking, with parallel fan-out/fan-in recall, hallucination detection, and automatic retry loops
- **Study Planning** — Adversarial multi-agent architecture: a drafter generates plans, two reviewers evaluate in parallel, and consensus must be unanimous before release
- **Emotional Support** — Guidance delivered through a seasoned homeroom teacher persona, balancing warmth with practical advice
- **Intent Routing** — Qwen2.5-7B supervisor for low-latency, high-accuracy query classification
- **LLM Failover** — Automatic cross-provider fallback when the primary API times out or returns errors
- **Distributed Tracing** — OpenTelemetry instrumentation with Jaeger (OTLP) and SQLite dual-channel export
- **State Persistence** — PostgreSQL-backed LangGraph Checkpointer for multi-turn conversation memory; gracefully degrades to stateless mode when no database is available
- **Configuration-Driven** — YAML-based runtime parameters and XML prompt registry; behavior changes require no code modifications
- **Real-Time Observability** — SSE-driven node status updates, reasoning path traces, token usage, and error stream delivery
- **Markdown Rendering** — Full GFM support including tables, code blocks, LaTeX formulas, and lists

---

## Getting Started

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/luboyan6/exam-mind.git
cd exam-mind

cp .env.example .env
# Fill in DEEPSEEK_API_KEY and SILICONFLOW_API_KEY

# Launch (backend + frontend + PostgreSQL)
docker compose up -d

# Optional: enable Jaeger tracing
docker compose --profile observability up -d
```

Frontend: `http://localhost:3000` · Backend API: `http://localhost:8000` · Jaeger: `http://localhost:16686`

### Option 2: Local Development

#### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- PostgreSQL (optional; auto-degrades to stateless mode if not configured)

#### Backend Setup

```bash
conda create -n exam_mind python=3.11 -y
conda activate exam_mind

pip install -e ".[dev]"

cp .env.example .env
# Fill in API keys
```

#### Build Knowledge Base

Place exam `.txt` / `.pdf` files in `data/chinese/` or `data/math/`, then run:

```bash
python scripts/build_index.py
```

#### Frontend Setup

```bash
cd frontend
npm install
```

#### Run Both

```bash
# Terminal 1 — Backend
uvicorn app:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

---

## Testing

```bash
# Unit tests (no online API required)
OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short

# Frontend build check
cd frontend && npm run build
```

---

## License

[MIT](./LICENSE)