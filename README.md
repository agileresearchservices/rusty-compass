# Rusty Compass

A production-grade, fully-local **LangGraph ReAct agent** with real-time streaming, hybrid search, and self-improving retrieval.

## Quick Links

- **[Setup Guide](langchain_agent/SETUP.md)** - Complete setup from scratch (single `python setup.py` command)
- **[User Guide](langchain_agent/README.md)** - How to use the agent and customize behavior
- **[Developer Guide](langchain_agent/DEVELOPER.md)** - Architecture, components, and extending the system

## What is It?

A fully local RAG (Retrieval-Augmented Generation) agent that combines:

- **LangGraph ReAct Agent** - Graph-based state machine orchestration with reasoning
- **Hybrid Search** - Vector + full-text search with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** - BGE reranker for improved relevance scoring
- **Self-Improving Reflection Loop** - Document grading, query transformation, response grading
- **Persistent Memory** - PostgreSQL-backed conversation history with context compaction
- **Real-Time Streaming** - Character-by-character output via WebSocket/CLI
- **100% Local** - No external API calls, complete privacy

## Architecture

### System Overview

```mermaid
flowchart TB
    subgraph UI["User Interfaces"]
        CLI["CLI<br/>(main.py)"]
        WEB["Web UI<br/>(React + TypeScript)"]
    end

    subgraph Agent["LangGraph ReAct Agent"]
        QE["Query Evaluator<br/>Classify → Adjust λ"]
        REACT["ReAct Loop<br/>Reason → Act → Observe"]
        TOOLS["Knowledge Base Tool"]
    end

    subgraph Reflection["Self-Improving Reflection Loop"]
        DG["Document Grader<br/>Score relevance"]
        QT["Query Transformer<br/>Rewrite on failure"]
        RG["Response Grader<br/>Evaluate quality"]
        RI["Response Improver<br/>Retry with feedback"]
    end

    subgraph Search["Hybrid Search Pipeline"]
        VS["Vector Search<br/>(768-dim embeddings)"]
        TS["Full-Text Search<br/>(tsvector/tsquery)"]
        RRF["RRF Fusion<br/>(k=60)"]
        RERANK["BGE Reranker<br/>Cross-encoder scoring"]
    end

    subgraph Storage["PostgreSQL + PGVector"]
        DOCS["Documents & Chunks"]
        IDX["IVFFlat + GIN Indexes"]
        MEM["Conversation Memory<br/>(Checkpoints)"]
    end

    subgraph Ollama["Ollama (Local LLM)"]
        LLM["gpt-oss:20b"]
        EMB["nomic-embed-text"]
    end

    UI --> QE
    QE --> REACT
    REACT --> TOOLS
    TOOLS --> Search
    VS --> RRF
    TS --> RRF
    RRF --> RERANK
    RERANK --> DG
    DG -->|pass| REACT
    DG -->|fail| QT
    QT -->|retry| QE
    REACT --> LLM
    LLM --> RG
    RG -->|pass| UI
    RG -->|fail| RI
    RI --> REACT
    Search --> Storage
    REACT --> MEM
    VS --> EMB
```

### Hybrid Search & Reranking Pipeline

```mermaid
flowchart LR
    subgraph Input
        Q["User Query"]
    end

    subgraph Embedding
        E["nomic-embed-text<br/>768 dimensions"]
    end

    subgraph Search["Parallel Search"]
        VS["Vector Search<br/>(Cosine Distance)"]
        TS["Full-Text Search<br/>(ts_rank_cd)"]
    end

    subgraph Fusion
        RRF["Reciprocal Rank Fusion<br/>score = Σ 1/(rank + 60)"]
    end

    subgraph Rerank
        BGE["BGE-Reranker-v2-m3<br/>Cross-encoder scoring<br/>15 → 4 documents"]
    end

    subgraph Output
        DOCS["Top 4 Documents<br/>Relevance scored"]
    end

    Q --> E
    E --> VS
    Q --> TS
    VS --> RRF
    TS --> RRF
    RRF --> BGE
    BGE --> DOCS
```

### Reflection Loop (Self-Improvement)

```mermaid
flowchart TD
    START["Query"] --> QE["Query Evaluator<br/>Classify type → Set λ"]
    QE --> AGENT["Agent<br/>ReAct reasoning"]
    AGENT --> TOOLS["Retrieve Documents<br/>Hybrid + Rerank"]
    TOOLS --> DG["Document Grader<br/>LLM scores each doc"]

    DG -->|"≥1 relevant doc<br/>score ≥ 0.3"| GEN["Generate Response"]
    DG -->|"0 relevant docs<br/>iteration < max"| QT["Query Transformer<br/>Rewrite query"]
    QT --> QE

    GEN --> RG["Response Grader<br/>Check quality"]
    RG -->|"PASS<br/>confidence > 0.85"| END["Stream to User"]
    RG -->|"FAIL<br/>confidence < 0.5"| RI["Response Improver<br/>Add feedback"]
    RI --> AGENT

    style DG fill:#f9f,stroke:#333
    style RG fill:#f9f,stroke:#333
    style QT fill:#bbf,stroke:#333
    style RI fill:#bbf,stroke:#333
```

## Tech Stack

| Category             | Technology                        | Purpose                                 |
| -------------------- | --------------------------------- | --------------------------------------- |
| **LLM**              | Ollama + gpt-oss:20b              | Local reasoning engine (20B parameters) |
| **Embeddings**       | nomic-embed-text                  | 768-dimensional semantic vectors        |
| **Reranker**         | BAAI/bge-reranker-v2-m3           | Cross-encoder relevance scoring         |
| **Agent Framework**  | LangGraph + LangChain             | Graph-based state machine orchestration |
| **Vector Database**  | PostgreSQL + PGVector             | Semantic search with IVFFlat indexing   |
| **Full-Text Search** | PostgreSQL (tsvector)             | Keyword search with GIN indexing        |
| **Memory**           | PostgreSQL + langgraph-checkpoint | Persistent conversation state           |
| **Backend API**      | FastAPI + WebSocket               | REST API with real-time streaming       |
| **Frontend**         | React 18 + TypeScript + Tailwind  | Modern web UI with Zustand state        |
| **Containerization** | Docker Compose                    | PostgreSQL orchestration                |

## Key Techniques

| Technique | Description |
| --- | --- |
| **Reciprocal Rank Fusion (RRF)** | Combines vector and full-text search rankings: `score = Σ 1/(rank + k)` where k=60 |
| **Cross-Encoder Reranking** | BGE model directly scores query-document relevance (0.0-1.0) |
| **Adaptive Lambda** | Dynamically adjusts vector vs. lexical weight based on query type |
| **Document Grading** | LLM evaluates each retrieved document's relevance |
| **Query Transformation** | Rewrites failed queries for better retrieval |
| **Response Grading** | Evaluates response quality (relevance, completeness, clarity) |
| **Context Compaction** | Summarizes older messages when conversation exceeds token limits |
| **Confidence-Based Early Stopping** | Skips retries when >85% confident, forces retry when <50% |

## Setup (3 Steps)

```bash
# 1. Start PostgreSQL
docker compose up -d

# 2. Run unified setup (choose your knowledge base)
cd langchain_agent

# Option A: Use LangChain/LangGraph/LangSmith documentation (recommended)
python setup.py --docs-source langchain

# Option B: Use sample documents (for quick testing)
python setup.py

# 3. Run the agent
python main.py
```

That's it! The `setup.py` script handles:

- Database initialization
- PGVector extension setup
- Vector indexes & full-text search
- Ollama model pulling (LLM, embeddings, reranker)
- Document loading with embeddings

### LangChain Documentation Knowledge Base

The agent can be configured with official LangChain documentation as its knowledge base:

```bash
# Full setup with LangChain docs
python setup.py --docs-source langchain

# Or run ingestion separately
python ingest_langchain_docs.py

# Update to latest docs
python ingest_langchain_docs.py --update

# Check current stats
python ingest_langchain_docs.py --stats
```

This ingests **~2,000 documents** from:

- **LangChain** - Core framework documentation
- **LangGraph** - Graph-based agent orchestration
- **LangSmith** - Observability and tracing platform

## Features

- **Automated Setup** - Single command initialization (`python setup.py`)
- **Real-Time Streaming** - Token-by-token output via WebSocket/CLI
- **Hybrid Search** - Vector + full-text with RRF fusion
- **Cross-Encoder Reranking** - BGE reranker scores document relevance
- **Reflection Loop** - Document grading, query transformation, response grading
- **Query Evaluation** - Dynamic lambda adjustment based on query type
- **Persistent Memory** - Multi-turn conversations with context compaction
- **Conversation Management** - Create, list, load, clear conversations
- **Web UI** - React frontend with real-time graph visualization
- **CLI** - Interactive command-line interface
- **Local Only** - All data stays on your machine
- **Fully Documented** - Setup, User, and Developer guides included  

## Example Queries

With LangChain documentation knowledge base:

```text
You: What is LangGraph?
[Reranker] Reranking 15 candidates → top 4 selected
Agent (response): LangGraph is a library for building stateful, multi-actor applications...

You: How do I create a ReAct agent in LangChain?
Agent (response): To create a ReAct agent, you can use create_react_agent()...
```

With sample documents:

```text
You: What is Python programming?
Agent (response): Python is a high-level programming language...
```

## Documentation

### For Users

- **[README](langchain_agent/README.md)** - Features, usage, commands, troubleshooting
- **[SETUP](langchain_agent/SETUP.md)** - Installation and configuration

### For Developers

- **[DEVELOPER](langchain_agent/DEVELOPER.md)** - Architecture, components, extending

## Directory Structure

```text
rusty-compass/
├── README.md                     # This file (project overview)
├── docker-compose.yml            # PostgreSQL + PGVector setup
├── sample_docs/                  # Sample knowledge base documents
│   ├── python_basics.txt
│   ├── machine_learning_intro.txt
│   └── web_development.txt
├── langchain_agent/              # Backend application
│   ├── setup.py                  # Unified setup (ONE COMMAND)
│   ├── main.py                   # Agent entry point (CLI)
│   ├── config.py                 # Configuration constants
│   ├── ingest_langchain_docs.py  # LangChain docs ingestion
│   ├── requirements.txt          # Python dependencies
│   ├── README.md                 # User guide
│   ├── SETUP.md                  # Setup guide
│   ├── DEVELOPER.md              # Developer guide
│   └── test_*.py                 # Test suites
└── web/                          # Frontend application
    ├── src/
    │   ├── components/           # React components
    │   ├── stores/               # Zustand state management
    │   └── App.tsx               # Main application
    ├── package.json              # Node dependencies
    └── vite.config.ts            # Vite build configuration
```

## Performance

| Operation | Time |
| --------- | ---- |
| First query | 15-30s (model loading) |
| Subsequent queries | 6-32s (search + reasoning) |
| Vector search | ~600ms |
| Reranking (15 docs) | ~1-2s |
| LLM response | 5-30s |

## Getting Started

1. **Read**: [SETUP.md](langchain_agent/SETUP.md) - 10 minutes
2. **Run**: `python setup.py` - 10-40 minutes (first run)
3. **Use**: `python main.py` - Start chatting!

For more details, see [README.md](langchain_agent/README.md).

---

**Status**: Production Ready
**Last Updated**: 2025-12-29
