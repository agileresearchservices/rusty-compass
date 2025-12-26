# Claude AI Assistant Guide for Rusty Compass

This guide provides comprehensive instructions for AI assistants (like Claude Code) working with the Rusty Compass codebase. It covers architecture, conventions, development workflows, and best practices.

## Project Overview

**Rusty Compass** is a production-grade LangGraph ReAct agent with:
- **Local LLM reasoning**: gpt-oss:20b via Ollama
- **Semantic search**: PostgreSQL + PGVector (768-dimensional embeddings)
- **Cross-encoder reranking**: Qwen3-Reranker-8B for relevance scoring
- **Persistent memory**: Conversation history with context compaction
- **Real-time streaming**: Character-by-character thinking and responses
- **Hybrid search**: Vector similarity + full-text search with RRF fusion
- **Query evaluation**: Dynamic search parameter adjustment

**Status**: Production Ready ✓
**Last Updated**: 2025-12-26

---

## Quick Reference

### Directory Structure
```
rusty-compass/
├── README.md                          # Project overview
├── CLAUDE.md                          # This file (AI assistant guide)
├── docker-compose.yml                 # PostgreSQL + PGVector setup
├── postgres-credentials.md            # Database credentials reference
├── .claude/
│   └── settings.local.json           # Claude Code permissions
├── sample_docs/                       # Knowledge base documents
│   ├── python_basics.txt
│   ├── machine_learning_intro.txt
│   └── web_development.txt
└── langchain_agent/                   # Main application
    ├── main.py                        # Core agent logic (1430 lines)
    ├── config.py                      # All configuration constants
    ├── setup.py                       # Unified initialization script
    ├── requirements.txt               # Production dependencies
    ├── requirements-dev.txt           # Development tools
    ├── Makefile                       # Development commands
    ├── README.md                      # User guide
    ├── SETUP.md                       # Setup instructions
    ├── DEVELOPER.md                   # Architecture reference
    ├── test_reranker.py              # Reranker test suite
    ├── test_hybrid_search.py         # Hybrid search tests
    ├── test_query_evaluator.py       # Query evaluator tests
    └── .claude/
        └── agents/
            └── code-modernization-reviewer.md  # Custom agent
```

### Key Files & Line Numbers

| File | Purpose | Key Sections |
|------|---------|--------------|
| `main.py:95-234` | Qwen3Reranker class | Cross-encoder reranking logic |
| `main.py:236-250` | CustomAgentState | LangGraph state schema |
| `main.py:252-537` | SimplePostgresVectorStore | PGVector integration |
| `main.py:539-585` | PostgresRetriever | Custom retriever with reranking |
| `main.py:587-1428` | LangChainAgent | Main agent class, tools, graph |
| `main.py:955` | @tool decorator | Knowledge base search tool |
| `config.py:1-61` | Exports & constants | All configuration options |
| `setup.py` | Setup script | 7-step initialization process |

---

## Core Architecture

### Component Flow
```
User Query
    ↓
[Query Evaluator] → Classifies query type → Adjusts lambda_mult
    ↓
[LangGraph Agent] → ReAct reasoning loop
    ↓
[Knowledge Base Tool] → Invoked by agent
    ↓
[Hybrid Search] → Vector (PGVector) + Full-Text (PostgreSQL)
    ↓
[RRF Fusion] → Combines results (k=60)
    ↓
[Qwen3-Reranker-8B] → Cross-encoder scoring (top 4 from 15)
    ↓
[LLM (gpt-oss:20b)] → Generates response
    ↓
[PostgreSQL Memory] → Saves conversation
    ↓
Streamed Output → User
```

### Key Classes

**Qwen3Reranker** (`main.py:95-234`)
- Uses Ollama's `generate` API with Qwen3-Reranker-8B
- Input format: `query: {query}\ndoc: {document}`
- Extracts numerical scores from output
- Handles batch reranking with error fallback

**SimplePostgresVectorStore** (`main.py:252-537`)
- Direct PGVector integration (no LangChain abstraction)
- Vector similarity: `<=> ` operator (cosine distance)
- Full-text search: `to_tsquery`, `ts_rank_cd`
- RRF fusion: `1 / (k + rank)` with k=60
- Connection pooling for performance

**PostgresRetriever** (`main.py:539-585`)
- Wraps SimplePostgresVectorStore
- Orchestrates hybrid search + reranking
- Configurable via `RETRIEVER_*` and `RERANKER_*` config

**LangChainAgent** (`main.py:587-1428`)
- LangGraph ReAct agent
- Tools: knowledge base search, conversation management
- State: messages, context, reasoning output
- Checkpointing: PostgresSaver for persistence

---

## Development Conventions

### Code Style
- **Formatting**: Black (line length 88)
- **Linting**: Pylint (disable missing-docstring)
- **Type hints**: Use for all public methods (mypy compatible)
- **Comments**: Explain "why", not "what"
- **Imports**: Standard lib → Third party → Local

### Naming Conventions
- **Constants**: UPPER_SNAKE_CASE (in `config.py`)
- **Classes**: PascalCase
- **Functions/methods**: snake_case
- **Private methods**: `_leading_underscore`
- **Type hints**: Use `from typing import` imports

### File Modification Rules

**DO:**
- ✅ Use the Read tool before editing any file
- ✅ Prefer editing existing files over creating new ones
- ✅ Run tests after changes: `make test`
- ✅ Update relevant documentation (README, SETUP, DEVELOPER)
- ✅ Add type hints to new functions
- ✅ Use connection pooling (never create raw connections)

**DON'T:**
- ❌ Create new files unless absolutely necessary
- ❌ Add emojis to code/docs unless user requests
- ❌ Create markdown documentation proactively
- ❌ Over-engineer solutions (KISS principle)
- ❌ Add error handling for impossible scenarios
- ❌ Create abstractions for single-use code

---

## Common Development Tasks

### 1. Adding a New Tool to the Agent

**Location**: `main.py:955` (after existing tool)

**Steps**:
1. Define tool function with `@tool` decorator
2. Add to `tools` list in `create_agent_graph()`
3. Test with queries
4. Document in `DEVELOPER.md`

**Example**:
```python
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Returns current time in specified timezone."""
    # Implementation here
    pass

# In create_agent_graph():
tools = [kb_search_tool, get_current_time]
```

**Testing**: Run agent, ask "What time is it?" and verify tool invocation.

### 2. Modifying Configuration

**Location**: `config.py`

**Steps**:
1. Update constant value
2. Restart agent to apply
3. Test affected functionality
4. Document in `SETUP.md` if user-facing

**Common configs**:
- `LLM_MODEL`: Change reasoning model
- `RETRIEVER_K`: Number of final results (default: 4)
- `RERANKER_TOP_K`: Reranker output count (default: 4)
- `RETRIEVER_LAMBDA_MULT`: Vector/text balance (0.0-1.0)
- `ENABLE_RERANKING`: Toggle reranking (default: True)

### 3. Adding Sample Documents

**Location**: `sample_docs/`

**Steps**:
1. Create `.txt` file in `sample_docs/`
2. Run `cd langchain_agent && python setup.py`
3. Verify: query should return new content

**Document format**: Plain text, UTF-8 encoded.

### 4. Debugging Search Results

**Locations**:
- Hybrid search: `main.py:346-430` (SimplePostgresVectorStore.similarity_search)
- Reranking: `main.py:95-234` (Qwen3Reranker)
- Query evaluation: `main.py:587` (LangChainAgent._evaluate_query)

**Debug outputs** (printed during execution):
```
[Query Evaluator] Query type: fact_lookup, lambda_mult: 0.9
[Hybrid Search] Retrieved 15 chunks
[Reranker] Reranking 15 candidates → top 4 selected
[Reranker] Score: 0.85 - "Python is a high-level..."
```

**Common issues**:
- Low reranker scores (<0.7): Add more relevant documents
- No results: Check database with `psql -U postgres -d langchain_agent -c "SELECT COUNT(*) FROM document_chunks;"`
- Slow queries: Check PGVector index: `\d document_chunks` in psql

### 5. Performance Optimization

**Current benchmarks**:
- First query: 15-30s (model loading)
- Subsequent queries: 6-32s
- Vector search: ~600ms
- Reranking (15 docs): ~1-2s
- LLM response: 5-30s

**Optimization targets**:
1. **Hybrid search**: Tune `RETRIEVER_FETCH_K` (default: 20)
2. **Reranking**: Adjust `RERANKER_FETCH_K` (default: 15)
3. **Database**: Check connection pool size (`DB_POOL_MAX_SIZE`: 20)
4. **Embeddings**: Batch operations (already implemented in setup.py)

---

## Testing

### Running Tests

**All tests**:
```bash
cd langchain_agent
make test
```

**Individual test suites**:
```bash
make test-reranker      # Qwen3-Reranker-8B tests
make test-hybrid        # Hybrid search tests
make test-query         # Query evaluator tests
```

**Expected output**:
```
[1/3] Running reranker tests...
test_reranker.py: 6/6 passing ✓

[2/3] Running hybrid search tests...
test_hybrid_search.py: tests passing ✓

[3/3] Running query evaluator tests...
test_query_evaluator.py: tests passing ✓

✓ All tests passed!
```

### Test Coverage

**Reranker tests** (`test_reranker.py`):
- Model loading and initialization
- Single document scoring
- Batch reranking
- Edge cases (empty queries, malformed output)

**Hybrid search tests** (`test_hybrid_search.py`):
- Vector similarity search
- Full-text search
- RRF fusion
- Lambda multiplier adjustment

**Query evaluator tests** (`test_query_evaluator.py`):
- Query type classification
- Lambda multiplier calculation
- Timeout handling

### Writing New Tests

**Pattern**:
```python
def test_feature():
    """Test description."""
    # Setup
    component = initialize_component()

    # Execute
    result = component.method(test_input)

    # Assert
    assert result == expected_output
    print("✓ test_feature passed")

if __name__ == "__main__":
    test_feature()
```

---

## Git Workflow

### Branch Naming

**Current branch**: `claude/add-claude-documentation-vXp5b`

**Convention**: `claude/<description>-<session-id>`
- `claude/` prefix is REQUIRED for push permissions
- Session ID suffix is REQUIRED (matches Claude Code session)
- Use kebab-case for description

### Commit Messages

**Format**:
```
Brief summary (50 chars or less)

## Detailed description

### Key Changes
- Bullet point 1
- Bullet point 2

### Impact
- Performance improvement
- Bug fix
- etc.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <model> <noreply@anthropic.com>
```

**Use heredoc for multi-line commits**:
```bash
git commit -m "$(cat <<'EOF'
Add feature X

## Description
Details here...
EOF
)"
```

### Pushing Changes

**CRITICAL**: Always use `-u origin <branch-name>`:
```bash
git push -u origin claude/add-claude-documentation-vXp5b
```

**Retry logic**: If push fails with network error, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s).

### Pre-commit Checklist

1. ✅ Run `make test` - all tests passing
2. ✅ Run `make lint` - no critical issues
3. ✅ Run `make format` - code formatted
4. ✅ Update relevant docs (README, SETUP, DEVELOPER)
5. ✅ Verify changes with manual testing
6. ✅ Review git diff before committing

---

## Claude Code Integration

### Custom Agent

**Code Modernization Reviewer** (`.claude/agents/code-modernization-reviewer.md`)
- **Purpose**: Review code for deprecations, best practices, modernization
- **Model**: Opus (high-quality analysis)
- **Use when**: After writing/modifying significant code
- **Capabilities**:
  - Deprecation detection
  - Best practices evaluation
  - Documentation quality assessment
  - Web research for current standards

**Invocation**:
```
Ask Claude: "Use the code-modernization-reviewer agent to review [file/component]"
```

### Makefile Commands

```bash
make help           # Show all commands
make setup          # Run setup.py
make test           # Run all tests
make test-reranker  # Test reranker only
make test-hybrid    # Test hybrid search only
make test-query     # Test query evaluator only
make lint           # Code quality checks (pylint)
make format         # Format code (black)
make type-check     # Type checking (mypy)
make run            # Start agent
make clean          # Remove caches
make clean-db       # Reset database (⚠️ destructive)
```

### Permissions

**Allowed commands** (`.claude/settings.local.json`):
- Docker: `docker compose up`, `docker-compose up`
- Python: `python`, `python3`, `pip install`, `.venv/bin/python`
- Git: `git add`, `git commit`, `git push`, `git restore`, `git rm`
- Database: `psql`
- Utilities: `ls`, `cat`, `find`, `tree`, `wc`
- Ollama: `ollama pull`
- WebFetch: LangChain docs, PyPI, Anthropic docs, dev blogs
- WebSearch: General web search

**Timeout overrides**:
- `timeout 180 python` - 3 minutes (for tests)
- `timeout 600 python` - 10 minutes (for setup)

### Useful Patterns for Claude Code

**Understanding the codebase**:
```
"Explain the data flow from user query to final response"
"Where is the reranking logic implemented?"
"Show me how hybrid search works"
```

**Debugging**:
```
"I'm seeing [error]. Where in [component] would this happen?"
"Why are reranker scores low?"
"The agent isn't invoking the knowledge base tool. Help diagnose."
```

**Development**:
```
"Help me add a tool that [does X]"
"Review my changes to [file] for best practices"
"Write a test for [feature] that covers edge cases"
```

**Optimization**:
```
"What's the performance bottleneck in [function]?"
"How can I speed up hybrid search?"
"Profile the query execution and suggest improvements"
```

---

## Configuration Reference

### Key Configuration Constants

**Ollama** (`config.py:64-75`):
```python
LLM_MODEL = "gpt-oss:20b"              # Reasoning model
EMBEDDINGS_MODEL = "nomic-embed-text:latest"  # 768-dim embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
```

**PostgreSQL** (`config.py:78-97`):
```python
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "langchain_agent"
DB_POOL_MAX_SIZE = 20
```

**PGVector** (`config.py:100+`):
```python
VECTOR_DIMENSION = 768
VECTOR_INDEX_TYPE = "ivfflat"
VECTOR_SIMILARITY_METRIC = "cosine"
```

**Retriever** (`config.py:28-32`):
```python
RETRIEVER_K = 4                        # Final results
RETRIEVER_FETCH_K = 20                 # Initial retrieval
RETRIEVER_LAMBDA_MULT = 0.5            # Vector/text balance (0.0-1.0)
```

**Reranker** (`config.py:33-38`):
```python
ENABLE_RERANKING = True
RERANKER_MODEL = "qwen3-reranker:8b"
RERANKER_FETCH_K = 15                  # Documents to rerank
RERANKER_TOP_K = 4                     # Top results after reranking
```

**Query Evaluation** (`config.py:45-47`):
```python
ENABLE_QUERY_EVALUATION = True
DEFAULT_LAMBDA_MULT = 0.5
QUERY_EVAL_TIMEOUT_MS = 5000
```

**Conversation Compaction** (`config.py:54-60`):
```python
ENABLE_COMPACTION = True
MAX_CONTEXT_TOKENS = 8000
COMPACTION_THRESHOLD_PCT = 0.8
MESSAGES_TO_KEEP_FULL = 2
MIN_MESSAGES_FOR_COMPACTION = 5
```

---

## Troubleshooting

### Common Issues

**1. Database connection errors**
```
Error: could not connect to server
```
**Fix**: Start PostgreSQL: `docker compose up -d`

**2. Ollama model not found**
```
Error: model 'gpt-oss:20b' not found
```
**Fix**: Run setup: `cd langchain_agent && python setup.py`

**3. No search results**
```
[Hybrid Search] Retrieved 0 chunks
```
**Fix**: Load data: `python setup.py` (step 6)

**4. Low reranker scores**
```
[Reranker] Score: 0.12 - "..."
```
**Fix**: Add more relevant documents to `sample_docs/`

**5. Slow queries**
```
Vector search: 5000ms (expected ~600ms)
```
**Fix**: Check PGVector index exists: `psql -U postgres -d langchain_agent -c "\d document_chunks"`

**6. Agent not using knowledge base**
```
Agent responds without invoking tool
```
**Fix**: Rephrase query to clearly require factual information

### Debugging Commands

**Check database**:
```bash
psql -U postgres -d langchain_agent -c "SELECT COUNT(*) FROM document_chunks;"
psql -U postgres -d langchain_agent -c "SELECT COUNT(*) FROM documents;"
```

**Check Ollama models**:
```bash
ollama list
```

**Check vector index**:
```bash
psql -U postgres -d langchain_agent -c "\d document_chunks"
# Should show: idx_document_chunks_embedding (ivfflat)
```

**Test embeddings**:
```python
from langchain_ollama import OllamaEmbeddings
embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
result = embedder.embed_query("test")
print(f"Dimension: {len(result)}")  # Should be 768
```

**Test reranker**:
```bash
cd langchain_agent
python test_reranker.py
```

---

## Advanced Topics

### Adding New Query Types

**Location**: `main.py:587` (`LangChainAgent._evaluate_query`)

**Current types**:
- `fact_lookup`: High vector weight (0.9)
- `how_to`: Balanced (0.7)
- `conceptual`: Slightly text-weighted (0.4)
- `conversational`: Text-heavy (0.2)

**To add new type**:
1. Modify evaluation prompt with new category
2. Add lambda_mult mapping in `_evaluate_query`
3. Test with representative queries
4. Document in `DEVELOPER.md`

### Custom Reranker Models

**Location**: `config.py:36`

**Options**:
- `qwen3-reranker:8b` (default, production)
- `qwen3-reranker:4b` (faster, less accurate)
- Any Ollama-compatible reranker

**Testing**:
```bash
# Pull model
ollama pull qwen3-reranker:4b

# Update config
# config.py: RERANKER_MODEL = "qwen3-reranker:4b"

# Test
cd langchain_agent
python test_reranker.py
```

### Connection Pool Tuning

**Location**: `config.py:92-97`

**Current settings**:
```python
DB_POOL_MAX_SIZE = 20
DB_CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}
```

**Tuning**:
- Increase `DB_POOL_MAX_SIZE` for high concurrency
- Monitor pool exhaustion in logs
- Use `psql` to check active connections: `SELECT count(*) FROM pg_stat_activity;`

### Conversation Compaction

**Purpose**: Prevent context overflow by summarizing old messages

**Trigger**: When messages exceed 80% of `MAX_CONTEXT_TOKENS` (8000)

**Behavior**:
- Keeps last 2 messages in full (`MESSAGES_TO_KEEP_FULL`)
- Summarizes older messages
- Requires min 5 messages (`MIN_MESSAGES_FOR_COMPACTION`)

**Disable**: Set `ENABLE_COMPACTION = False` in `config.py`

---

## Dependencies

### Production (`requirements.txt`)

```
langchain>=1.2.0,<2.0.0
langchain-core>=1.2.5,<2.0.0
langchain-ollama>=1.0.1,<2.0.0
langchain-chroma>=1.1.0,<2.0.0
langchain-postgres>=0.0.16
langgraph>=1.0.5,<2.0.0
langgraph-checkpoint-postgres>=3.0.2,<4.0.0
psycopg[binary]>=3.3.0
psycopg-pool>=3.3.0
pgvector>=0.3.0
sqlalchemy>=2.0.0
chromadb>=1.4.0
```

### Development (`requirements-dev.txt`)

```
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-timeout>=2.1.0

# Code Quality
black>=23.0.0
pylint>=2.17.0
flake8>=6.0.0
mypy>=1.0.0
isort>=5.12.0

# Development Tools
ipython>=8.0.0
ipdb>=0.13.0
pre-commit>=3.0.0

# Performance
line-profiler>=4.0.0
memory-profiler>=0.60.0
```

---

## Documentation Map

### For Users
- **[README.md](README.md)**: Project overview, quick start, features
- **[langchain_agent/README.md](langchain_agent/README.md)**: User guide, commands, usage
- **[langchain_agent/SETUP.md](langchain_agent/SETUP.md)**: Installation, configuration, troubleshooting

### For Developers
- **[langchain_agent/DEVELOPER.md](langchain_agent/DEVELOPER.md)**: Architecture, components, extending
- **[CLAUDE.md](CLAUDE.md)**: This file (AI assistant guide)

### Quick Decision Tree

**User asks how to use the agent?** → `langchain_agent/README.md`
**User asks how to install/configure?** → `langchain_agent/SETUP.md`
**Developer asks about architecture?** → `langchain_agent/DEVELOPER.md`
**AI assistant needs context?** → `CLAUDE.md` (this file)

---

## Support & Resources

### For Claude Code Help
- **Built-in help**: `/help`
- **Ask questions**: "How do I..." or "Can Claude Code..."
- **Report issues**: https://github.com/anthropics/claude-code/issues

### For Project Help
- **Architecture reference**: `langchain_agent/DEVELOPER.md`
- **Makefile tasks**: `make help`
- **Test patterns**: Review `test_*.py` files
- **Recent changes**: `git log --oneline -10`

### External Resources
- **LangGraph docs**: https://langchain-ai.github.io/langgraph/
- **LangChain docs**: https://python.langchain.com/
- **PGVector docs**: https://github.com/pgvector/pgvector
- **Ollama docs**: https://ollama.ai/

---

## Quick Start for AI Assistants

When you first interact with this codebase:

1. **Read this file** (CLAUDE.md) - you're doing it! ✓
2. **Check git status**: `git status` - see current branch and changes
3. **Review recent commits**: `git log --oneline -5` - understand recent work
4. **Read the docs**: Start with `README.md`, then `DEVELOPER.md`
5. **Run tests**: `cd langchain_agent && make test` - verify setup
6. **Explore the code**: Focus on `main.py` (core logic) and `config.py` (settings)
7. **Ask clarifying questions**: Use the patterns in "Claude Code Integration" section

**Remember**:
- Always read files before editing
- Run tests after changes
- Follow git conventions for branches/commits
- Document significant changes
- Keep solutions simple (avoid over-engineering)

Happy coding! 🚀
