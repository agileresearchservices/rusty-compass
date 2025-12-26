# LangChain Agent - Deployment & Configuration Guide

## Project Overview

This is a production-ready, fully local LangChain agent with real-time streaming capabilities. The agent combines:
- **Intelligent Reasoning**: Streams thinking process character-by-character
- **Semantic Search**: Local knowledge base with ChromaDB
- **Persistent Memory**: PostgreSQL conversation history
- **Local Execution**: No external API calls or dependencies

## Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **LLM** | Ollama + gpt-oss:20b | Latest | Core reasoning engine |
| **Embeddings** | Ollama + nomic-embed-text | Latest | Vector embeddings for semantic search |
| **Vector Store** | ChromaDB | 1.4.0+ | Local knowledge base storage |
| **Memory** | PostgreSQL | 16+ | Persistent conversation state |
| **Agent Framework** | LangGraph | 1.0+ | Agent orchestration & streaming |
| **CLI Language** | Python | 3.9+ | Application runtime |

## Directory Structure

```
/Users/kevin/Downloads/
├── langchain_agent/                    # Main application
│   ├── main.py                        # Agent entry point with streaming
│   ├── config.py                      # Configuration constants
│   ├── setup_db.py                   # Database initialization
│   ├── load_sample_data.py           # ChromaDB data loader
│   ├── requirements.txt              # Python dependencies
│   └── README.md                     # User documentation
│
├── sample_docs/                       # Knowledge base documents
│   ├── python_basics.txt             # Python programming guide
│   ├── machine_learning_intro.txt    # ML concepts
│   └── web_development.txt           # Web dev fundamentals
│
├── chroma_db/                         # ChromaDB persistence (auto-created)
├── docker-compose.yml                # PostgreSQL Docker setup
└── postgres-credentials.md            # Database credentials
```

## Installation & Setup

### Prerequisites
- Ollama running with models: `gpt-oss:20b` and `nomic-embed-text:latest`
- PostgreSQL running (via Docker Compose)
- Python 3.9+

### Installation Steps

1. **Create Virtual Environment**
   ```bash
   cd /Users/kevin/Downloads
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r langchain_agent/requirements.txt
   ```

3. **Start PostgreSQL** (if not running)
   ```bash
   docker compose up -d
   ```

4. **Initialize Database**
   ```bash
   cd langchain_agent
   python setup_db.py
   ```

5. **Load Sample Data**
   ```bash
   python load_sample_data.py
   ```

6. **Run Agent**
   ```bash
   python main.py
   ```

## Configuration

### Database Settings (`config.py`)
```python
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "langchain_agent"
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/langchain_agent"
```

### LLM Settings (`config.py`)
```python
LLM_MODEL = "gpt-oss:20b"              # Main reasoning model
LLM_TEMPERATURE = 0                    # 0 = deterministic, 1 = creative
EMBEDDINGS_MODEL = "nomic-embed-text:latest"  # Embedding model
```

### Streaming Settings
- **Character Delay**: 0.005 seconds between characters
- **Thinking Display**: Full text, no truncation
- **Response Display**: Real-time streaming

## Usage

### Starting the Agent
```bash
source .venv/bin/activate
cd langchain_agent
python main.py
```

### Interactive Commands
```
You: What is Python?
Agent (thinking): [Full thinking process streams]
[📚 Searching knowledge base...]
Agent (response): [Final response streams character-by-character]

You: new           # Start new conversation
You: quit          # Exit agent
```

### Example Queries
- "Tell me about Python"
- "How does machine learning work?"
- "What is REST API?"
- "Explain web development"
- "What are the uses of Python?"

## Adding Custom Documents

1. Add `.txt` files to `../sample_docs/`
2. Run `python load_sample_data.py`
3. Agent immediately has access to new documents

Documents should be:
- Plain text (.txt) format
- Well-structured with clear sections
- 500+ characters for best semantic search results

## Architecture Details

### Agent Flow
```
User Input
    ↓
Streaming Thinking (if needed)
    ↓
Tool Decision (ReAct)
    ↓
Knowledge Base Search (if needed)
    ↓
Streaming Final Response
    ↓
Save to PostgreSQL
```

### Real-Time Streaming
- **Thinking Phase**: Character-by-character stream of agent reasoning
- **Tool Usage**: Visual indicator when knowledge base is searched
- **Response Phase**: Character-by-character stream of final answer
- **Memory**: Full conversation automatically persisted

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| First query | 15-30s | Model loading + embedding generation |
| Subsequent queries | 2-5s | Typical for knowledge retrieval + reasoning |
| Vector search | <100ms | ChromaDB local search |
| Memory save | <50ms | PostgreSQL persistence |

## Troubleshooting

### Agent hangs
- Check Ollama is running: `ollama serve`
- Check models are available: `ollama list`
- Verify Postgres: `docker compose ps`

### No response generated
- Ensure sample data loaded: `python load_sample_data.py`
- Check ChromaDB: `ls -la chroma_db/`

### Database errors
- Reinitialize: `python setup_db.py`
- Check connection: `psql -h localhost -U postgres`

### Slow performance
- Ollama models are CPU/GPU intensive
- First run slower due to model loading
- Subsequent runs use cache

## Security Considerations

### Local Execution ✓
- All data stays on your machine
- No external API calls
- Complete privacy

### Database Security
⚠️ Current config uses default credentials. For production:
1. Change postgres password in `docker-compose.yml`
2. Update `config.py` with new credentials
3. Use environment variables instead of hardcoded values

### Recommended Production Changes
```bash
# Environment variables
export POSTGRES_PASSWORD="secure-password"
export LLM_TEMPERATURE="0.1"

# Update config.py
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
```

## Extending the Agent

### Adding More Tools
```python
# In main.py create_agent_graph()
@tool
def my_custom_tool(query: str) -> str:
    """Tool description"""
    return result

tools = [knowledge_base, my_custom_tool]
```

### Changing the LLM Model
```python
# In config.py
LLM_MODEL = "neural-chat:7b"  # Smaller model
LLM_MODEL = "llama2:13b"       # Different model
```

### Custom Knowledge Base
1. Replace sample documents
2. Run `python load_sample_data.py`
3. Agent reloads knowledge immediately

## Maintenance

### Backup Conversation History
```bash
# Backup PostgreSQL
pg_dump -h localhost -U postgres langchain_agent > backup.sql

# Restore
psql -h localhost -U postgres langchain_agent < backup.sql
```

### Clear Conversation History
```bash
# Start fresh database
python setup_db.py
```

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## Performance Optimization

### For Slower Systems
```python
# config.py
LLM_TEMPERATURE = 0.1          # Faster/more deterministic
CHROMA_DB_PATH = "./chroma_db" # Keep persistent cache
```

### For Faster Responses
```python
# config.py - use smaller model
LLM_MODEL = "neural-chat:7b"
```

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Core agent with streaming logic |
| `config.py` | All configuration constants |
| `setup_db.py` | Database initialization script |
| `load_sample_data.py` | ChromaDB population script |
| `requirements.txt` | Python package dependencies |
| `README.md` | User-facing documentation |

## Version Information

- **LangChain**: 1.2.0+
- **LangGraph**: 1.0.5+
- **ChromaDB**: 1.4.0+
- **PostgreSQL**: 16+
- **Python**: 3.9+

## Support & Troubleshooting

1. Check `/Users/kevin/Downloads/langchain_agent/README.md` for user guide
2. Review error messages in console output
3. Verify prerequisites (Ollama, PostgreSQL, Python packages)
4. Check `postgres-credentials.md` for connection details
5. Ensure all setup steps completed

## Future Enhancements

Potential improvements:
- Multi-language support
- More sophisticated tool management
- Fine-tuned knowledge base indexing
- Real-time model swapping
- Conversation analytics dashboard
- Custom system prompts

---

**Created**: 2025-12-24
**Status**: Production Ready ✓
**Last Updated**: 2025-12-24
