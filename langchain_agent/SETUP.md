# Complete Setup Guide

A detailed guide for setting up the LangChain Agent from scratch.

## Architecture Overview

The system is built on three core components:

1. **PostgreSQL + PGVector** - Vector database for semantic search
2. **Ollama** - Local LLM runtime (LLM, embeddings, reranker)
3. **LangGraph** - Agent orchestration with memory management

## Prerequisites

### 1. System Requirements

- **OS**: macOS, Linux, or Windows (with WSL2)
- **Python**: 3.10+
- **RAM**: 16GB+ (for Ollama models)
- **Disk**: 50GB+ (for Ollama models, database)

### 2. Docker & PostgreSQL

Ensure Docker is installed and running:

```bash
# Start PostgreSQL with PGVector support
docker compose up -d

# Verify it's running
docker compose ps
# Should show: postgres service with status "Up"

# Test connection
psql -h localhost -U postgres -d postgres
# Command: \q to exit
```

### 3. Ollama

Install and start Ollama:

```bash
# Install Ollama from https://ollama.com
# Or if already installed, start the service

ollama serve
# Runs on http://localhost:11434 (default)
```

## Complete Setup (Automated)

Run the unified setup script:

```bash
python setup.py
```

This handles all initialization in 7 steps:

### Step 1-4: PostgreSQL Setup
```
[1/7] Creating database...
      ✓ Database creation
      ✓ Connection verification
[2/7] Enabling PGVector extension...
      ✓ Vector extension enabled
[3/7] Creating database tables...
      ✓ Collections table
      ✓ Documents table
      ✓ Document chunks table
      ✓ Conversation metadata table
      ✓ LangGraph checkpoint tables
[4/7] Creating database indexes...
      ✓ IVFFlat vector indexes (fast similarity search)
      ✓ GIN full-text indexes (keyword search)
      ✓ Collection indexes (filtering)
```

### Step 5: Ollama Model Pulling
```
[5/7] Setting up Ollama models...
      [LLM model] gpt-oss:20b
            Pulling gpt-oss:20b...
            ✓ Model pulled successfully
      [Embeddings model] nomic-embed-text:latest
            Pulling nomic-embed-text:latest...
            ✓ Model pulled successfully
      [Reranker model] BAAI/bge-reranker-v2-m3
            Pulling BAAI/bge-reranker-v2-m3...
            ✓ Model pulled successfully
```

### Step 6-7: Data Loading
```
[6/7] Loading sample documents...
      Loading 3 documents from ../sample_docs/
      Processing: python_basics.txt
            → Split into 3 chunks
            → Generating embeddings...
            ✓ Generated 3 embeddings
            ✓ Loaded 3 chunks
      [... more documents ...]
      ✓ Successfully loaded 11 total chunks
[7/7] Verifying setup...
      ✓ Documents: 3
      ✓ Chunks: 11
      ✓ Chunks with embeddings: 11

✓ SETUP COMPLETE!
```

## Manual Setup (Step by Step)

If you prefer to understand each step:

### Step 1: Database & Vector Extension

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres

# Create database
CREATE DATABASE langchain_agent;
\c langchain_agent

# Enable PGVector extension
CREATE EXTENSION vector;

# Exit
\q
```

### Step 2: Create Tables

```bash
python -c "
import psycopg
from config import DATABASE_URL

with psycopg.connect(DATABASE_URL) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Create tables (from setup.py)
        cur.execute('''CREATE TABLE IF NOT EXISTS collections ...''')
        # [see setup.py for full SQL]
"
```

### Step 3: Create Indexes

```bash
# Handled by setup.py
python setup.py
# Select step 4 manually if needed
```

### Step 4: Pull Ollama Models

```bash
# LLM model
ollama pull gpt-oss:20b

# Embeddings model
ollama pull nomic-embed-text:latest

# Reranker model (downloaded via HuggingFace on first run)
# BAAI/bge-reranker-v2-m3
```

### Step 5: Load Sample Data

```bash
python -c "
from setup import load_sample_data, verify_data_load
chunk_count = load_sample_data()
verify_data_load()
"
```

## Configuration

All settings are in `config.py`:

```python
# Ollama models
LLM_MODEL = "gpt-oss:20b"
EMBEDDINGS_MODEL = "nomic-embed-text:latest"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
OLLAMA_BASE_URL = "http://localhost:11434"

# PostgreSQL
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_DB = "langchain_agent"

# Vector settings
VECTOR_DIMENSION = 768  # nomic-embed-text dimension
VECTOR_INDEX_TYPE = "ivfflat"  # fast for read-heavy
VECTOR_SIMILARITY_METRIC = "cosine"

# Retrieval
RETRIEVER_K = 4  # final documents
RETRIEVER_FETCH_K = 30  # candidates before reranking
RETRIEVER_LAMBDA_MULT = 0.25  # 75% lexical + 25% semantic

# Reranking
ENABLE_RERANKING = True  # always enabled
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_FETCH_K = 15  # candidates to rerank
RERANKER_TOP_K = 4  # final documents after reranking
```

### Alternative Models

**Different LLM**:
```python
# Smaller, faster
LLM_MODEL = "mistral:7b"

# Larger, more capable
LLM_MODEL = "llama2:13b"
```

**Different Embeddings**:
```python
# (Only nomic-embed-text:latest officially tested)
EMBEDDINGS_MODEL = "nomic-embed-text:latest"
```

**Different Reranker**:
```python
# Default, fast (~2.3GB)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Larger, more accurate (~1.2GB)
RERANKER_MODEL = "BAAI/bge-reranker-v2-large"
```

## Troubleshooting

### PostgreSQL Connection Failed

```
✗ Error: cannot connect to Postgres
```

**Solution**:
```bash
# Check if Docker container is running
docker compose ps

# Start if needed
docker compose up -d

# Test connection
psql -h localhost -U postgres
```

### PGVector Extension Not Found

```
✗ Error: extension "vector" does not exist
```

**Solution**:
```bash
# Restart setup to enable extension
python setup.py
# Or manually enable
psql -h localhost -U postgres -d langchain_agent
# CREATE EXTENSION vector;
```

### No Sample Documents Found

```
✗ No .txt files found in ../sample_docs/
```

**Solution**:
```bash
# Add sample files to ../sample_docs/
ls ../sample_docs/  # Should show .txt files

# Then run setup again
python setup.py
```

### Ollama Models Not Pulling

```
✗ ollama command not found
```

**Solution**:
```bash
# Install Ollama from https://ollama.com
# Or if installed, ensure it's in PATH

which ollama
# Should show /usr/local/bin/ollama (or similar)

# Start Ollama service
ollama serve
```

### Reranker Model Download Issues

```
⚠ Timeout pulling BAAI/bge-reranker-v2-m3
```

**Solution**:
Check your internet connection and retry. The model is ~2.3GB and should download in 1-2 minutes:
```python
# Alternatively, disable reranking in config.py
ENABLE_RERANKING = False
```

## Verification

After setup, verify everything works:

```bash
# 1. Test database connection
python -c "
import psycopg
from config import DATABASE_URL
with psycopg.connect(DATABASE_URL) as conn:
    print('✓ Database connected')
"

# 2. Test embeddings
python -c "
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDINGS_MODEL, OLLAMA_BASE_URL
embeddings = OllamaEmbeddings(
    model=EMBEDDINGS_MODEL,
    base_url=OLLAMA_BASE_URL
)
result = embeddings.embed_query('test')
print(f'✓ Embeddings working (768 dimensions)')
"

# 3. Test LLM
python -c "
from langchain_ollama import ChatOllama
from config import LLM_MODEL, OLLAMA_BASE_URL
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
result = llm.invoke('Say hello')
print(f'✓ LLM working: {result.content[:50]}')
"

# 4. Test reranker
python test_reranker.py
# Should show: 6/6 tests passing

# 5. Start the agent
python main.py
# Try: "What is Python?"
```

## Performance Expectations

### First Run Times
- Model pulling: 2-10 minutes (~2.3GB download for BGE reranker)
- Database setup: < 1 minute
- Sample data load: 2-5 minutes (embeddings generation)
- Total: 10-40 minutes depending on internet speed

### Query Times
- Hybrid search: ~600ms (vector + full-text)
- Reranking: ~1-2 seconds (15 documents scored)
- LLM response: 5-30 seconds
- Total: 6-32 seconds per query

### Database Size
- PostgreSQL container: ~2GB
- Vector indexes: ~100MB
- Full-text indexes: ~50MB
- Sample data: ~1MB

## Next Steps

1. **Run the Agent**:
   ```bash
   python main.py
   ```

2. **Try Example Queries**:
   - "What is Python programming?"
   - "How does machine learning work?"
   - "Tell me about web development"

3. **Add Custom Documents**:
   - Place `.txt` files in `../sample_docs/`
   - Re-run `python setup.py`

4. **Customize Configuration**:
   - Edit `config.py`
   - Adjust `RETRIEVER_K`, `RERANKER_MODEL`, etc.
   - Restart agent

## Monitoring

### Check Database

```bash
psql -h localhost -U postgres -d langchain_agent

# Document count
SELECT COUNT(*) FROM documents;

# Chunk count
SELECT COUNT(*) FROM document_chunks;

# Conversation count
SELECT COUNT(*) FROM conversation_metadata;

# Index usage
SELECT schemaname, tablename, indexname FROM pg_indexes
WHERE tablename IN ('documents', 'document_chunks');
```

### Check Ollama

```bash
# List available models
ollama list

# Check model details
curl http://localhost:11434/api/tags

# Monitor model memory usage
# (Check Ollama process in Activity Monitor / Task Manager)
```

### Monitor Agent

```bash
# Check logs during queries (agent prints observability)
python main.py

# Watch database with psql open in another terminal
watch 'psql -h localhost -U postgres -d langchain_agent -c "SELECT COUNT(*) FROM checkpoint_writes"'
```

## Production Considerations

### For Production Deployment

1. **Database**:
   - Use managed PostgreSQL (AWS RDS, GCP Cloud SQL)
   - Enable SSL/TLS connections
   - Regular backups
   - Read replicas for scaling

2. **Ollama**:
   - Deploy as containerized service
   - Load balancer for multiple instances
   - Monitor GPU/CPU usage
   - Queue for concurrent requests

3. **Configuration**:
   - Use environment variables (not hardcoded)
   - Separate configs per environment
   - Secrets management (passwords, API keys)

4. **Monitoring**:
   - Application logging (not just prints)
   - Database performance metrics
   - Model inference latency
   - Error rates and alerting

5. **Scaling**:
   - Connection pooling (already implemented)
   - Batch request processing
   - Caching layer for embeddings
   - Async processing for background tasks

## Support

For issues:
1. Check **Troubleshooting** section above
2. Review `config.py` settings
3. Check logs and error messages
4. Verify all prerequisites are running
