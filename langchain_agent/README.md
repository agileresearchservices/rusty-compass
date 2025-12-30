# LangChain Agent with Local Knowledge Base & Memory

A fully local, production-grade LangGraph ReAct agent powered by:
- **LLM**: Ollama running `gpt-oss:20b`
- **Vector Store**: PostgreSQL with PGVector for semantic knowledge retrieval
- **Memory**: PostgreSQL for persistent conversation history
- **Embeddings**: Ollama `nomic-embed-text:latest`
- **Real-time Streaming**: Character-by-character streaming of agent thinking and final responses

The agent can intelligently decide when to search your local knowledge base, displays its reasoning process in real-time, and maintains conversation context across sessions.

## Prerequisites

Before running the agent, ensure you have:

### 1. Ollama Running
```bash
# Start Ollama service
ollama serve

# In another terminal, verify models are available
ollama list
```

Expected models:
- `gpt-oss:20b` - Main LLM for reasoning
- `nomic-embed-text:latest` - Embedding model for retrieval
- `BAAI/bge-reranker-v2-m3` - Cross-encoder reranker for result relevance (loaded via HuggingFace transformers, not Ollama)

### 2. PostgreSQL Running
Start the Docker container from the parent directory:
```bash
cd ..
docker compose up -d

# Verify it's running
docker compose ps
```

Credentials:
- Host: localhost
- Port: 5432
- User: postgres
- Password: postgres
- Database: postgres
- Connection String: `postgresql://postgres:postgres@localhost:5432/postgres`

### 3. Reranker Model (Optional but Recommended)
The agent includes a cross-encoder reranker to improve search result quality. The model is downloaded automatically on first run from HuggingFace:

```bash
# The bge-reranker-v2-m3 model (~2.3GB) will be downloaded automatically
# On first run, this may take 1-2 minutes depending on your internet speed
# The model is cached in ~/.cache/huggingface/

# (Optional) Pre-download the model to avoid delay on first run:
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3', trust_remote_code=True); AutoModel.from_pretrained('BAAI/bge-reranker-v2-m3', trust_remote_code=True)"
```

Alternative reranker models available in `config.py`:
- `BAAI/bge-reranker-v2-m3` - Fast and accurate (default, recommended)
- `BAAI/bge-reranker-v2-large` - More accurate but slower

See the Configuration section below for details on selecting a different reranker.

### 4. Python Dependencies
```bash
pip install -r requirements.txt
```

## Key Features

### Live Reasoning & Real-Time Streaming Output
The agent displays its thinking process and streams responses character-by-character for an engaging, interactive experience:
- **Live Thinking**: Watch the agent's reasoning process as it thinks through the problem (when reasoning is enabled)
- **Response Streaming**: Final answers appear character-by-character for real-time feedback
- **Knowledge Base Integration**: Intelligently searches local documents when needed
- **Transparent Decision-Making**: See how the agent decides to use tools and formulates responses

Example output:
```
You: What is Python?

Agent (thinking):
The user is asking about Python. This is a general knowledge question about a
programming language. I should search my knowledge base to provide accurate and
comprehensive information...

Agent (response):
Python is a high-level, general-purpose programming language known for its clear,
readable syntax and powerful capabilities...
(text streams character by character)
```

### Persistent Memory
All conversations are automatically saved to PostgreSQL. You can:
- Close and reopen the agent - your conversation context is preserved
- Start a new conversation with the `new` command
- Maintain multiple separate conversation threads

### Local-Only Architecture
- No API calls or external dependencies
- All data stays on your machine
- Complete privacy and control

## Quick Start

### Prerequisites
Before running setup, ensure:
1. **Docker & PostgreSQL**: Running PostgreSQL container (from parent directory)
   ```bash
   docker compose up -d
   ```

2. **Ollama**: Running Ollama service (starts your LLM, embeddings, and reranker models)
   ```bash
   ollama serve
   ```

### Complete Setup (One Command)
Run the unified setup script to initialize everything:

```bash
python setup.py
```

This handles:
- ✅ PostgreSQL database creation
- ✅ Vector extension setup (PGVector)
- ✅ Database tables & indexes (IVFFlat vector, GIN full-text)
- ✅ Conversation memory tables (checkpoints)
- ✅ Pulling Ollama models (LLM, embeddings, reranker)
- ✅ Loading sample documents with embeddings
- ✅ Creating vector embeddings for semantic search
- ✅ Building full-text search indexes

Expected output:
```
======================================================================
LANGCHAIN AGENT - COMPLETE SETUP
======================================================================

[1/7] Creating database...
      ✓ Database 'langchain_agent' created
      ✓ Connected to: PostgreSQL 16.x
[2/7] Enabling PGVector extension...
      ✓ PGVector extension enabled
[3/7] Creating database tables...
      ✓ Database tables created
[4/7] Creating database indexes...
      ✓ Indexes created (IVFFlat vector, GIN full-text)
[5/7] Setting up Ollama models...
      ✓ gpt-oss:20b pulled successfully
      ✓ nomic-embed-text:latest pulled successfully
      ✓ BAAI/bge-reranker-v2-m3 pulled successfully
[6/7] Loading sample documents...
      ✓ Loaded: python_basics.txt
      ✓ Documents: 3
      ✓ Chunks: 11
      ✓ Chunks with embeddings: 11
[7/7] Verifying setup...
      ✓ Setup complete!

You can now run the agent:
  python main.py
```

### Start the Agent (CLI)
Run the interactive command-line agent:

```bash
python main.py
```

### Start the Web UI
Run the web interface with real-time observability:

```bash
# Terminal 1: Start the API server
cd api && uvicorn main:app --reload --port 8000

# Terminal 2: Start the frontend
cd web && npm run dev
```

Then open http://localhost:5173 in your browser.

**Web UI Features:**
- **Conversation Sidebar**: View, switch between, and delete previous conversations
- **Real-time Chat**: Send messages and see streaming responses
- **Observability Panel**: Watch the agent's execution in real-time
  - Step-by-step node execution (Query Evaluator → Agent → Tools → Graders)
  - Document retrieval and reranking details
  - Response grading scores and feedback
- **Agent Graph**: Visual workflow diagram showing node execution
- **Metrics View**: Performance timing breakdown

**Conversation Management:**
- New conversations automatically appear in the sidebar after the first message
- Click any conversation to load its full message history
- Conversations persist across browser sessions

## Usage

### Interactive Conversation with Streaming

The agent responds to your questions with real-time character-by-character streaming:

```
You: What is Python?

Agent (response):
Python is a high-level, general-purpose programming language known for its clear, readable syntax...
(response streams character by character in real-time)

You: Tell me more about machine learning

Agent (response):
Machine Learning is a subset of artificial intelligence that enables systems to learn and improve...

You: new
✓ New conversation started
Conversation ID: conversation_a1b2c3d4

You: What did we just talk about?
Agent: [New conversation - previous context is not available]

You: quit
Goodbye!
```

### Commands

| Command | Description |
|---------|-------------|
| `your question` | Ask the agent anything about the knowledge base |
| `new` | Start a new conversation (generates new ID) |
| `list` | Show all previous conversations with titles and dates |
| `load <id>` | Resume a specific conversation by ID |
| `clear` | Delete all conversations and history (with confirmation) |
| `quit` / `exit` | Exit the agent |
| `Ctrl+C` | Force exit |

### Resuming Previous Conversations

All conversations are automatically saved to PostgreSQL with **LLM-generated titles** that summarize the conversation content. Titles are updated after each message to reflect the evolving discussion.

1. **View previous conversations**:
   ```
   You: list
   📋 Previous Conversations:
     1. RAG Explained
        ID: conversation_c1bb5ee0 | 2025-12-28 19:04
     2. Human-in-the-Loop Overview
        ID: conversation_99af0063 | 2025-12-28 19:15
     3. What is langsmith?
        ID: conversation_a7c16fd7 | 2025-12-28 18:49
   ```

2. **Resume a specific conversation**:
   ```
   You: load conversation_a1b2c3d4
   ✓ Loaded conversation: conversation_a1b2c3d4

   You: What did we talk about earlier?
   Agent: [Context from previous conversation is available]
   ```

3. **Start a fresh conversation**:
   ```
   You: new
   ✓ New conversation started
   Conversation ID: conversation_newid123
   ```

Each conversation maintains its own:
- Full message history
- User context and queries
- Agent responses
- Retrieved knowledge base references

### Clearing All Conversations

To delete all previous conversations and start fresh:

```
You: clear
⚠️  This will delete ALL conversations and history. Continue? (yes/no): yes
✓ Cleared 6 conversation(s) and 42 checkpoint record(s)
```

The `clear` command:
- Deletes all conversation metadata and titles
- Removes all checkpoint/history records
- Requires confirmation to prevent accidental deletion
- Type `yes` to confirm, anything else to cancel

### Example Questions

Try these to test the agent's knowledge retrieval:

- "What are Python's main features?"
- "Explain supervised learning"
- "How does REST API work?"
- "What is machine learning used for?"
- "Tell me about web security"

## Architecture

### System Components

```mermaid
flowchart TB
    subgraph User["User Interface"]
        CLI["CLI (main.py)"]
        API["FastAPI + WebSocket"]
    end

    subgraph Agent["LangGraph ReAct Agent"]
        QE["Query Evaluator"]
        REACT["ReAct Loop"]
        TOOLS["Knowledge Base Tool"]
    end

    subgraph Search["Hybrid Search"]
        VS["Vector Search<br/>(768-dim)"]
        TS["Full-Text Search"]
        RRF["RRF Fusion"]
        RERANK["BGE Reranker"]
    end

    subgraph Reflection["Reflection Loop"]
        DG["Document Grader"]
        QT["Query Transformer"]
        RG["Response Grader"]
    end

    subgraph Storage["PostgreSQL + PGVector"]
        DOCS["Documents"]
        CHUNKS["Chunks + Embeddings"]
        MEM["Checkpoints"]
    end

    subgraph LLM["Ollama"]
        MODEL["gpt-oss:20b"]
        EMB["nomic-embed-text"]
    end

    User --> Agent
    Agent --> Search
    Search --> Reflection
    Reflection --> Agent
    Agent --> LLM
    Search --> Storage
    Agent --> MEM
```

### Data Flow

```mermaid
flowchart LR
    Q["1. User Query"] --> QE["2. Query Evaluator<br/>Classify → Set λ"]
    QE --> REACT["3. ReAct Agent"]
    REACT --> HYBRID["4. Hybrid Search<br/>Vector + Full-text"]
    HYBRID --> RRF["5. RRF Fusion<br/>15 candidates"]
    RRF --> RERANK["6. BGE Reranker<br/>→ Top 4"]
    RERANK --> DG["7. Document Grader"]
    DG --> GEN["8. Generate Response"]
    GEN --> RG["9. Response Grader"]
    RG --> STREAM["10. Stream Output"]
```

### Processing Steps

1. **User Query** → Agent receives input
2. **Query Evaluation** → LLM classifies query type using few-shot examples, sets lambda (vector vs. lexical weight)
   - Uses 8 few-shot examples covering lexical (versions, errors) to semantic (conceptual questions)
   - Key insight: "Specific" ≠ "Semantic" - version numbers need lexical search despite being specific
3. **ReAct Reasoning** → LLM decides if retrieval is needed
4. **Hybrid Search** → PostgreSQL with PGVector performs combined search:
   - Vector search: Semantic similarity (768-dim embeddings)
   - Full-text search: Keyword matching (tsvector/tsquery)
   - RRF Fusion: Combines rankings (~15 candidates)
5. **Cross-Encoder Reranking** → BGE model scores relevance (→ top 4)
6. **Document Grading** → LLM evaluates each document's relevance
   - If documents fail: Query Transformer rewrites and retries (max 2)
7. **Response Generation** → LLM generates response from graded documents
8. **Response Grading** → LLM evaluates quality (relevance, completeness, clarity)
   - If response fails: Response Improver adds feedback and retries (max 2)
9. **Title Generation** → LLM generates/updates conversation title
10. **Memory Storage** → State saved to PostgreSQL checkpoints
11. **Streaming Output** → Response streamed token-by-token

## File Structure

```
langchain_agent/
├── main.py                    # Main agent entry point with reflection loop
├── setup.py                   # Unified setup script (one command setup)
├── config.py                  # Configuration constants
├── requirements.txt           # Python dependencies
├── test_reranker.py           # Cross-encoder reranker tests (6 tests)
├── test_hybrid_search.py      # Hybrid search strategy validation
├── test_query_evaluator.py    # Query evaluation & dynamic lambda testing
├── test_reflection.py         # Reflection loop tests (6 tests)
└── README.md                  # This file

../
├── sample_docs/               # Sample documents for knowledge base
│   ├── python_basics.txt
│   ├── machine_learning_intro.txt
│   └── web_development.txt
└── docker-compose.yml         # PostgreSQL + PGVector Docker setup
```

## Configuration

Edit `config.py` to customize:

### LLM & Reasoning Settings
- **LLM_MODEL**: Change the language model (default: `gpt-oss:20b`)
- **LLM_TEMPERATURE**: Control response creativity (0=deterministic, 1=creative)
- **REASONING_ENABLED**: Enable live thinking output (default: `True`)
  - When enabled, the agent displays its reasoning process before the final response
  - Requires a reasoning-capable model like `gpt-oss:20b`
- **REASONING_EFFORT**: Control depth of reasoning (default: `"medium"`)
  - Options: `"low"`, `"medium"`, `"high"`
  - Higher effort takes more time but produces more thorough reasoning
- **EMBEDDINGS_MODEL**: Change embedding model

### Storage & Connection
- **POSTGRES_HOST/PORT**: Database connection settings
- **VECTOR_COLLECTION_NAME**: PostgreSQL vector collection name (default: `local_knowledge`)
- **VECTOR_DIMENSION**: Embedding vector dimension (default: 768 for nomic-embed-text)
- **VECTOR_INDEX_TYPE**: Index strategy - `ivfflat` (faster queries) or `hnsw` (faster updates)
- **VECTOR_SIMILARITY_METRIC**: Similarity metric - `cosine`, `l2`, or `inner_product`
- **OLLAMA_BASE_URL**: Ollama endpoint URL

### Retriever Configuration
- **RETRIEVER_K**: Number of documents to retrieve per query (default: 4)
- **RETRIEVER_FETCH_K**: Number of documents to fetch before filtering (default: 40)
- **RETRIEVER_LAMBDA_MULT**: Hybrid search weight for vector vs keyword search (default: 0.25)
  - 0.0 = pure lexical/full-text search
  - 0.25 = 75% lexical + 25% semantic (default)
  - 0.5 = balanced hybrid search
  - 1.0 = pure semantic/vector search
- **RETRIEVER_SEARCH_TYPE**: Search method - `"similarity"` (vector-only) or `"hybrid"` (vector + full-text)

### Reranker Configuration (Cross-Encoder Re-Ranking)
The agent uses a cross-encoder model to rerank hybrid search results for improved relevance before sending to the LLM.

- **ENABLE_RERANKING**: Enable cross-encoder reranking (default: `True`)
- **RERANKER_MODEL**: Cross-encoder model to use (default: `"BAAI/bge-reranker-v2-m3"`)
  - Alternatives:
    - `"BAAI/bge-reranker-v2-m3"` - Fast, compact (~2.3GB), recommended (default)
    - `"BAAI/bge-reranker-v2-large"` - Better accuracy, larger model (~1.2GB)
- **RERANKER_FETCH_K**: Number of candidates to fetch before reranking (default: 25)
- **RERANKER_TOP_K**: Final number of documents after reranking (default: 4)

**How It Works:**
1. Hybrid search retrieves 25 candidate documents using combined vector + lexical search
2. Reranker scores each document's relevance to the query (0.0-1.0)
3. Top 4 highest-scoring documents are selected
4. Final 4 documents sent to LLM for response generation

**Performance Impact:**
- Query latency: ~1-2 seconds additional for reranking 15 documents
- Quality: 20-40% improvement in answer relevance for complex queries
- Accuracy: Reduces irrelevant results from reaching the LLM

**Disabling Reranking:**
To disable reranking and fall back to standard hybrid search:
```python
# In config.py
ENABLE_RERANKING = False
RERANKER_FETCH_K = 4  # Back to standard retrieval
```

### Reflection Configuration (Self-Improving Agent)

The agent includes a reflection loop that grades retrieved documents and response quality, with automatic query transformation when retrieval fails.

- **ENABLE_REFLECTION**: Master switch for reflection loop (default: `True`)
- **ENABLE_DOCUMENT_GRADING**: Grade retrieved documents for relevance (default: `True`)
- **ENABLE_RESPONSE_GRADING**: Evaluate final response quality with strict criteria (default: `True`)
  - Fails responses that are vague, missing code examples, lack structure, or are incomplete
- **ENABLE_QUERY_TRANSFORMATION**: Rewrite query if documents are poor (default: `True`)
- **REFLECTION_MAX_ITERATIONS**: Maximum retrieval/response attempts (default: `2`, allows 1 retry)
- **REFLECTION_MIN_RELEVANT_DOCS**: Minimum relevant docs to pass grading (default: `1`)
- **REFLECTION_DOC_SCORE_THRESHOLD**: Minimum average relevance score (default: `0.3`)
- **REFLECTION_SHOW_STATUS**: Display reflection status in output (default: `True`)

**How It Works:**

```mermaid
flowchart TD
    QE["Query Evaluator"] --> AGENT["Agent"]
    AGENT --> TOOLS["Retrieve Documents"]
    TOOLS --> DG["Document Grader"]

    DG -->|"docs pass"| GEN["Generate Response"]
    DG -->|"docs fail & retry < max"| QT["Query Transformer"]
    QT --> QE

    GEN --> RG["Response Grader"]
    RG -->|"response good"| END["Stream to User"]
    RG -->|"response bad & retry < max"| RI["Response Improver"]
    RI --> AGENT

    style DG fill:#e1f5fe
    style RG fill:#e1f5fe
    style QT fill:#fff3e0
    style RI fill:#fff3e0
```

1. After retrieving documents, the **Document Grader** evaluates each document's relevance
2. If documents fail grading (< 2 relevant docs or avg score < 0.5), the **Query Transformer** rewrites the query
3. The agent retries with the transformed query (max 2 iterations)
4. After generating a response, the **Response Grader** evaluates quality
5. If response fails grading (incomplete, truncated, or low quality), the **Response Improver** adds feedback and triggers a retry (max 2 iterations)
6. Reflection status is displayed in the console output

**Example Output:**
```
[Document Grader] ✓ 3/4 documents relevant (avg score: 0.78)
[Response Grader] ✓ PASS (score: 0.85)
  Answer is comprehensive and addresses the query directly.
```

**Query Transformation Example:**
```
[Document Grader] ✗ 0/4 documents relevant (avg score: 0.15)
[Reflection] Documents failed grading. Retry 1/2
[Query Transformer] 'quantum basics' → 'introduction to quantum computing fundamentals'
```

**Response Retry Example:**
```
[Response Grader] ✗ FAIL (score: 0.10)
  The response does not explain the concept and is incomplete.
[Reflection] Response failed grading. Retry 1/2
[Response Grader] ✓ PASS (score: 0.90)
  The response accurately defines the concept and is well-structured.
```

**Anti-Hallucination Protection:**

When no relevant documents are found in the knowledge base, the Response Grader includes special handling to prevent hallucination:

1. **Context-Aware Grading**: The grader receives document grading results and knows when no relevant docs exist
2. **Hallucination Detection**: Responses that fabricate information when no documents support it are marked as FAIL
3. **Honest Response Pass**: Responses that honestly acknowledge "I couldn't find that information" automatically pass
4. **Corrective Feedback**: If the LLM initially hallucinates, the Response Improver instructs it to acknowledge the lack of information

Example:
```
[Document Grader] ✗ 0/4 documents relevant (avg score: 0.12)
[Response Grader] ✗ FAIL (score: 0.30)
  The assistant provides detailed information not supported by any retrieved documents (hallucination).
[Response Improver] Feedback: Acknowledge the lack of information rather than present fabricated details.
[Response Grader] ✓ PASS (score: 0.90)
  The response honestly acknowledges the information is not available.
```

**Performance Impact:**
- Document grading: ~2-4s per query (LLM evaluates each document)
- Response grading: ~1-2s per query
- Query transformation (if triggered): ~1-2s
- Response retry (if triggered): ~3-5s per retry (includes re-generation)
- Title generation: ~1-2s per turn
- Total worst case (failed retrieval + failed response with retries): ~15-20s additional

**Disabling Reflection:**
```python
# In config.py
ENABLE_REFLECTION = False  # Disable all reflection features
```

### Advanced Features
- **ENABLE_COMPACTION**: Auto-compress long conversations (default: `True`)
- **MAX_CONTEXT_TOKENS**: Context window limit (default: `3000`)
- **COMPACTION_THRESHOLD_PCT**: Trigger compaction at N% full (default: `0.8`)

## Adding Custom Documents

1. Add `.txt` files to the `../sample_docs/` directory
2. Run: `python load_sample_data_pgvector.py`
3. The agent will now have access to your documents in the PostgreSQL vector store

Documents should be plain text files with clear, informative content. The agent uses semantic search, so well-written documents with good structure work best.

### Document Chunking Strategy

The data loader automatically chunks documents for improved semantic search accuracy:

- **Chunk Size**: 1000 characters per chunk
- **Overlap**: 200 characters between adjacent chunks (prevents losing context at boundaries)
- **Storage**: Full documents stored in `documents` table; chunks with embeddings stored in `document_chunks` table
- **Retrieval**: Similarity search queries the `document_chunks` table for precise, context-aware results

This chunking approach ensures that:
- Large documents are broken into semantically meaningful pieces
- Context is preserved across chunk boundaries via overlap
- Vector similarity search returns relevant chunks, not entire documents
- Memory is efficiently used while maintaining semantic accuracy

## Troubleshooting

### "Cannot connect to Postgres"
```bash
# Check if Docker container is running
docker compose ps

# Start it if needed
docker compose up -d

# Test connection
psql -h localhost -U postgres -d postgres
```

### "Vector store has no documents"
```bash
# Run the PostgreSQL data loader
python load_sample_data_pgvector.py
```

### "PGVector extension not found"
```bash
# Initialize the database with PGVector support
python setup_db.py
```

### "Ollama models not found"
```bash
# Check available models
ollama list

# Pull missing models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
```

### Reranker Troubleshooting

#### "Reranker model download is slow / timing out"
The bge-reranker-v2-m3 model is ~2.3GB. On first run, it downloads from HuggingFace:
```bash
# Monitor the download progress
# Models are cached in ~/.cache/huggingface/

# If download fails/times out, check your internet connection and retry
```

#### "CUDA out of memory / GPU memory exceeded"
The reranker loads the model into GPU memory:
```python
# In config.py, disable reranking if needed:

# Option 1: Use CPU-based reranking (slower but uses system RAM)
# Set in config.py and model will auto-detect CPU if CUDA fails

# Option 2: Disable reranking entirely
ENABLE_RERANKING = False
```

#### "Reranking makes queries slower"
This is expected - reranking adds ~1-2 seconds per query. Trade-off:
- **Without reranking**: Fast (~600ms) but sometimes less relevant results
- **With reranking**: Slower (~2-3s) but better result quality

If speed is critical:
```python
# In config.py
ENABLE_RERANKING = False  # Disable to get back to ~600ms query time
```

#### "Reranker scores are all the same / uniform"
This can indicate the model output format differs from expected. Check:
1. The model is downloaded correctly: `ls ~/.cache/huggingface/hub/ | grep bge`
2. Check transformers version: `pip list | grep transformers` (should be 4.40+)

#### "ImportError: cannot import AutoTokenizer"
The transformers library is not installed:
```bash
pip install --upgrade transformers torch
```

#### "No module named torch"
PyTorch is not installed:
```bash
# Install CPU-only version (smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or GPU version (NVIDIA CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Agent responses are slow"
This is normal on first run. The embedding model takes time to load and generate embeddings. Subsequent queries will be faster. Vector search with PGVector is typically 100-500ms for similarity search operations.

### "Port 5432 already in use"
Change the port in docker-compose.yml and update config.py accordingly.

## Performance Notes

- **First Query**: May take 15-30 seconds (model loading, embedding generation)
- **Subsequent Queries**: 2-5 seconds typically
- **Vector Search**: PGVector with IVFFlat indexing provides sub-100ms similarity search for static knowledge bases
- **Embedding Generation**: ~50-200ms per document depending on content length
- **Conversation Memory**: All messages persisted to PostgreSQL for multi-session continuity
- **Index Type Impact**: IVFFlat (used by default) optimizes read performance; HNSW available for frequent updates

## Persistence & Memory

Each conversation is assigned a unique `thread_id`. This means:

- ✓ Close and restart the agent - context is preserved
- ✓ Multiple conversations can run in parallel with different thread IDs
- ✓ All conversation history is stored in PostgreSQL
- ✓ No conversation history is lost

To start fresh, use the `new` command in the chat.

### Smart Conversation Compaction

When conversations get very long (20+ messages), the agent automatically applies **smart compaction** to prevent exceeding the LLM's context window:

**How It Works:**
- Monitors estimated token count (1 token ≈ 4 characters)
- When conversation exceeds 80% of context limit (~2400 tokens)
- Summarizes older messages while keeping recent messages in full
- Always preserves the last 10 messages uncompacted

**Example:**
```
You: [Message 40 in a long conversation]

[🗜️  Compacted 30 older messages to maintain context]

Agent (response):
[Agent generates response with full context of older topics via summary]
```

**Configuration (in `config.py`):**
```python
ENABLE_COMPACTION = True                # Master switch
MAX_CONTEXT_TOKENS = 3000              # Context limit
COMPACTION_THRESHOLD_PCT = 0.8         # Trigger at 80%
MESSAGES_TO_KEEP_FULL = 10             # Always preserve last N
MIN_MESSAGES_FOR_COMPACTION = 20       # Don't compact small conversations
```

**Benefits:**
- Conversations can now exceed 50+ messages without losing context
- Automatic - no user intervention needed
- Preserves key information through smart summarization
- Recent messages always available in full

## Security Considerations

- **Local Only**: All data stays on your machine
- **No API Calls**: Uses local Ollama, ChromaDB, and PostgreSQL
- **Database**: Default credentials in config.py (change for production)
- **HTTPS**: Not needed for local development

## Testing the System

### Test the Reranker
Verify the cross-encoder reranker is working correctly:

```bash
# Run the reranker test suite (6 comprehensive tests)
python test_reranker.py
```

Expected output:
```
======================================================================
BGE-RERANKER-V2-M3 CROSS-ENCODER TESTS
======================================================================

Test 1: Machine Learning Query
✓ Reranking completed
  BEFORE (Original Order):
    1. [ml_guide.txt] Machine learning is a subset of artificial intelligence...
    2. [neural_nets.txt] Neural networks are computational models...
    3. [python_guide.txt] Python is a high-level programming language...

  AFTER (Reranked by Relevance Score):
    1. score=0.9234 [ml_guide.txt] Machine learning is a subset of artificial intelligence...
    2. score=0.7891 [neural_nets.txt] Neural networks are computational models...
    3. score=0.6543 [python_guide.txt] Python is a high-level programming language...

[... 5 more tests ...]

======================================================================
TEST SUMMARY
======================================================================
Passed: 6/6
Failed: 0/6
✓ All tests passed! BGE-Reranker is working correctly.
```

### Test Hybrid Search
Compare different search strategies (vector-only, lexical-only, balanced hybrid):

```bash
python test_hybrid_search.py
```

This tests the impact of the `lambda_mult` parameter on search results.

### Test Query Evaluation
Verify dynamic lambda adjustment based on query type classification:

```bash
python test_query_evaluator.py
```

### Test Reflection Loop
Verify document grading, response grading, and query transformation:

```bash
python test_reflection.py
```

Expected output:
```
======================================================================
REFLECTION LOOP TESTS
======================================================================

Test 1: Configuration Loading
  ENABLE_REFLECTION: True
  ...
✓ Configuration loaded successfully

Test 2: State Type Definitions
  ✓ CustomAgentState.iteration_count defined
  ✓ CustomAgentState.document_grades defined
  ...
✓ State types defined correctly

Test 3: Graph Creation with Reflection Nodes
  Graph nodes: ['query_evaluator', 'agent', 'tools', 'document_grader',
                'query_transformer', 'response_grader', ...]
✓ Graph creation test passed

Test 4: Document Grader Logic
  [Document Grader] ✓ 1/3 documents relevant (avg score: 0.27)
✓ Document grader test passed

Test 5: Response Grader Logic
  [Response Grader] ✓ PASS (score: 0.90)
✓ Response grader test passed

Test 6: Query Transformer Logic
  [Query Transformer] 'quantum computing basics' → 'introduction to quantum computing fundamentals'
✓ Query transformer test passed

======================================================================
TEST SUMMARY
======================================================================
  Total: 6/6 tests passed
✓ All tests passed! Reflection loop is working correctly.
```

## Extending the Agent

### Add More Tools
Modify `main.py` to add additional tools:

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(input: str) -> str:
    """Description of what this tool does"""
    return result

# Add to tools list in create_agent_graph()
tools = [knowledge_base, my_custom_tool]
```

### Change the LLM Model
In `config.py`:
```python
LLM_MODEL = "neural-chat:7b"  # Use a smaller model
LLM_MODEL = "llama2:13b"       # Use a different model
```

### Customize Reranking
Switch between different cross-encoder models based on your needs:

```python
# In config.py

# Option 1: Fast and compact (Default, recommended)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # ~2.3GB, fast

# Option 2: Large (Maximum accuracy)
RERANKER_MODEL = "BAAI/bge-reranker-v2-large"  # ~1.2GB, slower but more accurate

# Option 3: Disable reranking (Fastest queries, no reranking)
ENABLE_RERANKING = False
RERANKER_FETCH_K = 4  # Go back to standard retrieval
```

**Model Comparison:**

| Model | Size | Speed | Accuracy | Languages | Recommendation |
|-------|------|-------|----------|-----------|---|
| BAAI/bge-reranker-v2-m3 | 2.3GB | Fast | Good | 100+ | **Default** |
| BAAI/bge-reranker-v2-large | 1.2GB | Medium | Very Good | 100+ | Higher accuracy |

### Customize Retrieval
Modify the retriever settings in `main.py`:

```python
retriever = self.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 results
)
```

## License

This project uses open-source components. See individual package licenses for details.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all prerequisites are running
3. Check logs for error messages
4. Review config.py for correct settings

## Next Steps

- Add your own documents to `../sample_docs/`
- Customize the LLM temperature in config.py
- Explore different Ollama models
- Integrate additional tools for specialized tasks
