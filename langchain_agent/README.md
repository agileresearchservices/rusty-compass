# LangChain Agent with Local Knowledge Base & Memory

A fully local, production-grade LangGraph ReAct agent powered by:
- **LLM**: Ollama running `gpt-oss:20b`
- **Vector Store**: ChromaDB for semantic knowledge retrieval
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

### 3. Python Dependencies
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

### Step 1: Initialize the Database
Run this once to create the necessary tables in PostgreSQL:

```bash
python setup_db.py
```

Expected output:
```
✓ Database 'langchain_agent' created successfully
✓ Connected to: PostgreSQL 16.x
✓ Checkpoint tables initialized successfully
✓ Database setup complete!
```

### Step 2: Load Sample Data
Populate ChromaDB with sample documents:

```bash
python load_sample_data.py
```

Expected output:
```
✓ Loaded: python_basics.txt (1234 chars)
✓ Loaded: machine_learning_intro.txt (5678 chars)
✓ Loaded: web_development.txt (9012 chars)
✓ Created vector store at: ./chroma_db
✓ Testing retrieval with query: 'Python'
✓ Found 2 relevant documents
✓ Data loading complete!
```

### Step 3: Start the Agent
Run the interactive agent:

```bash
python main.py
```

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

All conversations are automatically saved to PostgreSQL. You can:

1. **View previous conversations**:
   ```
   You: list
   📋 Previous Conversations:
     1. conversation_a1b2c3d4
     2. conversation_f5e6d7c8
     3. conversation_p9o8n7m6
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

### Components

```
┌─────────────────────────────────────────────────────┐
│                   User Input                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│             LLM (gpt-oss:20b)                       │
│         ReAct Agent Decision Making                  │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐   ┌─────────────────────┐
│ Knowledge Base   │   │   Conversation      │
│ (ChromaDB)       │   │   Memory (Postgres) │
│                  │   │                     │
│ • Documents      │   │ • Chat History      │
│ • Embeddings     │   │ • State/Context     │
│ • Vector Index   │   │ • Thread IDs        │
└──────────────────┘   └─────────────────────┘
```

### Data Flow

1. **User Query** → Agent receives input
2. **Reasoning** → LLM decides if retrieval is needed
3. **Optional Retrieval** → ChromaDB searches knowledge base using embeddings
4. **Response Generation** → LLM generates response
5. **Memory Storage** → Conversation state saved to Postgres
6. **Output** → Response sent to user

## File Structure

```
langchain_agent/
├── main.py                    # Main agent entry point
├── config.py                  # Configuration constants
├── setup_db.py               # Database initialization script
├── load_sample_data.py       # Sample data loader
├── requirements.txt          # Python dependencies
└── README.md                 # This file

../
├── sample_docs/              # Sample documents for knowledge base
│   ├── python_basics.txt
│   ├── machine_learning_intro.txt
│   └── web_development.txt
├── chroma_db/                # ChromaDB persistence (auto-created)
└── docker-compose.yml        # PostgreSQL Docker setup
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
- **CHROMA_DB_PATH**: Vector store location
- **OLLAMA_BASE_URL**: Ollama endpoint URL

### Advanced Features
- **ENABLE_COMPACTION**: Auto-compress long conversations (default: `True`)
- **MAX_CONTEXT_TOKENS**: Context window limit (default: `3000`)
- **COMPACTION_THRESHOLD_PCT**: Trigger compaction at N% full (default: `0.8`)

## Adding Custom Documents

1. Add `.txt` files to the `../sample_docs/` directory
2. Run: `python load_sample_data.py`
3. The agent will now have access to your documents

Documents should be plain text files with clear, informative content. The agent uses semantic search, so well-written documents with good structure work best.

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

### "ChromaDB not initialized"
```bash
# Run the data loader
python load_sample_data.py
```

### "Ollama models not found"
```bash
# Check available models
ollama list

# Pull missing models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
```

### "Agent responses are slow"
This is normal on first run. ChromaDB creates embeddings, which takes time. Subsequent queries will be faster.

### "Port 5432 already in use"
Change the port in docker-compose.yml and update config.py accordingly.

## Performance Notes

- **First Query**: May take 15-30 seconds (model loading, embedding generation)
- **Subsequent Queries**: 2-5 seconds typically
- **Vector Search**: Uses cosine similarity for fast retrieval
- **Conversation Memory**: All messages persisted to Postgres for multi-session continuity

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
