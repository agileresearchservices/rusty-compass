# Claude Code Integration Guide

This guide explains how to use Claude Code effectively with the Rusty Compass project.

## Project Overview

**Rusty Compass** is a production-grade LangGraph ReAct agent with:
- Local LLM reasoning (gpt-oss:20b via Ollama)
- Semantic search with PostgreSQL + PGVector
- Cross-encoder reranking (Qwen3-Reranker-8B)
- Persistent conversation memory
- Real-time streaming output

See [README.md](README.md) for full project overview.

## Quick Start for Claude Code Users

### Initial Setup

```bash
# 1. Clone and navigate
git clone <repo>
cd rusty-compass

# 2. Start Docker PostgreSQL
docker compose up -d

# 3. Run unified setup (handles all initialization)
cd langchain_agent
python setup.py

# 4. Start the agent
python main.py
```

See [langchain_agent/SETUP.md](langchain_agent/SETUP.md) for detailed setup instructions.

## Using Claude Code with This Project

### Code Review Workflow

Before committing changes:

```bash
# 1. Make your changes to main.py, config.py, etc.

# 2. Run tests to validate
make test

# 3. Use Claude Code to review
# Ask Claude: "Review my changes to [file]"
# Claude will check: type safety, error handling, consistency

# 4. Lint and format
make lint
make format

# 5. Commit when ready
git add .
git commit -m "Your change description"
```

### Recommended Claude Code Commands

**For understanding the codebase:**
```
claude --help                    # See available commands
claude code explore             # Browse project structure
claude code document            # Generate documentation
```

**For development:**
```
claude code review              # Review code quality
claude code explain [file]      # Understand a specific file
claude code suggest [file]      # Get improvement suggestions
```

**For testing:**
```
make test                       # Run all test suites
make test-reranker              # Test reranker specifically
```

### Common Development Tasks

#### Adding a New Tool to the Agent

1. **Edit `langchain_agent/main.py`** (around line 800)
2. **Create tool function** with `@tool` decorator
3. **Add to tools list** in `create_agent_graph()`
4. **Test** with sample queries
5. **Document** in DEVELOPER.md

Ask Claude: "Help me add a new tool called [tool_name] to the agent"

#### Modifying Configuration

1. **Edit `langchain_agent/config.py`**
2. **Update** relevant constant
3. **Test** by running agent
4. **Document** in SETUP.md if user-facing

Ask Claude: "Review my config changes and suggest improvements"

#### Performance Optimization

1. **Identify bottleneck** (use `make profile` or observe output)
2. **Research solution** (ask Claude for optimization patterns)
3. **Implement** and test
4. **Document** in DEVELOPER.md

Ask Claude: "How can I optimize [component] performance?"

### Testing with Claude Code

**Quick test run:**
```bash
make test
```

**Expected output:**
```
test_reranker.py: 6/6 passing ✓
test_hybrid_search.py: tests passing ✓
test_query_evaluator.py: tests passing ✓
```

**For specific testing:**
- Ask Claude: "Write a test for [feature]"
- Claude will generate test code for your specific use case

### File Organization

Key files for Claude Code integration:

| File | Purpose | Edit When |
|------|---------|-----------|
| `langchain_agent/main.py` | Core agent logic | Adding features, fixing bugs |
| `langchain_agent/config.py` | Configuration | Adjusting performance, changing models |
| `langchain_agent/setup.py` | Initialization | Changing setup process |
| `langchain_agent/DEVELOPER.md` | Architecture | Documenting new components |
| `langchain_agent/README.md` | User guide | Adding user features |
| `langchain_agent/SETUP.md` | Setup guide | Changing setup instructions |

## Claude Code Features for This Project

### Useful Patterns

**Investigating Issues:**
```
Claude, I'm seeing [symptom]. Where in [component] would this happen?
```

**Understanding Architecture:**
```
Claude, explain the data flow from [input] to [output]
```

**Optimization:**
```
Claude, what's the performance bottleneck in [function]?
```

**Testing:**
```
Claude, write a test for the [feature] that covers edge cases
```

### MCP Servers for This Project

Recommended MCP integrations:

1. **Web Search MCP** - Research cross-encoder alternatives, latest LangChain patterns
2. **Bash MCP** - Run tests, deploy changes
3. **File Editor MCP** - Edit multiple files atomically

### Pre-commit Validation

Consider adding a hook to validate reranker functionality:

```bash
# .git/hooks/pre-commit
python langchain_agent/test_reranker.py || exit 1
```

Ask Claude: "Help me set up a pre-commit hook for testing"

## Workflow Recommendations

### For Bug Fixes
1. Ask Claude to help diagnose (provide error message + relevant code)
2. Implement fix together
3. Run tests to validate
4. Commit with descriptive message

### For New Features
1. Discuss architecture with Claude
2. Plan implementation (see DEVELOPER.md for patterns)
3. Implement incrementally
4. Test each component
5. Update documentation
6. Commit with feature description

### For Performance Work
1. Profile to identify bottleneck (ask Claude how)
2. Research solutions (Claude can search web)
3. Implement with validation
4. Benchmark before/after
5. Document optimization in DEVELOPER.md

## Documentation

### Updating Docs

Edit the appropriate guide:
- **User questions?** → `langchain_agent/README.md`
- **Setup issues?** → `langchain_agent/SETUP.md`
- **Architecture question?** → `langchain_agent/DEVELOPER.md`
- **Project overview?** → Root `README.md`

Ask Claude: "Update the docs to reflect [change]"

### Code Comments

- Use comments for **why**, not **what**
- Ask Claude: "Add comments explaining [complex_logic]"
- Focus on non-obvious decisions

## Common Questions

**Q: How do I test my changes?**
```bash
make test  # Run all tests
python test_reranker.py  # Run specific test
```

**Q: How do I see what the agent is doing?**
```bash
python main.py  # Run with full logging
# Watch for [Reranker], [Query Evaluator], [Hybrid] output
```

**Q: How do I add a custom document?**
1. Add `.txt` file to `sample_docs/`
2. Re-run setup: `python setup.py`
3. Queries will now search custom documents

**Q: How do I change the LLM model?**
Edit `langchain_agent/config.py`:
```python
LLM_MODEL = "mistral:7b"  # or any Ollama model
```
Then restart agent.

**Q: How do I improve answer quality?**
1. Check reranker output (should show scores > 0.7)
2. Add more relevant documents to `sample_docs/`
3. Adjust `RETRIEVER_LAMBDA_MULT` for different search blend
4. Try different reranker model in config

## Support

For help with Claude Code integration:
- `/help` - Claude Code built-in help
- Ask questions: "How do I..." or "Can Claude Code..."
- Report issues: https://github.com/anthropics/claude-code/issues

For project-specific help:
- See `langchain_agent/DEVELOPER.md` - architecture reference
- Run `make help` - development tasks
- Review test files for patterns

## Next Steps

1. **Try the agent:** `python main.py` and test a query
2. **Review architecture:** Read `langchain_agent/DEVELOPER.md`
3. **Run tests:** `make test` to verify setup
4. **Explore code:** Ask Claude to explain any component
5. **Start contributing:** Pick an item from DEVELOPER.md and improve it

Happy coding! 🚀
