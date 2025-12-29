.PHONY: help setup test lint format clean run stop test-reranker test-hybrid test-query dev dev-api dev-web install-web

help:
	@echo "Rusty Compass - Development Commands"
	@echo ""
	@echo "Setup & Initialization:"
	@echo "  make setup              - Run complete system setup (database, models, data)"
	@echo "  make install-web        - Install web UI dependencies (npm install)"
	@echo ""
	@echo "Testing:"
	@echo "  make test               - Run all test suites"
	@echo "  make test-reranker      - Run reranker tests only"
	@echo "  make test-hybrid        - Run hybrid search tests only"
	@echo "  make test-query         - Run query evaluator tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint               - Run code quality checks (pylint)"
	@echo "  make format             - Format code with black"
	@echo "  make type-check         - Check types with mypy (if installed)"
	@echo ""
	@echo "Running:"
	@echo "  make run                - Start the CLI agent"
	@echo "  make dev                - Start full dev environment (API + Web UI)"
	@echo "  make dev-api            - Start FastAPI backend only (port 8000)"
	@echo "  make dev-web            - Start React frontend only (port 5173)"
	@echo "  make stop               - Stop all running services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - Remove generated files and caches"
	@echo "  make clean-db           - Reset PostgreSQL database (WARNING: removes all data)"
	@echo ""

setup:
	@echo "[1/1] Running complete setup..."
	python3 setup.py

test:
	@echo "[1/3] Running reranker tests..."
	python3 test_reranker.py || exit 1
	@echo ""
	@echo "[2/3] Running hybrid search tests..."
	python3 test_hybrid_search.py || exit 1
	@echo ""
	@echo "[3/3] Running query evaluator tests..."
	python3 test_query_evaluator.py || exit 1
	@echo ""
	@echo "✓ All tests passed!"

test-reranker:
	@echo "Running reranker tests..."
	python3 test_reranker.py

test-hybrid:
	@echo "Running hybrid search tests..."
	python3 test_hybrid_search.py

test-query:
	@echo "Running query evaluator tests..."
	python3 test_query_evaluator.py

lint:
	@echo "Running code quality checks..."
	@command -v pylint >/dev/null 2>&1 || { echo "pylint not installed. Install with: pip install pylint"; exit 1; }
	pylint main.py config.py setup.py --disable=missing-docstring || true

format:
	@echo "Formatting code with black..."
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Install with: pip install black"; exit 1; }
	black main.py config.py setup.py test_*.py

type-check:
	@echo "Checking types with mypy..."
	@command -v mypy >/dev/null 2>&1 || { echo "mypy not installed. Install with: pip install mypy"; exit 1; }
	mypy main.py config.py --ignore-missing-imports || true

run:
	@echo "Starting Rusty Compass agent..."
	python3 main.py

stop:
	@echo "Stopping all services..."
	@echo "  Stopping Python processes..."
	pkill -f "python.*main.py" || true
	pkill -f "uvicorn" || true
	@echo "  Stopping npm/Node processes..."
	pkill -f "npm.*dev" || true
	pkill -f "vite" || true
	@echo "✓ All services stopped"

clean:
	@echo "Cleaning up generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "✓ Cleanup complete"

clean-db:
	@echo "⚠ WARNING: This will delete ALL data in PostgreSQL database!"
	@echo "Press Ctrl+C to cancel, or continue..."
	@sleep 3
	python3 -c "from setup import drop_all_tables; drop_all_tables()" || echo "Database cleanup failed or database not initialized"
	@echo "✓ Database cleaned"

# Web UI development
install-web:
	@echo "Installing web UI dependencies..."
	cd web && npm install

dev-api:
	@echo "Starting FastAPI backend on http://localhost:8000..."
	@echo "API docs available at http://localhost:8000/docs"
	uvicorn api.main:app --reload --port 8000

dev-web:
	@echo "Starting React frontend on http://localhost:5173..."
	cd web && npm run dev

dev:
	@echo "Starting full development environment..."
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:5173"
	@echo ""
	@echo "Run in separate terminals:"
	@echo "  Terminal 1: make dev-api"
	@echo "  Terminal 2: make dev-web"
	@echo ""
	@echo "Or start both now (backend in background):"
	@( uvicorn api.main:app --reload --port 8000 & ) && cd web && npm run dev

.SILENT: help
