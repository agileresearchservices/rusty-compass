#!/bin/bash
# Quick test runner for LangChain agent improvements
# Usage: ./run_tests.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=================================================="
echo "LangChain Agent Improvements Test Suite"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at .venv${NC}"
    echo "Please create it with: python3 -m venv .venv"
    echo "Then activate and install dependencies: source .venv/bin/activate && pip install -r requirements.txt requirements-dev.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}pytest not found. Installing test dependencies...${NC}"
    pip install -r requirements-dev.txt
fi

# Parse arguments
TEST_ARGS="$@"

if [ -z "$TEST_ARGS" ]; then
    # Default: run all tests with verbose output
    TEST_ARGS="-v"
fi

# Run tests
echo -e "${GREEN}Running tests...${NC}"
echo ""
pytest test_improvements.py $TEST_ARGS

echo ""
echo -e "${GREEN}Test run complete!${NC}"
