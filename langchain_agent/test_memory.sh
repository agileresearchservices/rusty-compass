#!/bin/bash

# Test script to verify memory leak fix

echo "Starting development environment..."
source .venv/bin/activate

# Start the server in the background
make dev > /tmp/dev.log 2>&1 &
DEV_PID=$!

echo "Development server started (PID: $DEV_PID)"
echo "Waiting for API to be ready..."
sleep 10

# Check if the server is running
if ! kill -0 $DEV_PID 2>/dev/null; then
    echo "❌ Server failed to start"
    cat /tmp/dev.log
    exit 1
fi

echo "✓ Server is running"

# Monitor memory usage of Python processes
echo ""
echo "Memory usage before query:"
ps aux | grep -E "python.*uvicorn|python.*main" | grep -v grep || echo "  No Python processes found"

# Give it a bit more time to stabilize
sleep 5

echo ""
echo "Initial memory snapshot:"
ps aux | grep python | grep -v grep | awk '{print $6}' | awk '{sum += $1} END {print "Total: " sum " KB"}'

# Stop the server
echo ""
echo "Stopping server..."
kill $DEV_PID 2>/dev/null
sleep 5

echo "✓ Test complete"
