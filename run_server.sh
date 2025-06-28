#!/bin/bash
echo "=== Starting 2w12 Backend Server ==="

# Kill any existing server on port 8001
echo "Stopping any existing server on port 8001..."
sudo kill -9 $(sudo lsof -t -i:8001) 2>/dev/null || echo "No existing server found"

# Navigate to project directory
echo "Navigating to project directory..."
cd /opt/2w12-backend/

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Start the server in background
echo "Starting Python server in background..."
python main.py &

# Get the process ID
SERVER_PID=$!

echo "=== Server started successfully! ==="
echo "Server PID: $SERVER_PID"
echo "Server running at: http://0.0.0.0:8001"
echo "To stop the server, run: sudo kill $SERVER_PID"
