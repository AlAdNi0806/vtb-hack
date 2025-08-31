#!/bin/bash

# Startup script for the Real-time Speech-to-Text application
# This script starts both the backend and frontend servers

echo "🚀 Starting Real-time Speech-to-Text Application"
echo "================================================"

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    echo "Killing process on port $port..."
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
}

# Check if ports are already in use
if check_port 8000; then
    echo "⚠️  Port 8000 is already in use. Killing existing process..."
    kill_port 8000
    sleep 2
fi

if check_port 3000; then
    echo "⚠️  Port 3000 is already in use. Killing existing process..."
    kill_port 3000
    sleep 2
fi

# Start backend server
echo "🔧 Starting backend server..."
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   cd backend && ./setup_ubuntu.sh"
    exit 1
fi

# Activate virtual environment and start backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! check_port 8000; then
    echo "❌ Backend failed to start on port 8000"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Backend server started on http://localhost:8000"

# Start frontend server
echo "🎨 Starting frontend server..."
cd ../frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "❌ Node modules not found. Please run:"
    echo "   cd frontend && npm install"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

# Check if frontend started successfully
if ! check_port 3000; then
    echo "❌ Frontend failed to start on port 3000"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Frontend server started on http://localhost:3000"
echo ""
echo "🎉 Application is ready!"
echo "📱 Open your browser and go to: http://localhost:3000"
echo ""
echo "To stop the application, press Ctrl+C"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping application..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "✅ Application stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop the application
wait
