#!/bin/bash

# EEG Emotion Recognition Full System Startup Script

echo "🧠 EEG Emotion Recognition Full System"
echo "======================================="
echo "Starting both Backend (FastAPI) and Frontend (Next.js)"
echo

# Check if we're in the correct directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: backend/main.py not found"
    echo "Please run this script from the web directory"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: frontend/package.json not found"  
    echo "Please run this script from the web directory"
    exit 1
fi

echo "✅ Directory structure verified"
echo

# Function to start backend
start_backend() {
    echo "🚀 Starting Backend (FastAPI)..."
    cd backend
    echo "🧠 EEG Emotion Recognition Backend"
    echo "======================================="
    python main.py
}

# Function to start frontend  
start_frontend() {
    echo "🚀 Starting Frontend (Next.js)..."
    cd frontend
    echo "🧠 EEG Emotion Recognition Frontend"
    echo "======================================="
    pnpm dev
}

# Check if we can run both in background
if command -v gnome-terminal &> /dev/null; then
    # Use gnome-terminal if available
    echo "🚀 Starting Backend in new terminal..."
    gnome-terminal --title="EEG Backend (FastAPI)" -- bash -c "cd backend && echo '🧠 EEG Emotion Recognition Backend' && echo '=======================================' && python main.py; exec bash"
    
    sleep 3
    
    echo "🚀 Starting Frontend in new terminal..."
    gnome-terminal --title="EEG Frontend (Next.js)" -- bash -c "cd frontend && echo '🧠 EEG Emotion Recognition Frontend' && echo '=======================================' && pnpm dev; exec bash"
    
elif command -v xterm &> /dev/null; then
    # Use xterm if available
    echo "🚀 Starting Backend in new terminal..."
    xterm -title "EEG Backend (FastAPI)" -e "cd backend && echo '🧠 EEG Emotion Recognition Backend' && echo '=======================================' && python main.py; bash" &
    
    sleep 3
    
    echo "🚀 Starting Frontend in new terminal..."
    xterm -title "EEG Frontend (Next.js)" -e "cd frontend && echo '🧠 EEG Emotion Recognition Frontend' && echo '=======================================' && pnpm dev; bash" &
    
else
    # Fallback: use background processes
    echo "🚀 Starting Backend (FastAPI) in background..."
    start_backend &
    BACKEND_PID=$!
    
    echo "⏳ Waiting for backend to initialize..."
    sleep 5
    
    echo "🚀 Starting Frontend (Next.js) in foreground..."
    start_frontend
    
    # Kill backend when frontend exits
    kill $BACKEND_PID 2>/dev/null
    exit 0
fi

echo
echo "✅ System startup initiated!"
echo
echo "📋 Services:"
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"  
echo "  Frontend:     http://localhost:3000"
echo
echo "💡 Tips:"
echo "  - Both services will open in separate terminals"
echo "  - Backend loads your actual .mat files"
echo "  - Frontend provides interactive dashboard"
echo "  - Press Ctrl+C in each terminal to stop services"
echo
echo "⚠️  If you see errors:"
echo "  1. Check that Python and Node.js are installed"
echo "  2. Run 'pip install -r backend/requirements.txt'"
echo "  3. Run 'pnpm install' in frontend directory"
echo "  4. Update data paths in backend/main.py"
echo
