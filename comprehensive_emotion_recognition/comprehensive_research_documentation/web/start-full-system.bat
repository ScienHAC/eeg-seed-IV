@echo off
REM EEG Emotion Recognition Full System Startup Script

echo 🧠 EEG Emotion Recognition Full System
echo =======================================
echo Starting both Backend (FastAPI) and Frontend (Next.js)
echo.

REM Set console title
title EEG Emotion Recognition System

REM Check if we're in the correct directory
if not exist "backend\main.py" (
    echo ❌ Error: backend\main.py not found
    echo Please run this script from the web directory
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo ❌ Error: frontend\package.json not found  
    echo Please run this script from the web directory
    pause
    exit /b 1
)

echo ✅ Directory structure verified
echo.

REM Start Backend in a new window
echo 🚀 Starting Backend (FastAPI) in new window...
start "EEG Backend (FastAPI)" cmd /k "cd backend && echo 🧠 EEG Emotion Recognition Backend && echo ======================================= && python main.py"

REM Wait a moment for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

REM Start Frontend in a new window  
echo 🚀 Starting Frontend (Next.js) in new window...
start "EEG Frontend (Next.js)" cmd /k "cd frontend && echo 🧠 EEG Emotion Recognition Frontend && echo ======================================= && pnpm dev"

echo.
echo ✅ System startup initiated!
echo.
echo 📋 Services:
echo   Backend API:  http://localhost:8000
echo   API Docs:     http://localhost:8000/docs  
echo   Frontend:     http://localhost:3000
echo.
echo 💡 Tips:
echo   - Both services will open in separate windows
echo   - Backend loads your actual .mat files
echo   - Frontend provides interactive dashboard
echo   - Press Ctrl+C in each window to stop services
echo.
echo ⚠️  If you see errors:
echo   1. Check that Python and Node.js are installed
echo   2. Run 'pip install -r backend/requirements.txt'
echo   3. Run 'pnpm install' in frontend directory
echo   4. Update data paths in backend/main.py
echo.
pause
