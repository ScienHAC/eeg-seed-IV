@echo off
REM EEG Emotion Recognition Backend Startup Script for Windows (uv-managed)

REM Change to the backend directory
cd /d "D:\eeg-python-code\eeg-seed-IV\comprehensive_emotion_recognition\comprehensive_research_documentation\web\backend"

echo 🧠 EEG Emotion Recognition Backend Setup
echo =========================================
echo 📁 Working directory: %CD%
echo.

REM Check if uv is available
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv not found. Please install uv: https://docs.astral.sh/uv/getting-started/installation/
    echo    Or run: pip install uv
    pause
    exit /b 1
)

echo ✅ uv found
uv --version

REM Try pip as fallback if uv has conflicts
echo.
echo 📦 Installing dependencies with pip (fallback)...
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 numpy pandas scipy python-multipart

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies with pip
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

echo.
echo 🚀 Starting FastAPI server with uv...
echo Server will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server using uv run
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
