@echo off
REM Complete EEG Backend Test Script with uv

echo 🧠 EEG Backend Complete Test Suite
echo ===================================

REM Check if uv is available
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv not found. Please install uv first.
    echo    Visit: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

echo ✅ uv found: 
uv --version

REM Sync dependencies
echo.
echo 📦 Syncing dependencies...
uv sync
if %errorlevel% neq 0 (
    echo ❌ Failed to sync dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies synced successfully

REM Start server in background and test
echo.
echo 🚀 Starting server for testing...

REM Start server in background
start "EEG Backend Server" /MIN uv run uvicorn main:app --host 127.0.0.1 --port 8000

REM Wait for server to start
echo Waiting for server to start...
ping 127.0.0.1 -n 3 >nul

REM Run API tests
echo.
echo 🧪 Running API tests...
uv run python test_api.py

echo.
echo 🎉 Test complete! Check the minimized server window to stop the server.
echo    Or run: taskkill /F /IM python.exe /T
echo.

pause
