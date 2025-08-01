@echo off
REM EEG Emotion Recognition Research Website - Quick Start Script (Windows)
REM This script sets up and launches the interactive research website

echo 🧠 EEG Emotion Recognition Research Website
echo ==========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 18.0 or higher.
    echo    Download from: https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js version:
node --version

REM Navigate to the script directory
cd /d "%~dp0"

REM Check if package.json exists
if not exist "package.json" (
    echo ❌ package.json not found. Make sure you're in the correct directory.
    pause
    exit /b 1
)

echo 📁 Current directory: %cd%

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    
    REM Check for package managers
    pnpm --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo    Using pnpm...
        pnpm install
    ) else (
        yarn --version >nul 2>&1
        if %errorlevel% equ 0 (
            echo    Using yarn...
            yarn install
        ) else (
            echo    Using npm...
            npm install
        )
    )
    
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies.
        pause
        exit /b 1
    )
    
    echo ✅ Dependencies installed successfully!
) else (
    echo ✅ Dependencies already installed.
)

REM Create data directory if it doesn't exist
if not exist "public\data" mkdir "public\data"

REM Check if data files exist, create if missing
echo 📊 Checking data files...

if not exist "public\data\dataset.json" (
    echo ⏳ Creating dataset.json...
    echo {^
  "name": "SEED-IV",^
  "subjects": 15,^
  "sessions": 3,^
  "trials": 24,^
  "total_samples": 1080,^
  "emotions": ["Happy", "Sad", "Fear", "Neutral"],^
  "channels": 62,^
  "sampling_rate": 200,^
  "duration_per_trial": 60^
} > "public\data\dataset.json"
)

if not exist "public\data\features.json" (
    echo ⏳ Creating features.json...
    echo {^
  "total_features": 868,^
  "selected_features": 15,^
  "categories": {^
    "spectral": 310,^
    "statistical": 186,^
    "connectivity": 248,^
    "complexity": 124^
  }^
} > "public\data\features.json"
)

if not exist "public\data\results.json" (
    echo ⏳ Creating results.json...
    echo {^
  "stage1_accuracy": 77.64,^
  "stage2_accuracy": 97.7,^
  "best_model": "Random Forest + Sequential Forward Selection",^
  "total_stages": 6,^
  "completed_stages": 2^
} > "public\data\results.json"
)

echo ✅ Data files ready!

REM Build the project
echo 🔨 Building the project...
pnpm --version >nul 2>&1
if %errorlevel% equ 0 (
    pnpm run build
) else (
    yarn --version >nul 2>&1
    if %errorlevel% equ 0 (
        yarn build
    ) else (
        npm run build
    )
)

if %errorlevel% neq 0 (
    echo ❌ Build failed. Please check the errors above.
    pause
    exit /b 1
)

echo ✅ Build completed successfully!

REM Start the development server
echo.
echo 🚀 Starting the research website...
echo    Opening http://localhost:3000 in your browser...
echo.
echo    Press Ctrl+C to stop the server
echo.

REM Try to open browser
start http://localhost:3000

REM Start the server
pnpm --version >nul 2>&1
if %errorlevel% equ 0 (
    pnpm run dev
) else (
    yarn --version >nul 2>&1
    if %errorlevel% equ 0 (
        yarn dev
    ) else (
        npm run dev
    )
)
