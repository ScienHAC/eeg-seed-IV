#!/bin/bash

# EEG Emotion Recognition Research Website - Quick Start Script
# This script sets up and launches the interactive research website

echo "🧠 EEG Emotion Recognition Research Website"
echo "=========================================="
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18.0 or higher."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2)
REQUIRED_VERSION="18.0.0"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Node.js version $NODE_VERSION is too old. Please upgrade to 18.0.0 or higher."
    exit 1
fi

echo "✅ Node.js version: $NODE_VERSION"

# Navigate to the frontend directory
cd "$(dirname "$0")"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Make sure you're in the correct directory."
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    
    # Try different package managers
    if command -v pnpm &> /dev/null; then
        echo "   Using pnpm..."
        pnpm install
    elif command -v yarn &> /dev/null; then
        echo "   Using yarn..."
        yarn install
    else
        echo "   Using npm..."
        npm install
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        exit 1
    fi
    
    echo "✅ Dependencies installed successfully!"
else
    echo "✅ Dependencies already installed."
fi

# Create data directory if it doesn't exist
mkdir -p public/data

# Check if data files exist, create if missing
echo "📊 Checking data files..."

if [ ! -f "public/data/dataset.json" ]; then
    echo "⏳ Creating dataset.json..."
    cat > public/data/dataset.json << 'EOF'
{
  "name": "SEED-IV",
  "subjects": 15,
  "sessions": 3,
  "trials": 24,
  "total_samples": 1080,
  "emotions": ["Happy", "Sad", "Fear", "Neutral"],
  "channels": 62,
  "sampling_rate": 200,
  "duration_per_trial": 60
}
EOF
fi

if [ ! -f "public/data/features.json" ]; then
    echo "⏳ Creating features.json..."
    cat > public/data/features.json << 'EOF'
{
  "total_features": 868,
  "selected_features": 15,
  "categories": {
    "spectral": 310,
    "statistical": 186,
    "connectivity": 248,
    "complexity": 124
  }
}
EOF
fi

if [ ! -f "public/data/results.json" ]; then
    echo "⏳ Creating results.json..."
    cat > public/data/results.json << 'EOF'
{
  "stage1_accuracy": 77.64,
  "stage2_accuracy": 97.7,
  "best_model": "Random Forest + Sequential Forward Selection",
  "total_stages": 6,
  "completed_stages": 2
}
EOF
fi

echo "✅ Data files ready!"

# Build the project
echo "🔨 Building the project..."
if command -v pnpm &> /dev/null; then
    pnpm run build
elif command -v yarn &> /dev/null; then
    yarn build
else
    npm run build
fi

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi

echo "✅ Build completed successfully!"

# Start the development server
echo ""
echo "🚀 Starting the research website..."
echo "   Opening http://localhost:3000 in your browser..."
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Try to open browser (works on most systems)
if command -v open &> /dev/null; then
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
elif command -v start &> /dev/null; then
    start http://localhost:3000
fi

# Start the server
if command -v pnpm &> /dev/null; then
    pnpm run dev
elif command -v yarn &> /dev/null; then
    yarn dev
else
    npm run dev
fi
