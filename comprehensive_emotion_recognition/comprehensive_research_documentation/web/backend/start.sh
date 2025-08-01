# EEG Emotion Recognition Backend Startup Script (uv-managed)

echo "🧠 EEG Emotion Recognition Backend Setup"
echo "========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install uv: https://docs.astral.sh/uv/getting-started/installation/"
    echo "   Or run: pip install uv"
    exit 1
fi

echo "✅ uv found: $(uv --version)"

# Sync dependencies with uv
echo ""
echo "📦 Syncing dependencies with uv..."
uv sync

if [ $? -eq 0 ]; then
    echo "✅ Dependencies synced successfully"
else
    echo "❌ Failed to sync dependencies with uv"
    exit 1
fi

echo ""
echo "🚀 Starting FastAPI server with uv..."
echo "Server will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server using uv run
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
