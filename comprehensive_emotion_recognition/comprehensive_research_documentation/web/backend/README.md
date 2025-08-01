# EEG Emotion Recognition Backend

FastAPI backend server for processing SEED-IV EEG dataset `.mat` files and providing real-time data for the emotion recognition dashboard.

## 🧠 Overview

This backend server provides APIs to:
- Load and process SEED-IV `.mat` files
- Extract EEG features across different frequency bands
- Provide emotion labels and trial information
- Support real-time filtering by subject, session, frequency band, and trial

## 🚀 Quick Start

### Prerequisites

- **uv** (modern Python package manager) - [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.8+ (automatically managed by uv)

### Installation & Setup

1. **Using the startup script (Recommended):**
   ```bash
   # Windows
   start.bat
   
   # Unix/Linux/macOS
   chmod +x start.sh
   ./start.sh
   ```

2. **Manual setup:**
   ```bash
   # Sync dependencies
   uv sync
   
   # Start the server
   uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The server will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Endpoints

### Load EEG Data
```http
POST /load-data
Content-Type: application/json

{
  "subject": 1,
  "session": 1,
  "frequency_band": "de_LDS",
  "trial": 1
}
```

### Get Model Results
```http
POST /get-results
Content-Type: application/json

{
  "subject": 1,
  "session": 1
}
```

### Health Check
```http
GET /health
```

## 🏗️ Architecture

### Core Components

- **SeedIVMatLoader**: Main class for loading and processing `.mat` files
- **FastAPI App**: RESTful API server with automatic documentation
- **Data Processing**: Replicates the exact logic from the research pipeline

### File Structure

```
backend/
├── main.py              # FastAPI application and endpoints
├── pyproject.toml       # uv project configuration
├── start.bat           # Windows startup script
├── start.sh            # Unix startup script
├── README.md           # This file
└── requirements.txt    # Legacy pip requirements (kept for reference)
```

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `DEBUG`: Enable debug mode (default: True)

### uv Configuration

The project uses `pyproject.toml` for dependency management:

```toml
[project]
name = "eeg-backend"
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "scipy>=1.11.4",      # For .mat file processing
    "numpy>=1.24.3",      # Numerical computations
    "pandas>=2.0.3",      # Data manipulation
    # ... more dependencies
]
```

## 📊 Data Processing

### SEED-IV Dataset Support

- **Subjects**: 15 participants
- **Sessions**: 3 sessions per subject
- **Trials**: 24 trials per session
- **Emotions**: 4 categories (Neutral, Sad, Fear, Happy)
- **Channels**: 62 EEG channels
- **Frequency Bands**: 
  - `de_LDS`: Differential Entropy features
  - `de_movingAve`: Moving average features

### Feature Extraction

The backend replicates the exact feature extraction logic from the research:

1. Load `.mat` files using `scipy.io.loadmat`
2. Extract frequency band data (de_LDS1-24, de_movingAve1-24)
3. Process 62-channel EEG data
4. Apply emotion labeling (1=Neutral, 0=Sad, -1=Fear, 1=Happy)
5. Return structured data for visualization

## 🧪 Testing

```bash
# Run tests (when available)
uv run pytest

# Test specific endpoint
curl -X POST "http://localhost:8000/load-data" \
     -H "Content-Type: application/json" \
     -d '{"subject": 1, "session": 1, "frequency_band": "de_LDS", "trial": 1}'
```

## 🔄 Development

### Adding Dependencies

```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update dependencies
uv sync
```

### Running in Development Mode

```bash
# With auto-reload
uv run uvicorn main:app --reload

# With custom host/port
uv run uvicorn main:app --host 127.0.0.1 --port 3001
```

## 🚨 Troubleshooting

### Common Issues

1. **uv not found**
   ```bash
   # Install uv
   pip install uv
   # or follow: https://docs.astral.sh/uv/getting-started/installation/
   ```

2. **Port already in use**
   ```bash
   # Use different port
   uv run uvicorn main:app --port 8001
   ```

3. **CORS issues with frontend**
   - The backend includes CORS middleware for `http://localhost:3000`
   - Modify in `main.py` if frontend runs on different port

### Logs and Debugging

- Server logs are displayed in the terminal
- Use `--log-level debug` for verbose logging
- Check `/health` endpoint for server status

## 🤝 Integration

This backend integrates with:
- **Frontend Dashboard**: React/Next.js application at `../frontend/`
- **Research Pipeline**: Uses identical data processing logic
- **SEED-IV Dataset**: Direct `.mat` file processing

## 📈 Performance

- **Startup time**: < 5 seconds
- **Data loading**: ~1-2 seconds per `.mat` file
- **Memory usage**: ~100-500MB depending on loaded data
- **Concurrent requests**: Supports multiple simultaneous API calls

## 🛡️ Security

- CORS configured for local development
- No authentication required (development setup)
- File access limited to dataset directory
- Input validation on all endpoints

## 📝 License

Part of the EEG Emotion Recognition research project.