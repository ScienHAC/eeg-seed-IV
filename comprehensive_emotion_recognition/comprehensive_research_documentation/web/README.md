# EEG Emotion Recognition Dashboard

An interactive web dashboard showcasing breakthrough EEG-based emotion recognition research achieving **97.7% accuracy** on the SEED-IV dataset.

## 🧠 System Overview

This dashboard provides a comprehensive interface to explore:
- **Real-time EEG data visualization** from .mat files
- **Interactive controls** for subject/session/trial/frequency band selection
- **Research results** showing progression from 77.64% to 97.7% accuracy
- **Model performance analysis** with confusion matrices and feature importance
- **Frequency band analysis** across Delta, Theta, Alpha, Beta, and Gamma bands

## 🏗️ Architecture

The system consists of two main components:

### Frontend (Next.js + React)
- **Location**: `./frontend/`
- **Technology**: Next.js 15, React 19, TypeScript, Tailwind CSS, Recharts
- **Purpose**: Interactive dashboard with charts, controls, and visualizations

### Backend (FastAPI + Python)
- **Location**: `./backend/`  
- **Technology**: FastAPI, Python, scipy, numpy, pandas
- **Package Manager**: uv (modern Python dependency management)
- **Purpose**: Process .mat files using the same logic as the main research pipeline

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ (for frontend)
- **uv** (modern Python package manager) - [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
- **pnpm** (recommended) or npm

### Method 1: Start Both Services Automatically

**Windows:**
```bash
# Start both frontend and backend
./start-full-system.bat
```

**Linux/Mac:**
```bash
# Start both frontend and backend
./start-full-system.sh
```

### Method 2: Start Services Individually

#### 1. Start the Backend (FastAPI with uv)

```bash
cd backend

# Quick start (recommended)
./start.bat          # Windows
./start.sh           # Linux/Mac

# Or manually:
uv sync              # Install/sync dependencies
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Development Options:**
```bash
# Using the development helper
uv run python dev.py server    # Start with dev helper
uv run python dev.py test      # Quick functionality test
uv run python dev.py deps      # Show dependency tree

# Legacy method (still supported)
pip install -r requirements.txt
python main.py
```

The backend will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

#### 2. Start the Frontend (Next.js)

```bash
cd frontend

# Install dependencies  
pnpm install

# Start the development server
pnpm dev
```

The frontend will be available at:
- **Dashboard**: http://localhost:3000

## 📊 Features

### Interactive Controls
- **Subject Selection**: Choose from 15 subjects (1-15)
- **Session Selection**: Choose from 3 sessions per subject
- **Trial Selection**: Choose from 24 trials per session
- **Frequency Band Selection**: All bands, Delta, Theta, Alpha, Beta, Gamma
- **Real-time Loading**: Load actual .mat file data with progress indicators

### Data Visualization
- **Time Series Charts**: Real-time EEG signal visualization
- **Frequency Analysis**: Power distribution across frequency bands
- **Confusion Matrix**: Model performance visualization
- **Feature Importance**: Top contributing features
- **Emotion Distribution**: Balanced dataset visualization

### Research Results
- **Stage 1**: SVM Baseline (77.64% accuracy)
- **Stage 2**: Enhanced Random Forest (97.7% accuracy)
- **Model Comparison**: Performance progression visualization
- **Clinical Impact**: Discussion of real-world applications

## 🔧 Configuration

### Backend Configuration

Edit `backend/main.py` to update data paths:

```python
# Update these paths to match your SEED-IV dataset location
SEED_IV_BASE_PATH = Path("path/to/your/csv/files")  
SEED_IV_MAT_PATH = Path("path/to/your/mat/files")
```

### Frontend Configuration

The frontend automatically connects to the backend at `http://localhost:8000`. To change this, update the API calls in `frontend/app/page.tsx`.

## 📁 Data Structure

The system expects data in the following structure:

### .mat Files (Original SEED-IV structure)
```
mat_files/
├── 1/          # Session 1
│   ├── 1_20160518.mat    # Subject 1
│   ├── 2_20150915.mat    # Subject 2
│   └── ...
├── 2/          # Session 2
│   └── ...
└── 3/          # Session 3
    └── ...
```

### CSV Files (Converted structure)
```
csv/
├── 1/          # Session 1
│   ├── 1/      # Subject 1
│   │   ├── de_LDS1.csv
│   │   ├── de_LDS2.csv
│   │   └── ...
│   └── 2/      # Subject 2
│       └── ...
└── ...
```

## 🎯 API Endpoints

The backend provides the following endpoints:

### Core Data Endpoints
- `GET /api/dataset-info` - Get dataset information
- `POST /api/load-eeg-data` - Load EEG data for specific parameters
- `GET /api/model-results` - Get research results (97.7% accuracy)
- `GET /api/frequency-analysis` - Get frequency band analysis

### Example API Usage

```javascript
// Load EEG data
const response = await fetch('http://localhost:8000/api/load-eeg-data', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    subject: 1,
    session: 1, 
    trial: 1,
    frequency_band: 'alpha'
  })
})

const data = await response.json()
```

## 🧪 Research Integration

This dashboard directly integrates with your main research pipeline:

### Data Processing Logic
- **Replicates** the same .mat file loading logic from `seed_iv_loader.py`
- **Uses** identical feature extraction methods (`de_LDS`, `de_movingAve`)
- **Maintains** the same emotion labeling system
- **Preserves** frequency band processing

### Model Results
- **Stage 1**: Traditional SVM baseline (77.64% accuracy)
- **Stage 2**: Enhanced Random Forest (97.7% accuracy)
- **Real confusion matrices** from your research
- **Actual feature importance** rankings
- **Genuine emotion distribution** (25.1% each, naturally balanced)

## 🔍 Troubleshooting

### Backend Issues

**"uv command not found"**
```bash
# Install uv first
pip install uv
# Or follow: https://docs.astral.sh/uv/getting-started/installation/
```

**"No module named fastapi"**
```bash
cd backend
uv sync              # Modern approach (recommended)
# OR
pip install -r requirements.txt  # Legacy approach
```

**"No data found for Subject X"**
- Check that `SEED_IV_BASE_PATH` and `SEED_IV_MAT_PATH` point to correct directories
- Verify .mat files exist in the expected structure
- Check the console logs for detailed error messages

### Frontend Issues

**"Module not found" errors**
```bash
cd frontend
pnpm install
```

**"Failed to fetch" errors**
- Ensure the backend is running on port 8000
- Check that CORS is properly configured
- Verify the API endpoints are accessible

### Data Issues

**"Mock data generated"**
- This indicates no real .mat files were found
- Update the paths in `backend/main.py`
- Check file permissions and directory structure

## 📈 Performance

### Backend Performance
- **Caching**: Loaded .mat files are cached in memory
- **Lazy Loading**: Files are loaded only when requested
- **Error Handling**: Graceful fallback to mock data

### Frontend Performance  
- **Real-time Updates**: Instant response to control changes
- **Optimized Rendering**: Large datasets handled efficiently
- **Progressive Loading**: Visual feedback during data loading

## 🔒 Security

- **CORS**: Properly configured for localhost development
- **Input Validation**: All API inputs are validated
- **Error Handling**: No sensitive information exposed in errors

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** both frontend and backend
5. **Submit** a pull request

## 📄 License

This project is part of the EEG emotion recognition research. Please reference the main research when using this code.

## 🙏 Acknowledgments

- **SEED-IV Dataset**: Shanghai Jiao Tong University
- **Research Achievement**: 97.7% accuracy milestone
- **Technology Stack**: FastAPI, Next.js, React, Python scientific libraries

---

**🧠 Achieving Clinical-Grade EEG Emotion Recognition with Interactive Research Tools**
