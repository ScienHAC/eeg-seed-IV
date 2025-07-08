# SEED-IV EEG Emotion Recognition Dataset Explorer

## 🎯 Project Overview

This project presents an interactive web-based exploration of the SEED-IV dataset, which contains EEG (electroencephalography) brain signals recorded during emotional experiences. The goal is to develop machine learning models that can automatically recognize human emotions from brain activity patterns.

## 📊 Dataset Information

### SEED-IV Dataset Structure
- **Participants**: 15 subjects (6 males, 9 females)
- **Sessions**: 3 sessions per participant (on different days)
- **Trials**: 24 trials per session (72 unique film clips total)
- **Emotions**: 4 categories - Neutral (0), Sad (1), Fear (2), Happy (3)
- **EEG Channels**: 62 electrodes covering the entire scalp
- **Sampling Rate**: 1000 Hz (downsampled to 200 Hz for processing)

### Data Structure (3D)
The processed data has a 3-dimensional structure:
- **Dimension 1**: 62 EEG channels (brain electrode positions)
- **Dimension 2**: 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **Dimension 3**: ~42 time windows (4-second segments)

**Total Features**: 62 × 5 × ~42 = 13,020+ features per trial

## 🔬 Research Methodology

### Data Processing Pipeline
1. **Raw EEG Acquisition**: 62-channel signals recorded at 1000Hz
2. **Preprocessing**: 
   - Bandpass filtering (1-75 Hz)
   - Downsampling to 200Hz
   - Artifact removal
3. **Feature Extraction**: 
   - Differential Entropy (DE) computation
   - Analysis across 5 frequency bands
   - 4-second sliding windows
4. **Machine Learning**: Train classifiers for 4-emotion recognition

### Frequency Bands
- **Delta (1-4 Hz)**: Deep sleep, unconscious processes
- **Theta (4-8 Hz)**: Drowsiness, memory processing, emotional arousal
- **Alpha (8-14 Hz)**: Relaxation, reduced attention
- **Beta (14-31 Hz)**: Active thinking, alertness, problem-solving
- **Gamma (31-50 Hz)**: Higher-level cognitive functions, attention

## 📁 File Structure

```
eeg-seed-IV/
│
├── index.html              # Main interactive website
├── simple_output1.csv      # Sample processed EEG data (Subject 1, Session 1, Trial 1)
├── matTocsv.py            # Python script for MATLAB to CSV conversion
├── main.py                # Main data processing script
├── README.md              # This documentation
├── pyproject.toml         # Python project configuration
└── uv.lock               # Dependency lock file
```

## 🌐 Interactive Website Features

### 1. Dataset Overview
- Participant demographics visualization
- Experimental design explanation
- Data collection methodology

### 2. Data Structure Analysis
- 3D data structure visualization
- Processing pipeline explanation
- Feature extraction details

### 3. Interactive Data Explorer
- Real-time data visualization
- Channel-specific analysis
- Statistical summaries
- Emotion label mapping

### 4. Educational Content
- EEG frequency band explanations
- Research context and applications
- Technical implementation details

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Python 3.8+ (for data processing scripts)
- Required Python packages: `scipy`, `numpy`, `pandas`

### Running the Website

**⚠️ Important**: Due to browser security restrictions (CORS policy), you cannot simply open `index.html` directly. You need to run a local web server.

#### Method 1: Using the Provided Batch File (Windows)
1. Clone or download this repository
2. Double-click `start_server.bat` in the project folder
3. Open your browser and go to `http://localhost:8000`
4. Navigate through different sections using the top menu

#### Method 2: Manual Server Setup
1. Open Command Prompt/Terminal in the project directory
2. Run one of these commands:
   ```bash
   # Using Python (recommended)
   python -m http.server 8000
   
   # Using Node.js (if installed)
   npx http-server
   
   # Using PHP (if installed)
   php -S localhost:8000
   ```
3. Open your browser and go to `http://localhost:8000`
4. Use the Interactive Data Explorer to examine real EEG data

#### Troubleshooting
- If you get "python is not recognized", install Python from python.org
- If port 8000 is busy, try a different port: `python -m http.server 8080`
- Make sure `simple_output1.csv` is in the same folder as `index.html`

### Processing Additional Data

#### Converting MATLAB Files to CSV

The project includes two conversion methods:

**Method 1: Extract Specific Feature (Recommended)**
```python
# Extract a specific feature with auto-generated filename
from matTocsv import extract_specific_feature

# Example: Extract 'de_LDS1' from subject 1, session 1 file
mat_file = r"C:\path\to\data\1\1_20160518.mat"
csv_file = extract_specific_feature(mat_file, "de_LDS1")
# Creates: 1_1_de_LDS1.csv

# Other available features: 'psd_LDS1', 'de_movingAve1', 'psd_movingAve1', etc.
```

**Method 2: Convert All Features**
```python
# Convert the first available feature in the file
from matTocsv import simple_mat_to_csv

simple_mat_to_csv('input_file.mat', 'output_file.csv')
```

#### Setup Instructions
1. Install Python dependencies:
   ```bash
   pip install scipy numpy pandas
   ```
2. Place MATLAB (.mat) files in your desired directory
3. Use the conversion functions in Python:
   ```python
   # Interactive Python session
   python
   >>> from matTocsv import extract_specific_feature
   >>> extract_specific_feature('path/to/file.mat', 'de_LDS1')
   ```
4. Update the `csvFileMap` in `index.html` to include new files

#### Auto-Generated Filenames
The `extract_specific_feature` function automatically creates meaningful filenames:
- Input: `C:\data\subjects\3\3_20160518.mat` with feature `de_LDS1`
- Output: `3_3_de_LDS1.csv`
- Format: `{subject}_{session}_{feature}.csv`

## 📈 Current Status & Next Steps

### ✅ Completed
- [x] Dataset download and organization
- [x] Data structure analysis and understanding
- [x] Feature extraction pipeline development
- [x] CSV export functionality for processed features
- [x] Interactive visualization website
- [x] Educational documentation

### 🔄 In Progress / Next Steps
- [ ] Machine learning model development
- [ ] Cross-validation and performance evaluation
- [ ] Feature importance analysis
- [ ] Real-time emotion recognition system
- [ ] Academic paper preparation

## 🎓 Educational Use

This project is designed to be educational and informative for:
- **Students**: Learning about EEG signal processing and emotion recognition
- **Researchers**: Understanding the SEED-IV dataset structure and processing methods
- **Teachers**: Demonstrating real-world applications of machine learning in neuroscience
- **General Public**: Exploring the intersection of brain science and artificial intelligence

## 📊 Sample Data

The included `simple_output1.csv` file contains processed EEG features for:
- Subject 1, Session 1, Trial 1
- Emotion: Sad (label = 1)
- 310 features (62 channels × 5 frequencies)
- 44 time windows (176 seconds of data)

## 🔬 Technical Details

### Feature Extraction: Differential Entropy (DE)
Differential Entropy measures the complexity or randomness of EEG signals:
- Higher DE values indicate more complex, information-rich brain activity
- Computed for each frequency band separately
- Smoothed using Moving Average or Linear Dynamic System (LDS) methods

### Data Format
CSV files contain features organized as:
- **Columns**: Ch1_Freq1, Ch1_Freq2, ..., Ch62_Freq5 (310 total)
- **Rows**: Time windows (typically 40-50 per trial)
- **Values**: Differential Entropy features (floating-point numbers)

## 🤝 Applications & Impact

### Potential Applications
- **Brain-Computer Interfaces**: Direct control of devices using thoughts
- **Mental Health Monitoring**: Early detection of emotional disorders
- **Human-Computer Interaction**: Emotion-aware computing systems
- **Neurofeedback Systems**: Real-time brain training applications

### Research Impact
This work contributes to the growing field of affective computing and provides insights into:
- Neural correlates of human emotions
- Signal processing techniques for brain data
- Machine learning applications in neuroscience
- Educational tools for STEM learning

## 📚 References & Dataset

- **SEED-IV Dataset**: Available from BCMI Laboratory, Shanghai Jiao Tong University
- **Original Paper**: Various publications on EEG-based emotion recognition
- **Processing Methods**: Based on established signal processing and machine learning techniques

## 📞 Contact & Support

For questions about this project or the SEED-IV dataset analysis, please refer to the interactive website's documentation and educational materials.

---

*This project demonstrates the intersection of neuroscience, signal processing, and machine learning in understanding human emotions through brain activity analysis.*
