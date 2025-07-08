# 🧠 EEG Emotion Classification System - SEED-IV Dataset

A comprehensive, production-ready system for emotion recognition from EEG signals using advanced deep learning and intelligent feature selection.

## 🎯 Objective

Build a lightweight yet high-accuracy EEG-based emotion classification model that can recognize four emotional states:
- **0** = Neutral 😐
- **1** = Sad 😢  
- **2** = Fear 😨
- **3** = Happy 😊

Perfect for mental health monitoring, assistive devices, and real-world patient applications.

## 📊 System Architecture

```
Raw EEG Data (.csv) → Feature Selection → Advanced CNN-LSTM Model → Emotion Prediction
     ↓                     ↓                      ↓                      ↓
 Auto-labeling         SelectKBest            PyTorch Model        Clean Dataset
```

## 🔍 How Feature Selection Works

### 📈 SelectKBest with F-Classification
```python
# Uses ANOVA F-test to score features
# Higher F-score = better feature for emotion discrimination
selector = SelectKBest(score_func=f_classif, k=150)
```

**What it does:**
- Calculates F-statistic for each EEG feature (Ch1_alpha, Ch2_beta, etc.)
- Measures how well each feature separates emotion classes
- Keeps only the top K most discriminative features
- **Example:** Ch25_alpha might score 45.2 while Ch17_delta scores 2.1

### 🧠 Alternative: Mutual Information
```python
# Measures information shared between feature and emotion label
selector = SelectKBest(score_func=mutual_info_classif, k=150)
```

**Benefits:**
- Captures non-linear relationships
- Better for complex EEG patterns
- More robust to noise

## 🚀 Quick Start

### 1. Run the Complete System
```bash
cd d:\eeg-python-code\eeg-seed-IV
uv run python eeg_emotion_classifier.py
```

### 2. What You'll Get
- ✅ **Automatic data loading** from your `csv/` directory
- ✅ **Intelligent feature selection** (310 → 150 best features)
- ✅ **Clean dataset** saved as `clean_eeg_dataset.csv`
- ✅ **Advanced CNN-LSTM model** trained on selected features
- ✅ **Performance metrics** and visualizations

## 📁 Data Format Requirements

Your CSV files should contain EEG features in this format:
```
Ch1_delta, Ch1_theta, Ch1_alpha, Ch1_beta, Ch1_gamma,
Ch2_delta, Ch2_theta, Ch2_alpha, Ch2_beta, Ch2_gamma,
...
Ch62_delta, Ch62_theta, Ch62_alpha, Ch62_beta, Ch62_gamma
```

**Directory Structure:**
```
csv/
├── subject1/
│   ├── session1/
│   │   ├── de_movingAve1.csv
│   │   ├── de_movingAve2.csv
│   │   └── ...
│   ├── session2/
│   └── session3/
└── subject2/
    └── ...
```

## 🧬 Advanced Model Architecture

### CNN-LSTM Hybrid with Attention
```python
class AdvancedEEGModel(nn.Module):
    def __init__(self):
        # 1D Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(256, 128, num_layers=2)
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(128, num_heads=8)
        
        # Classification layers
        self.classifier = nn.Linear(128, 4)  # 4 emotion classes
```

**Why This Architecture?**
- **CNN layers**: Extract spatial patterns from EEG channels
- **LSTM layers**: Capture temporal dependencies in brain signals  
- **Attention**: Focus on most important time periods
- **Dropout & BatchNorm**: Prevent overfitting

## 📈 Feature Selection Results

The system will show you the most important brain regions and frequency bands:

```
🏆 Top 10 most important features:
   1. Ch25_alpha        (Score:   45.23) - Ch25 alpha band
   2. Ch31_beta         (Score:   42.18) - Ch31 beta band
   3. Ch44_gamma        (Score:   38.95) - Ch44 gamma band
   4. Ch12_theta        (Score:   35.67) - Ch12 theta band
   5. Ch55_alpha        (Score:   33.42) - Ch55 alpha band
   ...
```

## 💾 Clean Dataset Output

After processing, you'll get `clean_eeg_dataset.csv` with:
- Only the most informative features (e.g., 150 out of 310)
- Proper emotion labels for each sample
- Metadata (subject, session, trial info)
- Ready for any ML framework

**Sample Clean Dataset:**
```csv
Ch25_alpha,Ch31_beta,Ch44_gamma,...,subject,session,emotion
0.245,0.156,0.389,...,1,1,2
0.123,0.278,0.445,...,1,1,2
0.334,0.198,0.267,...,1,2,0
...
```

## 🎯 Using the Trained Model

```python
# Initialize and train
classifier = EEGEmotionClassifier(data_dir="csv")
clean_data, features = classifier.prepare_data(k_features=150)
model = classifier.train_model(clean_data, features)

# Make predictions
eeg_sample = [0.245, 0.156, 0.389, ...]  # 150 features
result = classifier.predict_emotion(eeg_sample)

print(result)
# Output:
# {
#   'predicted_emotion': 'Happy',
#   'emotion_code': 3,
#   'confidence': 0.87,
#   'emotion_probabilities': {
#     'Neutral': 0.05, 'Sad': 0.03, 'Fear': 0.05, 'Happy': 0.87
#   }
# }
```

## 🔧 Customization Options

### Feature Selection Methods
```python
# Method 1: F-Classification (default)
classifier.prepare_data(feature_selection_method='f_classif', k_features=150)

# Method 2: Mutual Information
classifier.prepare_data(feature_selection_method='mutual_info_classif', k_features=100)
```

### Model Hyperparameters
```python
classifier.train_model(
    epochs=100,           # Training iterations
    batch_size=64,        # Samples per batch
    test_size=0.2,        # 20% for testing
)
```

## 📊 Performance Metrics

The system provides comprehensive evaluation:
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-emotion performance
- **Confusion Matrix**: Visual error analysis
- **Training Curves**: Loss and accuracy over time

## 🏥 Real-World Applications

- **Mental Health Monitoring**: Continuous emotion tracking
- **Brain-Computer Interfaces**: Emotion-aware assistive devices
- **Clinical Research**: Objective emotion assessment
- **Neurofeedback**: Real-time emotion training

## 🚀 Performance Tips

1. **More data = better performance**: Combine multiple subjects
2. **Feature selection sweet spot**: 100-200 features usually optimal
3. **Cross-validation**: Test on different subjects for generalization
4. **Ensemble methods**: Combine multiple models for robustness

## 🔬 Technical Details

- **Framework**: PyTorch for deep learning, scikit-learn for feature selection
- **Input**: EEG differential entropy features (5 frequency bands × 62 channels)
- **Output**: 4-class emotion probabilities
- **Training**: GPU-accelerated (CUDA if available)
- **Deployment**: CPU-friendly for real-time inference

---

## 🤝 Contributing

This system is designed for research and clinical applications. Feel free to:
- Add new feature selection methods
- Experiment with different architectures
- Optimize for specific hardware constraints
- Extend to more emotion classes

**Ready to classify emotions from brain signals! 🧠✨**
