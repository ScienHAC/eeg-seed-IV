"""
Simple CNN EEG Classifier for Deployment
=======================================

Clean CNN implementation for deployment:
1. Load best feature type (determined by clean_eeg_classifier.py)
2. Create 2D EEG spatial maps (62 channels × 5 frequency bands)
3. Simple CNN architecture for real-time deployment
4. Focus on stability and deployment readiness

Usage: python simple_cnn_classifier.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")

def load_best_eeg_data(csv_dir="csv", feature_type="de_LDS", max_subjects=15):
    """Load EEG data in format suitable for CNN deployment"""
    print(f"📂 Loading {feature_type} features for CNN")
    
    csv_path = Path(csv_dir)
    
    # SEED-IV labels
    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }
    
    all_features = []
    all_labels = []
    
    for session in range(1, 4):
        for subject in range(1, min(max_subjects + 1, 16)):
            subject_path = csv_path / str(session) / str(subject)
            if not subject_path.exists():
                continue
                
            for trial in range(1, 25):
                emotion_label = session_labels[session][trial - 1]
                file_path = subject_path / f"{feature_type}{trial}.csv"
                
                if file_path.exists():
                    try:
                        trial_data = pd.read_csv(file_path).values
                        # Average across time for stability
                        trial_features = np.mean(trial_data, axis=0)
                        all_features.append(trial_features)
                        all_labels.append(emotion_label)
                    except:
                        continue
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Reshape to 2D for CNN: 62 channels × 5 frequency bands
    X_2d = X.reshape(X.shape[0], 62, 5, 1)  # Add channel dimension
    
    print(f"✅ Loaded {X_2d.shape[0]} samples, shape: {X_2d.shape}")
    return X_2d, y

def create_simple_cnn(input_shape=(62, 5, 1), num_classes=4):
    """Simple CNN for deployment - fast and stable"""
    if not TF_AVAILABLE:
        return None
        
    model = Sequential([
        # Spatial feature extraction
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Deep feature extraction  
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Classification
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_simple_cnn_classification():
    """Run simple CNN classification for deployment"""
    print("🚀 SIMPLE CNN EEG CLASSIFIER FOR DEPLOYMENT")
    print("=" * 50)
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow required. Install with: pip install tensorflow")
        return None
    
    # Load data
    csv_dir = "d:/eeg-python-code/eeg-seed-IV/csv"
    X, y = load_best_eeg_data(csv_dir, feature_type="de_LDS")
    
    if len(X) == 0:
        X, y = load_best_eeg_data("csv", feature_type="de_LDS")
    
    if len(X) == 0:
        print("❌ Could not load data")
        return None
    
    # Check balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"🎯 Dataset balance:")
    emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"   {emotions[label]}: {count} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    X_scaled = X_scaled.reshape(X.shape)
    
    # Convert labels
    y_categorical = to_categorical(y, num_classes=4)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"📊 Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    
    # Create and train model
    model = create_simple_cnn()
    print(f"🏗️ Model parameters: {model.count_params():,}")
    
    # Train with early stopping
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🏆 Test Accuracy: {test_accuracy:.3f}")
    
    # Predictions for detailed analysis
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=emotions))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target: 80%')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good: 70%')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save model for deployment
    model.save('eeg_emotion_cnn.h5')
    print(f"\n💾 Model saved as 'eeg_emotion_cnn.h5'")
    
    # Save preprocessing info
    preprocessing_info = {
        'feature_type': 'de_LDS',
        'input_shape': (62, 5, 1),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'emotions': emotions,
        'test_accuracy': float(test_accuracy)
    }
    
    pd.DataFrame([preprocessing_info]).to_json('cnn_preprocessing.json')
    print(f"💾 Preprocessing info saved as 'cnn_preprocessing.json'")
    
    return {
        'model': model,
        'test_accuracy': test_accuracy,
        'scaler': scaler,
        'emotions': emotions
    }

if __name__ == "__main__":
    results = run_simple_cnn_classification()
