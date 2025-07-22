"""
Medical Grade EEG-CNN for SEED-IV Dataset
========================================

Transform your 50% SVM accuracy to 85%+ medical grade using:
1. Load your saved features from Sequential Feature Selection
2. Map features to proper EEG brain topology (62 channels)
3. Create 2D spatial brain maps for CNN
4. Train deep CNN with medical-grade accuracy targets
5. Use proper EEG channel positioning for spatial learning

Target: 85%+ accuracy (medical grade standard)
Input: Your saved .npy feature selections
Output: Medical grade emotion classifier

Author: AI Assistant  
Date: July 22, 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import os
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# EEG Channel positions for SEED-IV (62 channels) - Medical grade topology
EEG_CHANNEL_POSITIONS = {
    # Frontal region
    'FP1': (0, 2), 'FPZ': (0, 4), 'FP2': (0, 6),
    'AF3': (1, 1), 'AF4': (1, 7),
    'F7': (2, 0), 'F5': (2, 1), 'F3': (2, 2), 'F1': (2, 3), 'FZ': (2, 4), 
    'F2': (2, 5), 'F4': (2, 6), 'F6': (2, 7), 'F8': (2, 8),
    
    # Frontal-Central
    'FT7': (3, 0), 'FC5': (3, 1), 'FC3': (3, 2), 'FC1': (3, 3), 'FCZ': (3, 4),
    'FC2': (3, 5), 'FC4': (3, 6), 'FC6': (3, 7), 'FT8': (3, 8),
    
    # Central region  
    'T7': (4, 0), 'C5': (4, 1), 'C3': (4, 2), 'C1': (4, 3), 'CZ': (4, 4),
    'C2': (4, 5), 'C4': (4, 6), 'C6': (4, 7), 'T8': (4, 8),
    
    # Central-Parietal
    'TP7': (5, 0), 'CP5': (5, 1), 'CP3': (5, 2), 'CP1': (5, 3), 'CPZ': (5, 4),
    'CP2': (5, 5), 'CP4': (5, 6), 'CP6': (5, 7), 'TP8': (5, 8),
    
    # Parietal region
    'P7': (6, 0), 'P5': (6, 1), 'P3': (6, 2), 'P1': (6, 3), 'PZ': (6, 4),
    'P2': (6, 5), 'P4': (6, 6), 'P6': (6, 7), 'P8': (6, 8),
    
    # Parietal-Occipital
    'PO7': (7, 1), 'PO5': (7, 2), 'PO3': (7, 3), 'POZ': (7, 4), 
    'PO4': (7, 5), 'PO6': (7, 6), 'PO8': (7, 7),
    
    # Occipital region
    'CB1': (8, 2), 'O1': (8, 3), 'OZ': (8, 4), 'O2': (8, 5), 'CB2': (8, 6)
}

def load_saved_features_and_data():
    """Load your saved Sequential Feature Selection results"""
    print("🔄 Loading your saved Sequential Feature Selection results...")
    
    # Load your best model and features (modify path as needed)
    checkpoint_dir = "../../saved_models"
    best_n_features = 25  # From your results
    
    model_path = f"{checkpoint_dir}/rf_model_{best_n_features}_features.joblib"
    features_path = f"{checkpoint_dir}/selected_features_{best_n_features}.npy"
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        print(f"✅ Loading saved results for {best_n_features} features")
        saved_features = np.load(features_path)
        print(f"📊 Selected {np.sum(saved_features)} features out of 200")
        return saved_features
    else:
        print("❌ Saved features not found. Run Sequential Feature Selection first!")
        return None

def create_eeg_topology_map(features, grid_size=(9, 9)):
    """
    Convert selected 1D features to 2D EEG brain topology map
    Medical grade approach using proper channel positioning
    """
    print(f"🧠 Creating medical-grade EEG topology maps...")
    
    n_samples = features.shape[0]
    n_features = features.shape[1]
    
    # Create 2D brain maps
    brain_maps = np.zeros((n_samples, grid_size[0], grid_size[1], 5))  # 5 frequency bands
    
    # Calculate features per frequency band (assuming 62 channels × 5 bands = 310 original)
    features_per_band = n_features // 5
    
    for sample_idx in range(n_samples):
        for band in range(5):
            # Get features for this frequency band
            band_start = band * features_per_band
            band_end = (band + 1) * features_per_band
            band_features = features[sample_idx, band_start:band_end]
            
            # Map to 2D grid using EEG topology
            for feat_idx, feat_val in enumerate(band_features):
                if feat_idx < len(EEG_CHANNEL_POSITIONS):
                    # Get spatial position for this channel
                    channel_names = list(EEG_CHANNEL_POSITIONS.keys())
                    if feat_idx < len(channel_names):
                        pos = EEG_CHANNEL_POSITIONS[channel_names[feat_idx]]
                        if pos[0] < grid_size[0] and pos[1] < grid_size[1]:
                            brain_maps[sample_idx, pos[0], pos[1], band] = feat_val
    
    print(f"✅ Created brain topology maps: {brain_maps.shape}")
    return brain_maps

def create_medical_grade_cnn():
    """
    Create medical-grade CNN for EEG emotion recognition
    Designed for 85%+ accuracy (medical standard)
    """
    print("🏥 Building medical-grade CNN architecture...")
    
    model = models.Sequential([
        # Input layer - 2D EEG brain maps with 5 frequency bands
        layers.Input(shape=(9, 9, 5)),
        
        # First convolutional block - detect local brain patterns
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block - complex pattern detection
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block - high-level features
        layers.Conv2D(256, (2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Global pooling for spatial invariance
        layers.GlobalAveragePooling2D(),
        
        # Dense layers for classification
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer - 4 emotions
        layers.Dense(4, activation='softmax')
    ])
    
    # Compile with medical-grade optimization
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("✅ Medical-grade CNN architecture created")
    print(f"📊 Total parameters: {model.count_params():,}")
    return model

def train_medical_grade_model(X_maps, y, epochs=100):
    """
    Train CNN with medical-grade standards
    Target: 85%+ accuracy
    """
    print("🏥 Starting medical-grade CNN training...")
    print(f"📊 Target accuracy: 85%+ (medical standard)")
    
    # Convert labels to categorical
    y_cat = to_categorical(y, 4)
    
    # Create stratified train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_maps, y_cat, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"📊 Training data: {X_train.shape[0]} samples")
    print(f"📊 Validation data: {X_val.shape[0]} samples")
    
    # Create model
    model = create_medical_grade_cnn()
    
    # Medical-grade callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'medical_eeg_cnn_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history

def evaluate_medical_grade_performance(model, X_maps, y):
    """
    Comprehensive medical-grade evaluation
    """
    print("📊 Medical-grade performance evaluation...")
    
    # Convert labels
    y_cat = to_categorical(y, 4)
    
    # Stratified K-Fold cross-validation (medical standard)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    
    emotion_names = ['Neutral', 'Sad', 'Fear', 'Happy']
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_maps, y)):
        print(f"📋 Evaluating fold {fold + 1}/5...")
        
        X_train, X_test = X_maps[train_idx], X_maps[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]
        
        # Clone and train model for this fold
        fold_model = create_medical_grade_cnn()
        
        # Quick training for evaluation
        fold_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Evaluate
        y_pred = fold_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        fold_accuracy = accuracy_score(y_test_classes, y_pred_classes)
        cv_accuracies.append(fold_accuracy)
        
        print(f"   Fold {fold + 1} accuracy: {fold_accuracy:.3f}")
    
    mean_accuracy = np.mean(cv_accuracies)
    std_accuracy = np.std(cv_accuracies)
    
    print(f"\n🏆 MEDICAL-GRADE RESULTS:")
    print(f"   Cross-validation accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"   Medical standard (85%+): {'✅ ACHIEVED' if mean_accuracy >= 0.85 else '❌ NEEDS IMPROVEMENT'}")
    
    return mean_accuracy, std_accuracy

def plot_medical_results(history, accuracy):
    """Plot medical-grade training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training history
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.axhline(y=0.85, color='green', linestyle='--', label='Medical Standard (85%)')
    ax1.set_title('Medical-Grade CNN Training')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Loss curves
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss Curves')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # Accuracy comparison
    methods = ['SVM (Your Previous)', 'Medical CNN (Target)', 'Medical CNN (Achieved)']
    accuracies = [0.50, 0.85, accuracy]
    colors = ['red', 'green', 'blue']
    
    bars = ax3.bar(methods, accuracies, color=colors, alpha=0.7)
    ax3.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Medical Standard')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy Improvement')
    ax3.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Improvement metrics
    improvement = (accuracy - 0.50) / 0.50 * 100
    ax4.text(0.5, 0.7, f'Accuracy Improvement', ha='center', fontsize=16, fontweight='bold')
    ax4.text(0.5, 0.5, f'{accuracy:.1%}', ha='center', fontsize=24, fontweight='bold', color='blue')
    ax4.text(0.5, 0.3, f'+{improvement:.1f}% vs SVM', ha='center', fontsize=14, color='green')
    ax4.text(0.5, 0.1, 'Medical Grade' if accuracy >= 0.85 else 'Needs Improvement', 
             ha='center', fontsize=12, 
             color='green' if accuracy >= 0.85 else 'red')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('medical_cnn_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_medical_grade_pipeline():
    """
    Main pipeline for medical-grade EEG emotion recognition
    Transform your 50% SVM to 85%+ medical CNN
    """
    print("🏥 MEDICAL-GRADE EEG EMOTION RECOGNITION PIPELINE")
    print("=" * 60)
    print("🎯 Goal: Achieve 85%+ accuracy (medical standard)")
    
    # Step 1: Load your saved Sequential Feature Selection results
    saved_features = load_saved_features_and_data()
    if saved_features is None:
        return None
    
    # Step 2: Load the actual EEG data with selected features
    print("\n🔄 Loading EEG data with selected features...")
    import sys
    import importlib.util
    
    # Load the Sequential Feature Selection module
    sfs_path = Path("../sequential_feature_selection/clean_eeg_classifier.py").resolve()
    if sfs_path.exists():
        spec = importlib.util.spec_from_file_location("clean_eeg_classifier", sfs_path)
        clean_eeg_classifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clean_eeg_classifier)
        load_clean_seed_iv_data = clean_eeg_classifier.load_clean_seed_iv_data
        print(f"✅ Loaded Sequential Feature Selection module from: {sfs_path}")
    else:
        print(f"❌ Sequential Feature Selection module not found at: {sfs_path}")
        return None
    
    csv_dir = "../../csv"  # Path to CSV data
    X, y, metadata = load_clean_seed_iv_data(csv_dir, "de_LDS", max_subjects=15)
    
    if len(X) == 0:
        print("❌ Could not load EEG data")
        return None
    
    # Step 3: Apply your saved feature selection
    print("🎯 Applying your saved feature selection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply the same preprocessing as in your Sequential Feature Selection
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=min(200, X.shape[1]))
    X_filtered = selector.fit_transform(X_scaled, y)
    
    # Apply your saved feature selection
    X_selected = X_filtered[:, saved_features]
    print(f"✅ Using {X_selected.shape[1]} selected features")
    
    # Step 4: Create medical-grade EEG topology maps
    print("\n🧠 Creating medical-grade EEG topology maps...")
    X_maps = create_eeg_topology_map(X_selected)
    
    # Step 5: Train medical-grade CNN
    print(f"\n🏥 Training medical-grade CNN...")
    model, history = train_medical_grade_model(X_maps, y)
    
    # Step 6: Comprehensive evaluation
    print(f"\n📊 Medical-grade evaluation...")
    final_accuracy, accuracy_std = evaluate_medical_grade_performance(model, X_maps, y)
    
    # Step 7: Save results and plot
    print(f"\n💾 Saving medical-grade results...")
    plot_medical_results(history, final_accuracy)
    
    # Save final model
    model.save('medical_eeg_final_model.h5')
    
    # Save results summary
    results_summary = {
        'method': 'Medical-Grade CNN',
        'cv_accuracy': final_accuracy,
        'accuracy_std': accuracy_std,
        'medical_grade': final_accuracy >= 0.85,
        'improvement_over_svm': (final_accuracy - 0.50) / 0.50 * 100,
        'selected_features': X_selected.shape[1],
        'total_samples': len(X)
    }
    
    pd.DataFrame([results_summary]).to_csv('medical_eeg_cnn_results.csv', index=False)
    
    print(f"\n🏆 FINAL MEDICAL-GRADE RESULTS:")
    print(f"   Method: Medical-Grade CNN with EEG Topology")
    print(f"   Accuracy: {final_accuracy:.3f} ± {accuracy_std:.3f}")
    print(f"   Medical Standard: {'✅ ACHIEVED' if final_accuracy >= 0.85 else '❌ NEEDS IMPROVEMENT'}")
    print(f"   Improvement: +{(final_accuracy - 0.50) / 0.50 * 100:.1f}% vs SVM")
    print(f"   Files saved: medical_eeg_final_model.h5, medical_eeg_cnn_results.csv")
    
    return results_summary

if __name__ == "__main__":
    results = run_medical_grade_pipeline()
