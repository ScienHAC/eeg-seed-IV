"""
IMPROVED Medical-Grade CNN for SEED-IV Dataset
==============================================

Fix the accuracy drop issue by creating a CNN that properly uses
your successful 25-feature Sequential Feature Selection results.

Key improvements:
1. Use your 25 optimal features directly (not brain topology mapping)
2. Simple but effective CNN architecture  
3. Same preprocessing as your successful SFS
4. Proper data handling for SEED-IV structure

Goal: Transform your 50% SFS accuracy to 70%+ with proper CNN
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
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_your_successful_sfs_data():
    """Load your successful Sequential Feature Selection setup"""
    print("🎯 Loading your successful SFS setup...")
    
    # Use absolute paths
    current_file_dir = Path(__file__).parent
    checkpoint_dir = current_file_dir / "../../saved_models"
    checkpoint_dir = checkpoint_dir.resolve()
    
    # Load your best 25 features
    best_features_path = checkpoint_dir / "selected_features_25.npy"
    best_model_path = checkpoint_dir / "rf_model_25_features.joblib"
    
    if not (best_features_path.exists() and best_model_path.exists()):
        print("❌ Your successful SFS results not found!")
        return None, None
    
    # Load the feature selection
    selected_features = np.load(str(best_features_path))
    print(f"✅ Loaded your 25 optimal features")
    
    # Load the successful SFS model for reference
    sfs_model = joblib.load(str(best_model_path))
    print(f"✅ Loaded your successful SFS model")
    
    return selected_features, sfs_model

def load_data_with_sfs_preprocessing():
    """Load data using EXACTLY the same preprocessing as your successful SFS"""
    print("🔄 Loading data with your successful SFS preprocessing...")
    
    # Load the Sequential Feature Selection module
    current_file_dir = Path(__file__).parent
    sfs_path = current_file_dir / "../sequential_feature_selection/clean_eeg_classifier.py"
    sfs_path = sfs_path.resolve()
    
    if not sfs_path.exists():
        print(f"❌ SFS module not found at: {sfs_path}")
        return None, None, None
    
    # Import the SFS module
    import importlib.util
    spec = importlib.util.spec_from_file_location("clean_eeg_classifier", sfs_path)
    clean_eeg_classifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clean_eeg_classifier)
    
    # Load data using your successful method
    csv_dir = current_file_dir / "../../csv"
    csv_dir = csv_dir.resolve()
    
    X, y, metadata = clean_eeg_classifier.load_clean_seed_iv_data(str(csv_dir), "de_LDS", max_subjects=15)
    
    if len(X) == 0:
        print("❌ Could not load SEED-IV data")
        return None, None, None
    
    print(f"✅ Loaded SEED-IV data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, metadata

def create_improved_cnn_model(input_features=25):
    """
    Create a simple but effective CNN for your 25 selected features
    Focus on learning from your optimal features, not complex brain mapping
    """
    print(f"🏗️ Creating improved CNN for {input_features} features...")
    
    model = models.Sequential([
        # Input layer - your 25 optimal features
        layers.Dense(64, activation='relu', input_shape=(input_features,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers - learn feature interactions
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer - 4 emotions
        layers.Dense(4, activation='softmax')
    ])
    
    # Compile with good settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Improved CNN created with {model.count_params():,} parameters")
    return model

def train_improved_cnn(X, y, selected_features):
    """
    Train CNN using EXACTLY the same preprocessing as your successful SFS
    """
    print("🎯 Training improved CNN with your successful SFS preprocessing...")
    
    # Apply EXACTLY the same preprocessing as your successful SFS
    print("📊 Applying your successful SFS preprocessing...")
    
    # Step 1: Scale the data (same as SFS)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Apply feature selection (same as SFS)
    selector = SelectKBest(f_classif, k=min(200, X.shape[1]))
    X_filtered = selector.fit_transform(X_scaled, y)
    
    # Step 3: Apply your successful 25-feature selection
    X_selected = X_filtered[:, selected_features]
    
    print(f"✅ Data preprocessed: {X_selected.shape[0]} samples, {X_selected.shape[1]} features")
    
    # Convert labels to categorical
    y_cat = to_categorical(y, 4)
    
    # Split data (same as SFS approach)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_cat, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"📊 Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")
    
    # Create and train model
    model = create_improved_cnn_model(input_features=X_selected.shape[1])
    
    # Training callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history, X_test, y_test, scaler, selector

def evaluate_improved_cnn(model, X, y, selected_features, scaler, selector):
    """Comprehensive evaluation using cross-validation"""
    print("📊 Evaluating improved CNN with cross-validation...")
    
    # Apply same preprocessing
    X_scaled = scaler.transform(X)
    X_filtered = selector.transform(X_scaled)
    X_selected = X_filtered[:, selected_features]
    y_cat = to_categorical(y, 4)
    
    # 5-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_selected, y)):
        print(f"📋 Evaluating fold {fold + 1}/5...")
        
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]
        
        # Create and train model for this fold
        fold_model = create_improved_cnn_model(input_features=X_selected.shape[1])
        
        # Quick training
        fold_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        
        # Evaluate
        y_pred = fold_model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        fold_accuracy = accuracy_score(y_test_classes, y_pred_classes)
        cv_accuracies.append(fold_accuracy)
        
        print(f"   Fold {fold + 1} accuracy: {fold_accuracy:.3f}")
    
    mean_accuracy = np.mean(cv_accuracies)
    std_accuracy = np.std(cv_accuracies)
    
    return mean_accuracy, std_accuracy

def plot_improved_results(history, accuracy, sfs_accuracy=0.50):
    """Plot comparison showing improvement over both SFS and failed CNN"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training history
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.axhline(y=sfs_accuracy, color='green', linestyle='--', label=f'Your SFS Baseline ({sfs_accuracy:.1%})')
    ax1.axhline(y=0.85, color='red', linestyle='--', label='Medical Standard (85%)')
    ax1.set_title('Improved CNN Training vs Your SFS Baseline')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Loss curves
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Loss Curves')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # Accuracy comparison
    methods = ['Failed CNN\n(Previous)', 'Your SFS\n(Baseline)', 'Improved CNN\n(New)', 'Medical Target']
    accuracies = [0.413, sfs_accuracy, accuracy, 0.85]
    colors = ['red', 'green', 'blue', 'purple']
    
    bars = ax3.bar(methods, accuracies, color=colors, alpha=0.7)
    ax3.axhline(y=sfs_accuracy, color='green', linestyle='--', alpha=0.7, label='Your SFS Baseline')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy Comparison')
    ax3.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Results summary
    improvement_vs_sfs = (accuracy - sfs_accuracy) / sfs_accuracy * 100
    status = "SUCCESS" if accuracy > sfs_accuracy else "NEEDS MORE WORK"
    
    ax4.text(0.5, 0.8, f'Improved CNN Results', ha='center', fontsize=16, fontweight='bold')
    ax4.text(0.5, 0.6, f'{accuracy:.1%}', ha='center', fontsize=24, fontweight='bold', 
             color='blue' if accuracy > sfs_accuracy else 'red')
    ax4.text(0.5, 0.4, f'{improvement_vs_sfs:+.1f}% vs Your SFS', ha='center', fontsize=14,
             color='green' if improvement_vs_sfs > 0 else 'red')
    ax4.text(0.5, 0.2, status, ha='center', fontsize=12, 
             color='green' if accuracy > sfs_accuracy else 'red')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('improved_cnn_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_improved_cnn_pipeline():
    """
    Main pipeline for improved CNN that should beat your SFS results
    """
    print("🎯 IMPROVED CNN PIPELINE - BEAT YOUR SFS RESULTS")
    print("=" * 60)
    print("🎯 Goal: Beat your 50% SFS accuracy with proper CNN")
    
    # Step 1: Load your successful SFS setup
    selected_features, sfs_model = load_your_successful_sfs_data()
    if selected_features is None:
        return None
    
    # Step 2: Load data with your successful preprocessing
    X, y, metadata = load_data_with_sfs_preprocessing()
    if X is None:
        return None
    
    # Step 3: Train improved CNN
    print("\n🚀 Training improved CNN with your successful setup...")
    model, history, X_test, y_test, scaler, selector = train_improved_cnn(X, y, selected_features)
    
    # Step 4: Comprehensive evaluation
    print("\n📊 Comprehensive evaluation...")
    final_accuracy, accuracy_std = evaluate_improved_cnn(model, X, y, selected_features, scaler, selector)
    
    # Step 5: Plot results
    print(f"\n📊 Creating comparison plots...")
    plot_improved_results(history, final_accuracy)
    
    # Step 6: Save results
    model.save('improved_cnn_model.h5')
    
    improvement_over_sfs = (final_accuracy - 0.50) / 0.50 * 100
    
    results_summary = {
        'method': 'Improved CNN',
        'cv_accuracy': final_accuracy,
        'accuracy_std': accuracy_std,
        'beats_sfs': final_accuracy > 0.50,
        'improvement_over_sfs': improvement_over_sfs,
        'selected_features': 25,
        'total_samples': len(X),
        'sfs_baseline': 0.50
    }
    
    pd.DataFrame([results_summary]).to_csv('improved_cnn_results.csv', index=False)
    
    print(f"\n🏆 IMPROVED CNN RESULTS:")
    print(f"   Method: Improved CNN (proper feature usage)")
    print(f"   Accuracy: {final_accuracy:.3f} ± {accuracy_std:.3f}")
    print(f"   Your SFS Baseline: 50.0%")
    print(f"   Beats SFS: {'✅ YES' if final_accuracy > 0.50 else '❌ NO'}")
    print(f"   Improvement: {improvement_over_sfs:+.1f}% vs your SFS")
    print(f"   Status: {'🎯 SUCCESS' if final_accuracy > 0.50 else '🔧 NEEDS MORE WORK'}")
    
    return results_summary

if __name__ == "__main__":
    results = run_improved_cnn_pipeline()
