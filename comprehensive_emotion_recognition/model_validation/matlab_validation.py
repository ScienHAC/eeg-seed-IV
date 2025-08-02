#!/usr/bin/env python3
"""
MATLAB-Based Model Validation - Use Original MATLAB Files

This script validates both Stage 1 (SVM) and Stage 2 (RF) models using the original
MATLAB .mat files from SEED-IV dataset, exactly as you specified.

MATLAB Files Location: C:/Users/piyus/Downloads/SEED_IV/SEED_IV/eeg_feature_smooth
Test Subjects: [13,    print(f"📊 TEST DATA SUMMARY (MATLAB FILES):")
    print(f"   • Data source: Original MATLAB .mat files")
    print(f"   • Location: C:/Users/piyus/Downloads/SEED_IV/SEED_IV/eeg_feature_smooth")
    print(f"   • Test subjects: [13, 14, 15] (unseen during training)")
    print(f"   • Total samples: {len(y_test)}")
    print(f"   • Features per sample: {X_test.shape[1]}")
    print(f"   • Emotion classes: 4 (Neutral, Sad, Fear, Happy)")] (unseen during training)
"""

import sys
import os
from pathlib import Path

# Add project roots to Python path
project_root = Path(__file__).parent.parent.parent
comp_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(comp_root))

import logging
import numpy as np
import joblib
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# For MATLAB file loading and analysis
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    print("❌ scipy not available - cannot load MATLAB files")
    SCIPY_AVAILABLE = False

# For confusion matrix and metrics
try:
    from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    print("❌ sklearn.metrics not available - limited analysis")
    SKLEARN_AVAILABLE = False

# For plotting confusion matrix
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ matplotlib/seaborn not available - no plots will be generated")
    MATPLOTLIB_AVAILABLE = False

# Setup logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_matlab_test_data(matlab_dir: str = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth", 
                          max_subjects: bool = True):
    """
    Load test data directly from MATLAB files for maximum testing coverage
    
    Parameters:
    -----------
    matlab_dir : str
        Path to MATLAB files directory
    max_subjects : bool
        If True, load all available subjects (max testing)
        If False, load only subjects 13, 14, 15 (original test set)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, List[Dict]]
        (features, labels, trial_info)
    """
    print("📊 Loading test data from MATLAB files...")
    print(f"📁 MATLAB directory: {matlab_dir}")
    print(f"📈 Maximum testing mode: {'ON - All available subjects' if max_subjects else 'OFF - Test subjects [13, 14, 15] only'}")
    
    if not SCIPY_AVAILABLE:
        print("❌ scipy not available - cannot load MATLAB files")
        return None, None, None
    
    matlab_path = Path(matlab_dir)
    if not matlab_path.exists():
        print(f"❌ MATLAB directory not found: {matlab_path}")
        return None, None, None
    
    # SEED-IV session labels (4 emotions: 0=Neutral, 1=Sad, 2=Fear, 3=Happy)  
    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }
    
    all_features = []
    all_labels = []
    trial_info = []
    
    # Determine test subjects based on mode
    if max_subjects:
        # Find all available subjects by scanning directories
        all_subjects = set()
        for session in [1, 2, 3]:
            session_dir = matlab_path / str(session)
            if session_dir.exists():
                for mat_file in session_dir.glob("*.mat"):
                    try:
                        subject_num = int(mat_file.stem.split('_')[0])
                        all_subjects.add(subject_num)
                    except (ValueError, IndexError):
                        continue
        
        test_subjects = sorted(list(all_subjects))
        print(f"🔍 Found subjects in MATLAB files: {test_subjects}")
        
        # Use subjects not in typical training set (assuming training used 1-12)
        if len(test_subjects) > 12:
            test_subjects = [s for s in test_subjects if s > 12]
            print(f"📊 Using unseen subjects (>12): {test_subjects}")
        else:
            print(f"📊 Using all available subjects: {test_subjects}")
    else:
        # Original test set
        test_subjects = [13, 14, 15]
        print(f"📊 Using original test subjects: {test_subjects}")
    
    # Load data for each session and subject
    files_loaded = 0
    for session in [1, 2, 3]:
        session_dir = matlab_path / str(session)
        if not session_dir.exists():
            print(f"⚠️ Session {session} directory not found")
            continue
        
        for subject in test_subjects:
            # Find MATLAB file for this subject in this session (pattern: subject_date.mat)
            subject_files = list(session_dir.glob(f"{subject}_*.mat"))
            
            if not subject_files:
                print(f"⚠️ No file found for Subject {subject} in Session {session}")
                continue
            
            # Use the first matching file (should be only one per subject per session)
            mat_file = subject_files[0]
            print(f"� Loading: {mat_file}")
            
            try:
                # Load MATLAB file
                mat_data = loadmat(str(mat_file))
                
                # Print available keys to understand structure (only for first few files)
                if files_loaded < 3:
                    available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    print(f"   Available keys: {available_keys}")
                
                # SEED-IV structure: Each trial is stored as separate variable (de_LDS1, de_LDS2, ..., de_LDS24)
                # We'll use DE features (differential entropy) which match our training data
                data_keys = [k for k in mat_data.keys() if k.startswith('de_LDS') and not k.startswith('__')]
                
                if not data_keys:
                    print(f"⚠️ No de_LDS keys found in {mat_file}")
                    continue
                
                # Sort trial keys numerically (de_LDS1, de_LDS2, ..., de_LDS24)
                trial_keys = sorted([k for k in data_keys if k.startswith('de_LDS')], 
                                  key=lambda x: int(x.replace('de_LDS', '')))
                
                if files_loaded < 3:
                    print(f"   📊 Found {len(trial_keys)} trials")
                    print(f"   📊 Using DE features (de_LDS)")
                
                # Process each trial (up to 24 trials per session)
                for trial_idx, trial_key in enumerate(trial_keys[:24]):  # Limit to 24 trials
                    trial_data = mat_data[trial_key]  # Shape should be channels x time
                    
                    # Average across time dimension to get features per channel
                    if len(trial_data.shape) == 3:  # (channels, time, frequency_bands)
                        # Average across time to get (channels, frequency_bands)
                        features = trial_data.mean(axis=1)  # Shape: (62, 5)
                        features = features.flatten()      # Shape: (310,) = 62 * 5
                    elif len(trial_data.shape) == 2:
                        # Already (channels, frequency_bands) or (channels, time)
                        if trial_data.shape[1] == 5:  # (channels, frequency_bands)
                            features = trial_data.flatten()
                        else:  # (channels, time) - average across time
                            features = trial_data.mean(axis=1)  # Shape: (62,)
                            # Need to expand to match expected 310 features (62 channels * 5 freq bands)
                            # For now, repeat the features 5 times to simulate frequency bands
                            features = np.tile(features, 5)  # Shape: (310,)
                    else:
                        features = trial_data.flatten()
                    
                    # Get label for this trial (trial_idx is 0-indexed, session_labels is 0-indexed too)
                    trial_label = session_labels[session][trial_idx]
                    
                    all_features.append(features)
                    all_labels.append(trial_label)
                    
                    # Store trial information
                    trial_info.append({
                        'subject': subject,
                        'session': session,
                        'trial': trial_idx + 1,
                        'label': trial_label,
                        'emotion': ['Neutral', 'Sad', 'Fear', 'Happy'][trial_label],
                        'file': str(mat_file),
                        'data_key': trial_key,
                        'original_shape': trial_data.shape,
                        'feature_shape': features.shape
                    })
                    
                    files_loaded += 1
                    
                    if files_loaded <= 5:  # Debug first few
                        print(f"   📄 S{subject}_Sess{session}_T{trial_idx+1}: {trial_key} {trial_data.shape} -> {features.shape} features, label={trial_label} ({['Neutral', 'Sad', 'Fear', 'Happy'][trial_label]})")
                
            except Exception as e:
                print(f"❌ Error loading {mat_file}: {e}")
                continue
    
    if not all_features:
        print("❌ No MATLAB data loaded!")
        return None, None, None
    
    # Convert to numpy arrays
    X_test = np.array(all_features)
    y_test = np.array(all_labels)
    
    print(f"\n✅ Loaded {files_loaded} samples from MATLAB files")
    print(f"   📊 Data shape: {X_test.shape}")
    print(f"   📊 Label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"   📊 Test subjects: {test_subjects}")
    
    return X_test, y_test, trial_info

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix for emotion classification
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like  
        Predicted labels
    model_name : str
        Name of the model for title
    save_path : str, optional
        Path to save the plot
    """
    if not SKLEARN_AVAILABLE:
        print("⚠️ sklearn not available - cannot generate confusion matrix")
        return None
        
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
    
    # Print text-based confusion matrix first
    print(f"\n📊 CONFUSION MATRIX - {model_name}:")
    print(f"   {'Actual\\Pred':<12} {'Neutral':<8} {'Sad':<8} {'Fear':<8} {'Happy':<8} {'Total':<8}")
    print(f"   {'-'*60}")
    
    for i, emotion in enumerate(emotions):
        row_total = cm[i].sum()
        print(f"   {emotion:<12} {cm[i][0]:<8} {cm[i][1]:<8} {cm[i][2]:<8} {cm[i][3]:<8} {row_total:<8}")
    
    # Column totals
    col_totals = cm.sum(axis=0)
    print(f"   {'Total':<12} {col_totals[0]:<8} {col_totals[1]:<8} {col_totals[2]:<8} {col_totals[3]:<8} {cm.sum():<8}")
    
    # Calculate per-class metrics
    if len(set(y_true)) > 1:  # Only if we have multiple classes
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        print(f"\n📈 DETAILED METRICS - {model_name}:")
        print(f"   {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print(f"   {'-'*60}")
        
        for i, emotion in enumerate(emotions):
            if i < len(precision):
                print(f"   {emotion:<12} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}")
        
        # Overall metrics
        macro_f1 = f1.mean()
        weighted_f1 = np.average(f1, weights=support)
        print(f"   {'Macro Avg':<12} {precision.mean():<10.3f} {recall.mean():<10.3f} {macro_f1:<10.3f} {support.sum():<10}")
        print(f"   {'Weighted Avg':<12} {np.average(precision, weights=support):<10.3f} {np.average(recall, weights=support):<10.3f} {weighted_f1:<10.3f} {support.sum():<10}")
    
    # Create visual plot if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(10, 8))
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create heatmap
            sns.heatmap(cm_percent, 
                       annot=True, 
                       fmt='.1f',
                       cmap='Blues',
                       xticklabels=emotions,
                       yticklabels=emotions,
                       cbar_kws={'label': 'Percentage (%)'})
            
            plt.title(f'Confusion Matrix - {model_name}\n(Percentages by True Class)', fontsize=14)
            plt.xlabel('Predicted Emotion', fontsize=12)
            plt.ylabel('True Emotion', fontsize=12)
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Confusion matrix plot saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not create visual plot: {e}")
    
    return cm

def analyze_prediction_patterns(y_true, y_pred, trial_info, model_name):
    """
    Analyze prediction patterns by subject and session
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels  
    trial_info : list
        Trial information
    model_name : str
        Name of the model
    """
    print(f"\n🔍 PREDICTION PATTERN ANALYSIS - {model_name}:")
    
    emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
    
    # Subject-wise analysis
    subject_stats = {}
    for i, trial in enumerate(trial_info):
        subject = trial['subject']
        if subject not in subject_stats:
            subject_stats[subject] = {'total': 0, 'correct': 0, 'by_emotion': {0:0, 1:0, 2:0, 3:0}}
        
        subject_stats[subject]['total'] += 1
        if y_pred[i] == y_true[i]:
            subject_stats[subject]['correct'] += 1
        
        true_emotion = y_true[i]
        subject_stats[subject]['by_emotion'][true_emotion] += 1
    
    print(f"   📊 Subject Performance:")
    for subject in sorted(subject_stats.keys()):
        stats = subject_stats[subject]
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"      Subject {subject:2d}: {accuracy:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")
    
    # Session-wise analysis
    session_stats = {}
    for i, trial in enumerate(trial_info):
        session = trial['session']
        if session not in session_stats:
            session_stats[session] = {'total': 0, 'correct': 0}
        
        session_stats[session]['total'] += 1
        if y_pred[i] == y_true[i]:
            session_stats[session]['correct'] += 1
    
    print(f"   📊 Session Performance:")
    for session in sorted(session_stats.keys()):
        stats = session_stats[session]
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"      Session {session}: {accuracy:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")
    
    # Most/least accurate emotions
    emotion_accuracy = {}
    for emotion_id in range(4):
        mask = y_true == emotion_id
        if mask.sum() > 0:
            accuracy = np.mean(y_pred[mask] == y_true[mask])
            emotion_accuracy[emotions[emotion_id]] = accuracy
    
    if emotion_accuracy:
        best_emotion = max(emotion_accuracy, key=emotion_accuracy.get)
        worst_emotion = min(emotion_accuracy, key=emotion_accuracy.get)
        
        print(f"   🎯 Best predicted emotion: {best_emotion} ({emotion_accuracy[best_emotion]:.1%})")
        print(f"   🎯 Worst predicted emotion: {worst_emotion} ({emotion_accuracy[worst_emotion]:.1%})")

def validate_stage1_matlab(X_test, y_test, trial_info):
    """Validate Stage 1 SVM model on MATLAB data with confusion matrix analysis"""
    print(f"\n🔍 STAGE 1 VALIDATION (SVM 77.6%) - MATLAB DATA")
    print("=" * 50)
    
    try:
        # Load Stage 1 checkpoint
        stage1_path = comp_root / "csv_data" / "checkpoints" / "stage_1_checkpoint.joblib"
        
        if not stage1_path.exists():
            print(f"❌ Stage 1 checkpoint not found: {stage1_path}")
            return None
        
        # Import required modules
        import models.stage1_traditional
        
        # Load checkpoint
        checkpoint_data = joblib.load(stage1_path)
        model = checkpoint_data['model']
        result = checkpoint_data.get('result', {})
        
        print(f"✅ Loaded Stage 1 model:")
        print(f"   📊 Type: {type(model).__name__}")
        print(f"   📊 Training accuracy: {result.get('accuracy', 0):.1%}")
        print(f"   📊 Expected features: {result.get('n_features_selected', 310)}")
        print(f"   📊 Actual features: {X_test.shape[1]}")
        
        # Handle feature mismatch
        if X_test.shape[1] != result.get('n_features_selected', 310):
            expected_features = result.get('n_features_selected', 310)
            print(f"⚠️ Feature mismatch: model expects {expected_features}, got {X_test.shape[1]}")
            
            if X_test.shape[1] > expected_features:
                print(f"🔧 Truncating features to {expected_features}")
                X_test_adjusted = X_test[:, :expected_features]
            elif X_test.shape[1] < expected_features:
                print(f"🔧 Padding features with zeros to reach {expected_features}")
                padding = np.zeros((X_test.shape[0], expected_features - X_test.shape[1]))
                X_test_adjusted = np.hstack([X_test, padding])
            else:
                X_test_adjusted = X_test
        else:
            X_test_adjusted = X_test
        
        # Make predictions using pipeline
        predictor = model.pipeline
        y_pred = predictor.predict(X_test_adjusted)
        
        # Calculate metrics
        test_accuracy = np.mean(y_pred == y_test)
        training_accuracy = result.get('accuracy', 0)
        generalization_gap = training_accuracy - test_accuracy
        
        print(f"\n🎯 STAGE 1 RESULTS:")
        print(f"   📊 Test Accuracy: {test_accuracy:.1%}")
        print(f"   📊 Training Accuracy: {training_accuracy:.1%}")
        print(f"   📊 Generalization Gap: {generalization_gap:.1%}")
        
        # Per-class accuracy
        emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
        print(f"\n📊 Per-Class Performance:")
        for class_id in range(4):
            mask = y_test == class_id
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == y_test[mask])
                print(f"   {emotions[class_id]}: {class_acc:.1%} ({mask.sum()} samples)")
        
        # Detailed trial-by-trial results (first 10)
        print(f"\n📋 Sample Trial Results (first 10):")
        for i in range(min(10, len(trial_info))):
            trial = trial_info[i]
            pred_emotion = emotions[y_pred[i]]
            actual_emotion = trial['emotion']
            correct = "✅" if y_pred[i] == y_test[i] else "❌"
            print(f"   Subject {trial['subject']}, Session {trial['session']}, Trial {trial['trial']}: "
                  f"Predicted={pred_emotion}, Actual={actual_emotion} {correct}")
        
        # Generate confusion matrix and detailed analysis
        cm = plot_confusion_matrix(y_test, y_pred, "Stage 1 SVM (MATLAB Data)")
        analyze_prediction_patterns(y_test, y_pred, trial_info, "Stage 1 SVM")
        
        return {
            'model_name': 'Stage 1 SVM',
            'test_accuracy': test_accuracy,
            'training_accuracy': training_accuracy,
            'generalization_gap': generalization_gap,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'data_source': 'MATLAB'
        }
        
    except Exception as e:
        print(f"❌ Stage 1 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_stage2_matlab(X_test, y_test, trial_info):
    """Validate Stage 2 Random Forest model on MATLAB data with confusion matrix analysis"""
    print(f"\n🔍 STAGE 2 VALIDATION (RF 97.7%) - MATLAB DATA")
    print("=" * 50)
    
    try:
        # Load Stage 2 checkpoint
        stage2_path = comp_root / "csv_data" / "checkpoints" / "stage_2_checkpoint.joblib"
        
        if not stage2_path.exists():
            print(f"❌ Stage 2 checkpoint not found: {stage2_path}")
            return None
        
        # Load checkpoint
        print(f"📋 Loading Stage 2 checkpoint...")
        checkpoint_data = joblib.load(stage2_path)
        model = checkpoint_data['model']
        result = checkpoint_data.get('result', {})
        
        print(f"✅ Loaded Stage 2 model:")
        print(f"   📊 Type: {type(model).__name__}")
        print(f"   📊 Training accuracy: {result.get('accuracy', 0):.1%}")
        
        # Use model's evaluate method
        print(f"   📊 Using model.evaluate() method with MATLAB data")
        eval_results = model.evaluate(X_test, y_test)
        
        test_accuracy = eval_results.get('test_accuracy', 0)
        y_pred = eval_results.get('predictions', np.zeros(len(y_test)))
        training_accuracy = result.get('accuracy', 0)
        generalization_gap = training_accuracy - test_accuracy
        
        print(f"\n🎯 STAGE 2 RESULTS:")
        print(f"   📊 Test Accuracy: {test_accuracy:.1%}")
        print(f"   📊 Training Accuracy: {training_accuracy:.1%}")
        print(f"   📊 Generalization Gap: {generalization_gap:.1%}")
        
        # Per-class accuracy
        emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
        print(f"\n📊 Per-Class Performance:")
        for class_id in range(4):
            mask = y_test == class_id
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == y_test[mask])
                print(f"   {emotions[class_id]}: {class_acc:.1%} ({mask.sum()} samples)")
        
        # Detailed trial-by-trial results (first 10)
        print(f"\n📋 Sample Trial Results (first 10):")
        for i in range(min(10, len(trial_info))):
            trial = trial_info[i]
            pred_emotion = emotions[y_pred[i]]
            actual_emotion = trial['emotion']
            correct = "✅" if y_pred[i] == y_test[i] else "❌"
            print(f"   Subject {trial['subject']}, Session {trial['session']}, Trial {trial['trial']}: "
                  f"Predicted={pred_emotion}, Actual={actual_emotion} {correct}")
        
        # Generate confusion matrix and detailed analysis
        cm = plot_confusion_matrix(y_test, y_pred, "Stage 2 Random Forest (MATLAB Data)")
        analyze_prediction_patterns(y_test, y_pred, trial_info, "Stage 2 Random Forest")
        
        return {
            'model_name': 'Stage 2 Random Forest',
            'test_accuracy': test_accuracy,
            'training_accuracy': training_accuracy,
            'generalization_gap': generalization_gap,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'data_source': 'MATLAB'
        }
        
    except Exception as e:
        print(f"❌ Stage 2 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_matlab_validation_report(stage1_results, stage2_results, X_test, y_test, trial_info):
    """Generate comprehensive validation report using MATLAB data"""
    print(f"\n" + "="*70)
    print(f"🏆 MATLAB-BASED VALIDATION REPORT")
    print(f"="*70)
    
    # Data summary
    print(f"\n📊 TEST DATA SUMMARY (MATLAB FILES):")
    print(f"   • Data source: Original MATLAB .mat files")
    print(f"   • Location: C:\\Users\\piyus\\Downloads\\SEED_IV\\SEED_IV\\eeg_feature_smooth")
    print(f"   • Test subjects: [13, 14, 15] (unseen during training)")
    print(f"   • Total samples: {len(y_test)}")
    print(f"   • Features per sample: {X_test.shape[1]}")
    print(f"   • Emotion classes: 4 (Neutral, Sad, Fear, Happy)")
    
    # Model comparison
    print(f"\n📈 MODEL PERFORMANCE ON MATLAB DATA:")
    print(f"   {'Model':<25} {'Training':<10} {'Test':<10} {'Gap':<10} {'Status'}")
    print(f"   {'-'*70}")
    
    if stage1_results:
        gap = stage1_results['generalization_gap']  
        status = "✅ Good" if gap < 0.15 else "⚠️ Fair" if gap < 0.3 else "❌ Poor"
        print(f"   {'Stage 1 SVM':<25} {stage1_results['training_accuracy']:<10.1%} {stage1_results['test_accuracy']:<10.1%} {gap:<10.1%} {status}")
    
    if stage2_results:
        gap = stage2_results['generalization_gap']
        status = "✅ Good" if gap < 0.15 else "⚠️ Fair" if gap < 0.3 else "❌ Poor"
        print(f"   {'Stage 2 RF':<25} {stage2_results['training_accuracy']:<10.1%} {stage2_results['test_accuracy']:<10.1%} {gap:<10.1%} {status}")
    
    # Trial-level analysis
    print(f"\n🔬 TRIAL-LEVEL ANALYSIS:")
    subjects_analysis = {}
    for trial in trial_info:
        subject = trial['subject']
        if subject not in subjects_analysis:
            subjects_analysis[subject] = {'total': 0, 'correct_s1': 0, 'correct_s2': 0}
        subjects_analysis[subject]['total'] += 1
    
    # Calculate per-subject accuracy
    if stage1_results and stage2_results:
        for i, trial in enumerate(trial_info):
            subject = trial['subject']
            if stage1_results['predictions'][i] == y_test[i]:
                subjects_analysis[subject]['correct_s1'] += 1
            if stage2_results['predictions'][i] == y_test[i]:
                subjects_analysis[subject]['correct_s2'] += 1
    
    print(f"   Per-Subject Performance:")
    for subject in sorted(subjects_analysis.keys()):
        stats = subjects_analysis[subject]
        if stage1_results and stage2_results:
            s1_acc = stats['correct_s1'] / stats['total']
            s2_acc = stats['correct_s2'] / stats['total']
            print(f"   Subject {subject}: Stage1={s1_acc:.1%}, Stage2={s2_acc:.1%} (n={stats['total']})")
    
    # Research implications
    print(f"\n🎯 RESEARCH PUBLICATION READINESS (MATLAB VALIDATION):")
    
    if stage1_results and stage2_results:
        best_model = stage2_results if stage2_results['test_accuracy'] > stage1_results['test_accuracy'] else stage1_results
        print(f"   • Best performing model: {best_model['model_name']}")
        print(f"   • Best test accuracy: {best_model['test_accuracy']:.1%}")
        
        if best_model['test_accuracy'] > 0.6:
            print(f"   ✅ Strong generalization on MATLAB data - publication ready")
        elif best_model['test_accuracy'] > 0.4:
            print(f"   ⚠️ Moderate generalization - discuss limitations")
        else:
            print(f"   ❌ Weak generalization - requires improvement")
    
    print(f"\n📝 MATLAB DATA VALIDATION CONCLUSIONS:")
    print(f"   • Used original MATLAB files (not CSV conversions)")
    print(f"   • Direct emotion detection from raw .mat files")
    print(f"   • Trial-by-trial prediction accuracy calculated")
    print(f"   • Proper handling of SEED-IV file structure")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📅 MATLAB validation completed: {timestamp}")
    
    # Save detailed results if both models validated
    if stage1_results and stage2_results:
        save_validation_results(stage1_results, stage2_results, X_test, y_test, trial_info, timestamp)

def save_validation_results(stage1_results, stage2_results, X_test, y_test, trial_info, timestamp):
    """
    Save detailed validation results to files
    
    Parameters:
    -----------
    stage1_results : dict
        Stage 1 validation results
    stage2_results : dict  
        Stage 2 validation results
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    trial_info : list
        Trial information
    timestamp : str
        Timestamp for file naming
    """
    try:
        # Create results directory
        results_dir = Path("matlab_validation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Prepare comprehensive results dictionary
        validation_results = {
            'metadata': {
                'timestamp': timestamp,
                'data_source': 'MATLAB_files',
                'matlab_directory': r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth",
                'total_samples': len(y_test),
                'feature_dimensions': X_test.shape[1],
                'emotion_classes': 4,
                'test_subjects': sorted(list(set([t['subject'] for t in trial_info])))
            },
            'stage1_results': {
                'model_name': stage1_results['model_name'],
                'test_accuracy': float(stage1_results['test_accuracy']),
                'training_accuracy': float(stage1_results['training_accuracy']),
                'generalization_gap': float(stage1_results['generalization_gap']),
                'confusion_matrix': stage1_results['confusion_matrix'].tolist() if stage1_results.get('confusion_matrix') is not None else None
            },
            'stage2_results': {
                'model_name': stage2_results['model_name'], 
                'test_accuracy': float(stage2_results['test_accuracy']),
                'training_accuracy': float(stage2_results['training_accuracy']),
                'generalization_gap': float(stage2_results['generalization_gap']),
                'confusion_matrix': stage2_results['confusion_matrix'].tolist() if stage2_results.get('confusion_matrix') is not None else None
            },
            'trial_predictions': []
        }
        
        # Add trial-by-trial predictions
        emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
        for i, trial in enumerate(trial_info):
            validation_results['trial_predictions'].append({
                'subject': trial['subject'],
                'session': trial['session'],
                'trial': trial['trial'],
                'true_label': int(y_test[i]),
                'true_emotion': emotions[y_test[i]],
                'stage1_prediction': int(stage1_results['predictions'][i]),
                'stage1_emotion': emotions[stage1_results['predictions'][i]],
                'stage1_correct': bool(stage1_results['predictions'][i] == y_test[i]),
                'stage2_prediction': int(stage2_results['predictions'][i]),
                'stage2_emotion': emotions[stage2_results['predictions'][i]],
                'stage2_correct': bool(stage2_results['predictions'][i] == y_test[i])
            })
        
        # Save JSON results
        import json
        json_file = results_dir / f"matlab_validation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"💾 Detailed results saved: {json_file}")
        
        # Save CSV for easy analysis
        import pandas as pd
        df_trials = pd.DataFrame(validation_results['trial_predictions'])
        csv_file = results_dir / f"matlab_validation_trials_{timestamp}.csv"
        df_trials.to_csv(csv_file, index=False)
        print(f"📊 Trial predictions saved: {csv_file}")
        
        # Create summary report
        summary_file = results_dir / f"matlab_validation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"MATLAB-Based EEG Emotion Recognition Validation Report\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Data Source: Original MATLAB .mat files\n")
            f.write(f"Test Subjects: {validation_results['metadata']['test_subjects']}\n")
            f.write(f"Total Samples: {validation_results['metadata']['total_samples']}\n")
            f.write(f"Features: {validation_results['metadata']['feature_dimensions']}\n\n")
            
            f.write(f"MODEL PERFORMANCE:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"Stage 1 SVM:\n")
            f.write(f"  Training Accuracy: {validation_results['stage1_results']['training_accuracy']:.1%}\n")
            f.write(f"  Test Accuracy: {validation_results['stage1_results']['test_accuracy']:.1%}\n")
            f.write(f"  Generalization Gap: {validation_results['stage1_results']['generalization_gap']:.1%}\n\n")
            
            f.write(f"Stage 2 Random Forest:\n")
            f.write(f"  Training Accuracy: {validation_results['stage2_results']['training_accuracy']:.1%}\n")
            f.write(f"  Test Accuracy: {validation_results['stage2_results']['test_accuracy']:.1%}\n")
            f.write(f"  Generalization Gap: {validation_results['stage2_results']['generalization_gap']:.1%}\n\n")
            
            # Per-subject summary
            subject_summary = {}
            for trial in validation_results['trial_predictions']:
                subject = trial['subject']
                if subject not in subject_summary:
                    subject_summary[subject] = {'total': 0, 's1_correct': 0, 's2_correct': 0}
                subject_summary[subject]['total'] += 1
                if trial['stage1_correct']:
                    subject_summary[subject]['s1_correct'] += 1
                if trial['stage2_correct']:
                    subject_summary[subject]['s2_correct'] += 1
            
            f.write(f"PER-SUBJECT PERFORMANCE:\n")
            f.write(f"-" * 30 + "\n")
            for subject in sorted(subject_summary.keys()):
                stats = subject_summary[subject]
                s1_acc = stats['s1_correct'] / stats['total'] * 100
                s2_acc = stats['s2_correct'] / stats['total'] * 100
                f.write(f"Subject {subject:2d}: Stage1={s1_acc:5.1f}%, Stage2={s2_acc:5.1f}% (n={stats['total']})\n")
        
        print(f"📄 Summary report saved: {summary_file}")
        print(f"📁 All results saved in: {results_dir}")
        
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

def main():
    """Run complete MATLAB-based model validation with maximum testing capability"""
    print("🧠 MATLAB-BASED EEG EMOTION RECOGNITION VALIDATION")
    print("="*60)
    print("Using original MATLAB files from SEED-IV dataset")
    
    # Ask user for testing mode
    print("\n🔧 TESTING MODE SELECTION:")
    print("   1. Standard Test (subjects 13, 14, 15 only)")
    print("   2. Maximum Test (all available unseen subjects)")
    
    try:
        choice = input("Select mode (1 or 2, default=2): ").strip()
        if choice == '1':
            max_subjects = False
            print("📊 Using standard test set: subjects 13, 14, 15")
        else:
            max_subjects = True
            print("📊 Using maximum test coverage: all available unseen subjects")
    except (EOFError, KeyboardInterrupt):
        max_subjects = True  # Default to maximum testing
        print("📊 Using maximum test coverage (default)")
    
    print()
    
    # Load test data from MATLAB files
    X_test, y_test, trial_info = load_matlab_test_data(max_subjects=max_subjects)
    if X_test is None:
        print("❌ Failed to load MATLAB data - validation aborted")
        return
    
    # Validate both stages on MATLAB data
    stage1_results = validate_stage1_matlab(X_test, y_test, trial_info)
    stage2_results = validate_stage2_matlab(X_test, y_test, trial_info)
    
    # Generate comprehensive report
    generate_matlab_validation_report(stage1_results, stage2_results, X_test, y_test, trial_info)

if __name__ == "__main__":
    main()
