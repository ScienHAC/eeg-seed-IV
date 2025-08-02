#!/usr/bin/env python3
"""
Complete Model Validation - Test Both Stage 1 (SVM) and Stage 2 (RF) 

This script validates both high-accuracy models on unseen test subjects [13, 14, 15]
for research publication requirements.

Stage 1: SVM with 77.6% training accuracy (310 DE features)
Stage 2: Random Forest with 97.7% training accuracy (enhanced features)
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
import pandas as pd
import joblib
import time
from datetime import datetime

# Setup logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_test_data():
    """Load test data from CSV files for subjects 13, 14, 15"""
    print("📊 Loading test data for subjects [13, 14, 15]...")
    
    # Find CSV directory
    csv_base = project_root / "csv"
    if not csv_base.exists():
        print(f"❌ CSV directory not found: {csv_base}")
        return None, None
    
    # Session labels for SEED-IV (4 emotions: 0=Neutral, 1=Sad, 2=Fear, 3=Happy)
    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }
    
    all_features = []
    all_labels = []
    test_subjects = [13, 14, 15]
    
    # Load data for each session and subject
    files_loaded = 0
    for session in [1, 2, 3]:
        session_dir = csv_base / str(session)
        if not session_dir.exists():
            continue
            
        for subject in test_subjects:
            subject_dir = session_dir / str(subject)
            if not subject_dir.exists():
                continue
            
            # Load all 24 trials for this subject-session
            for trial in range(1, 25):  # 1 to 24
                csv_file = subject_dir / f"de_LDS{trial}.csv"
                
                if not csv_file.exists():
                    continue
                
                try:
                    # Load CSV file with header
                    trial_data = pd.read_csv(csv_file, header=0)
                    
                    # Average across time dimension (if multiple rows)
                    if len(trial_data.shape) > 1 and trial_data.shape[0] > 1:
                        features = trial_data.mean(axis=0).values
                    else:
                        features = trial_data.values.flatten()
                    
                    # Get label for this trial
                    trial_label = session_labels[session][trial - 1]
                    
                    all_features.append(features)
                    all_labels.append(trial_label)
                    files_loaded += 1
                    
                except Exception as e:
                    continue
    
    if not all_features:
        print("❌ No test data loaded!")
        return None, None
    
    # Convert to numpy arrays
    X_test = np.array(all_features)
    y_test = np.array(all_labels)
    
    print(f"✅ Loaded {files_loaded} test samples")
    print(f"   📊 Data shape: {X_test.shape}")
    print(f"   📊 Label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_test, y_test

def validate_stage1(X_test, y_test):
    """Validate Stage 1 SVM model"""
    print(f"\n🔍 STAGE 1 VALIDATION (SVM 77.6%)")
    print("=" * 40)
    
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
        print(f"   📊 Features: {result.get('n_features_selected', 310)}")
        
        # Make predictions using pipeline
        predictor = model.pipeline
        y_pred = predictor.predict(X_test)
        
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
        
        return {
            'model_name': 'Stage 1 SVM',
            'test_accuracy': test_accuracy,
            'training_accuracy': training_accuracy,
            'generalization_gap': generalization_gap,
            'predictions': y_pred
        }
        
    except Exception as e:
        print(f"❌ Stage 1 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_stage2(X_test, y_test):
    """Validate Stage 2 Random Forest model"""
    print(f"\n🔍 STAGE 2 VALIDATION (RF 97.7%)")
    print("=" * 40)
    
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
        
        # Make predictions
        if hasattr(model, 'evaluate'):
            # Use the model's evaluate method (preferred for Stage 2)
            print(f"   📊 Using model.evaluate() method")
            # Create a dummy test to get predictions
            eval_results = model.evaluate(X_test, y_test)
            test_accuracy = eval_results.get('test_accuracy', 0)
            y_pred = eval_results.get('predictions', np.zeros(len(y_test)))
            
        elif hasattr(model, 'predict'):
            print(f"   📊 Using model.predict() method")
            y_pred = model.predict(X_test)
            test_accuracy = np.mean(y_pred == y_test)
            
        elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
            print(f"   📊 Using model.model.predict() method")
            # This is more complex - we need to transform the data first
            # Let's try to reverse-engineer the pipeline
            try:
                # Assume the model has the necessary transformers
                processed_X = X_test
                
                # Apply feature engineering if available
                if hasattr(model, 'feature_engineer') and model.feature_engineer is not None:
                    processed_X = model.feature_engineer.extract_all_features(processed_X)
                
                # Apply scaling if available
                if hasattr(model, 'scaler') and model.scaler is not None:
                    processed_X = model.scaler.transform(processed_X)
                
                # Apply feature selection if available
                if hasattr(model, 'feature_selector') and model.feature_selector is not None:
                    processed_X = model.feature_selector.transform(processed_X)
                
                y_pred = model.model.predict(processed_X)
                test_accuracy = np.mean(y_pred == y_test)
                
            except Exception as e:
                print(f"   ❌ Error in pipeline processing: {e}")
                return None
                
        else:
            print(f"❌ Model doesn't have usable predict method")
            print(f"   Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            return None
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
        
        return {
            'model_name': 'Stage 2 Random Forest',
            'test_accuracy': test_accuracy,
            'training_accuracy': training_accuracy,
            'generalization_gap': generalization_gap,
            'predictions': y_pred
        }
        
    except Exception as e:
        print(f"❌ Stage 2 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_final_report(stage1_results, stage2_results, X_test, y_test):
    """Generate comprehensive validation report"""
    print(f"\n" + "="*60)
    print(f"🏆 COMPREHENSIVE VALIDATION REPORT")
    print(f"="*60)
    
    # Data summary
    print(f"\n📊 TEST DATA SUMMARY:")
    print(f"   • Test subjects: [13, 14, 15] (unseen during training)")
    print(f"   • Total samples: {len(y_test)}")
    print(f"   • Features per sample: {X_test.shape[1]}")
    print(f"   • Emotion classes: 4 (Neutral, Sad, Fear, Happy)")
    
    # Model comparison
    print(f"\n📈 MODEL PERFORMANCE COMPARISON:")
    print(f"   {'Model':<20} {'Training':<10} {'Test':<10} {'Gap':<10} {'Status'}")
    print(f"   {'-'*60}")
    
    if stage1_results:
        gap = stage1_results['generalization_gap']
        status = "✅ Good" if gap < 0.15 else "⚠️ Fair" if gap < 0.3 else "❌ Poor"
        print(f"   {'Stage 1 SVM':<20} {stage1_results['training_accuracy']:<10.1%} {stage1_results['test_accuracy']:<10.1%} {gap:<10.1%} {status}")
    
    if stage2_results:
        gap = stage2_results['generalization_gap']
        status = "✅ Good" if gap < 0.15 else "⚠️ Fair" if gap < 0.3 else "❌ Poor"
        print(f"   {'Stage 2 RF':<20} {stage2_results['training_accuracy']:<10.1%} {stage2_results['test_accuracy']:<10.1%} {gap:<10.1%} {status}")
    
    # Research implications
    print(f"\n🎯 RESEARCH PUBLICATION READINESS:")
    
    if stage1_results and stage2_results:
        best_model = stage2_results if stage2_results['test_accuracy'] > stage1_results['test_accuracy'] else stage1_results
        print(f"   • Best performing model: {best_model['model_name']}")
        print(f"   • Best test accuracy: {best_model['test_accuracy']:.1%}")
        
        if best_model['test_accuracy'] > 0.6:
            print(f"   ✅ Strong generalization - suitable for publication")
        elif best_model['test_accuracy'] > 0.4:
            print(f"   ⚠️ Moderate generalization - needs discussion")
        else:
            print(f"   ❌ Weak generalization - requires improvement")
    
    print(f"\n📝 RECOMMENDATIONS:")
    if stage1_results and stage1_results['generalization_gap'] > 0.3:
        print(f"   • Stage 1: High variance, consider regularization")
    if stage2_results and stage2_results['generalization_gap'] > 0.3:
        print(f"   • Stage 2: High variance, consider cross-validation")
    
    print(f"   • Consider domain adaptation techniques")
    print(f"   • Validate on additional unseen subjects if available")
    print(f"   • Report generalization gaps transparently in publication")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📅 Validation completed: {timestamp}")

def main():
    """Run complete model validation pipeline"""
    print("🧠 COMPREHENSIVE EEG EMOTION RECOGNITION VALIDATION")
    print("="*55)
    print("Testing Stage 1 (SVM) and Stage 2 (RF) on unseen subjects")
    print()
    
    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None:
        return
    
    # Validate both stages
    stage1_results = validate_stage1(X_test, y_test)
    stage2_results = validate_stage2(X_test, y_test)
    
    # Generate final report
    generate_final_report(stage1_results, stage2_results, X_test, y_test)

if __name__ == "__main__":
    main()
