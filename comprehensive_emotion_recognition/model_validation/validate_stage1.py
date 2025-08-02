#!/usr/bin/env python3
"""
Stage 1 only validation - test just the SVM model
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import joblib

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    """Validate Stage 1 model only"""
    print("🧠 Stage 1 Model Validation (SVM 77.6%)")
    print("=" * 45)
    
    try:
        # Load Stage 1 checkpoint directly
        stage1_path = Path(__file__).parent.parent / "csv_data" / "checkpoints" / "stage_1_checkpoint.joblib"
        print(f"📋 Loading Stage 1 checkpoint: {stage1_path}")
        
        if not stage1_path.exists():
            print(f"❌ Stage 1 checkpoint not found: {stage1_path}")
            return
        
        # Load checkpoint
        checkpoint_data = joblib.load(stage1_path)
        model = checkpoint_data['model']
        result = checkpoint_data.get('result', {})
        
        print(f"✅ Loaded Stage 1 model:")
        print(f"   📊 Type: {result.get('model_type', 'SVM')}")
        print(f"   📊 Training accuracy: {result.get('accuracy', 0):.1%}")
        print(f"   📊 Features: {result.get('n_features_selected', 310)}")
        
        # Load test data from CSV files
        print(f"\n📊 Loading test data for subjects [13, 14, 15]...")
        
        # Load CSV data
        csv_base = Path(__file__).parent.parent.parent / "csv"
        print(f"📁 CSV directory: {csv_base}")
        
        if not csv_base.exists():
            print(f"❌ CSV directory not found: {csv_base}")
            return
        
        # Session labels for SEED-IV
        session_labels = {
            1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
            3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
        }
        
        all_features = []
        all_labels = []
        test_subjects = [13, 14, 15]
        
        import pandas as pd
        
        # Load data for each session and subject
        for session in [1, 2, 3]:
            session_dir = csv_base / str(session)
            if not session_dir.exists():
                print(f"⚠️ Session {session} directory not found: {session_dir}")
                continue
                
            for subject in test_subjects:
                subject_dir = session_dir / str(subject)
                if not subject_dir.exists():
                    print(f"⚠️ Subject {subject} directory not found in session {session}")
                    continue
                
                print(f"📁 Loading Subject {subject}, Session {session}")
                
                # Load all 24 trials for this subject-session
                for trial in range(1, 25):  # 1 to 24
                    csv_file = subject_dir / f"de_LDS{trial}.csv"
                    
                    if not csv_file.exists():
                        continue
                    
                    try:
                        # Load CSV file
                        trial_data = pd.read_csv(csv_file, header=None)
                        
                        # Average across time dimension (if multiple rows)
                        if len(trial_data.shape) > 1 and trial_data.shape[0] > 1:
                            features = trial_data.mean(axis=0).values  # Average across time
                        else:
                            features = trial_data.values.flatten()  # Already flattened
                        
                        # Get label for this trial
                        trial_label = session_labels[session][trial - 1]  # trial-1 because list is 0-indexed
                        
                        all_features.append(features)
                        all_labels.append(trial_label)
                        
                    except Exception as e:
                        print(f"⚠️ Error loading {csv_file}: {e}")
                        continue
        
        if not all_features:
            print("❌ No test data loaded!")
            return
        
        # Convert to numpy arrays
        X_test = np.array(all_features)
        y_test = np.array(all_labels)
        
        print(f"✅ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"✅ Label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Test Stage 1 model
        print(f"\n🔍 Testing Stage 1 (SVM) on unseen data...")
        
        try:
            # Check feature compatibility
            expected_features = getattr(model, 'n_features_in_', X_test.shape[1])
            if expected_features != X_test.shape[1]:
                print(f"⚠️ Feature mismatch: model expects {expected_features}, got {X_test.shape[1]}")
                
                if X_test.shape[1] > expected_features:
                    print(f"🔧 Truncating features to {expected_features}")
                    X_test = X_test[:, :expected_features]
                else:
                    print(f"❌ Insufficient features")
                    return
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            test_accuracy = np.mean(y_pred == y_test)
            
            print(f"\n🎯 VALIDATION RESULTS:")
            print(f"   📊 Test Accuracy: {test_accuracy:.1%}")
            print(f"   📊 Training Accuracy: {result.get('accuracy', 0):.1%}")
            print(f"   📊 Generalization Gap: {result.get('accuracy', 0) - test_accuracy:.1%}")
            
            # Per-class accuracy
            emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
            print(f"\n📊 Per-Class Performance:")
            for class_id in range(4):
                mask = y_test == class_id
                if mask.sum() > 0:
                    class_acc = np.mean(y_pred[mask] == y_test[mask])
                    print(f"   {emotions[class_id]}: {class_acc:.1%} ({mask.sum()} samples)")
            
            # Overfitting assessment
            gap = result.get('accuracy', 0) - test_accuracy
            if gap < 0.05:
                status = "✅ Excellent generalization"
            elif gap < 0.1:
                status = "⚠️ Good generalization"
            else:
                status = "❌ Possible overfitting"
            
            print(f"\n🏆 CONCLUSION:")
            print(f"   {status}")
            print(f"   Stage 1 SVM achieves {test_accuracy:.1%} on unseen subjects [13,14,15]")
            
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
