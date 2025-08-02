#!/usr/bin/env python3
"""
Simple model validation runner for Stage 1 and Stage 2 models
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np

# Add the comprehensive_emotion_recognition to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    """Simple validation test"""
    print("🧠 Simple EEG Model Validation")
    print("=" * 40)
    
    try:
        # Import modules
        from model_validation.config import ValidationConfig
        from model_validation.model_loader import ModelLoader
        from model_validation.data_loader import UnseenDataLoader
        
        print("✅ All modules imported")
        
        # Initialize configuration
        config = ValidationConfig()
        print(f"✅ Config: Test subjects {config.test_subjects}")
        
        # Load models
        print("\n📋 Loading Stage models...")
        model_loader = ModelLoader(config)
        models = model_loader.load_all_models()
        
        print(f"✅ Loaded {len(models)} models:")
        for name, model_data in models.items():
            metadata = model_data.get('metadata', {})
            model_type = metadata.get('model_type', 'Unknown')
            accuracy = metadata.get('accuracy', 0)
            n_features = metadata.get('n_features', 0)
            print(f"   📊 {name}: {model_type} ({accuracy:.1%}, {n_features} features)")
        
        # Load test data
        print(f"\n📊 Loading test data for subjects {config.test_subjects}...")
        data_loader = UnseenDataLoader(config)
        test_data = data_loader.load_unseen_test_data()
        
        if test_data is None:
            print("❌ Failed to load test data")
            return
        
        X_test, y_test = test_data
        print(f"✅ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"✅ Label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Test each model
        print(f"\n🔍 Testing models on unseen data...")
        
        for model_name, model_data in models.items():
            print(f"\n[Testing {model_name}]")
            
            try:
                model = model_data['model']
                
                # Check feature compatibility
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    if expected_features != X_test.shape[1]:
                        print(f"   ⚠️ Feature mismatch: model expects {expected_features}, got {X_test.shape[1]}")
                        
                        # Try to handle feature mismatch
                        if X_test.shape[1] > expected_features:
                            print(f"   🔧 Truncating features to {expected_features}")
                            X_test_model = X_test[:, :expected_features]
                        else:
                            print(f"   ❌ Cannot proceed: insufficient features")
                            continue
                    else:
                        X_test_model = X_test
                else:
                    X_test_model = X_test
                
                # Make predictions
                y_pred = model.predict(X_test_model)
                
                # Calculate accuracy
                accuracy = np.mean(y_pred == y_test)
                
                print(f"   ✅ Test Accuracy: {accuracy:.1%}")
                
                # Per-class accuracy
                emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
                print(f"   📊 Per-class performance:")
                for class_id in range(4):
                    mask = y_test == class_id
                    if mask.sum() > 0:
                        class_acc = np.mean(y_pred[mask] == y_test[mask])
                        print(f"      {emotions[class_id]}: {class_acc:.1%} ({mask.sum()} samples)")
                
            except Exception as e:
                print(f"   ❌ Error testing {model_name}: {e}")
                continue
        
        print(f"\n🎉 Validation completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
