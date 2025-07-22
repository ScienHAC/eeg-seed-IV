"""
Quick test to verify CNN medical imports work correctly
"""
from pathlib import Path
import importlib.util
import sys

def test_imports():
    print("🧪 Testing Medical CNN imports...")
    
    # Test 1: Sequential Feature Selection import
    sfs_path = Path("../sequential_feature_selection/clean_eeg_classifier.py").resolve()
    if sfs_path.exists():
        spec = importlib.util.spec_from_file_location("clean_eeg_classifier", sfs_path)
        clean_eeg_classifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clean_eeg_classifier)
        print(f"✅ Sequential Feature Selection import: SUCCESS")
        print(f"   Functions available: {[func for func in dir(clean_eeg_classifier) if not func.startswith('_')][:5]}...")
    else:
        print(f"❌ Sequential Feature Selection import: FAILED - path not found")
        return False
    
    # Test 2: Saved models directory
    saved_models_path = Path("../../saved_models").resolve()
    if saved_models_path.exists():
        saved_files = list(saved_models_path.glob("*.npy"))
        print(f"✅ Saved models directory: SUCCESS - {len(saved_files)} .npy files found")
    else:
        print(f"⚠️  Saved models directory: NOT FOUND - Sequential Feature Selection needs to run first")
    
    # Test 3: CSV data directory
    csv_path = Path("../../csv").resolve()
    if csv_path.exists():
        csv_subdirs = [d for d in csv_path.iterdir() if d.is_dir()]
        print(f"✅ CSV data directory: SUCCESS - {len(csv_subdirs)} session directories found")
    else:
        print(f"❌ CSV data directory: FAILED - SEED-IV data not found")
        return False
    
    # Test 4: TensorFlow availability
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: SUCCESS - version {tf.__version__}")
    except ImportError:
        print(f"⚠️  TensorFlow: NOT INSTALLED - run 'pip install tensorflow' for medical CNN")
    
    print("\n🏥 Medical CNN is ready to run!")
    return True

if __name__ == "__main__":
    test_imports()
