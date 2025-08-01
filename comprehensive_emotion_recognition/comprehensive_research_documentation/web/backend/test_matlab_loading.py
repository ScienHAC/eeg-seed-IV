#!/usr/bin/env python3
"""
Test MATLAB Data Loading
========================

This script tests if the backend can properly load your MATLAB files
and extract real EEG data instead of falling back to random generation.

Run this to verify your MATLAB data is being loaded correctly.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from main import SeedIVMatLoader, SEED_IV_MAT_PATH, SEED_IV_BASE_PATH

def test_matlab_file_access():
    """Test if we can access MATLAB files"""
    print("🧠 Testing MATLAB File Access")
    print("=" * 50)
    
    print(f"MATLAB Path: {SEED_IV_MAT_PATH}")
    print(f"CSV Path: {SEED_IV_BASE_PATH}")
    print(f"MATLAB Path exists: {SEED_IV_MAT_PATH.exists()}")
    print(f"CSV Path exists: {SEED_IV_BASE_PATH.exists()}")
    
    if not SEED_IV_MAT_PATH.exists():
        print("❌ MATLAB path does not exist!")
        return False
    
    # Check specific session directories
    for session in [1, 2, 3]:
        session_dir = SEED_IV_MAT_PATH / str(session)
        print(f"\nSession {session} directory: {session_dir}")
        print(f"Session {session} exists: {session_dir.exists()}")
        
        if session_dir.exists():
            mat_files = list(session_dir.glob("*.mat"))
            print(f"MATLAB files in session {session}: {len(mat_files)}")
            for mat_file in mat_files[:3]:  # Show first 3
                print(f"  - {mat_file.name}")
    
    return True

def test_matlab_loading():
    """Test loading a specific MATLAB file"""
    print("\n🔬 Testing MATLAB File Loading")
    print("=" * 50)
    
    # Test with Subject 1, Session 1
    loader = SeedIVMatLoader(SEED_IV_MAT_PATH)
    
    try:
        # Find MATLAB file
        mat_file_path = loader.find_mat_file(subject=1, session=1)
        print(f"Found MATLAB file: {mat_file_path}")
        
        if not mat_file_path:
            print("❌ No MATLAB file found for Subject 1, Session 1")
            return False
        
        # Load MATLAB file
        if mat_file_path.suffix == '.mat':
            mat_data = loader.load_mat_file(mat_file_path)
            if mat_data:
                print(f"✅ Successfully loaded MATLAB file")
                print(f"Available features: {list(mat_data['features'].keys())[:10]}...")  # Show first 10
                
                # Test feature extraction
                feature_data = loader.extract_feature_data(mat_data, 'de_LDS', 1)
                if feature_data is not None:
                    print(f"✅ Successfully extracted de_LDS1 features")
                    print(f"Feature data shape: {feature_data.shape}")
                    print(f"Feature data type: {feature_data.dtype}")
                    print(f"Feature data range: [{feature_data.min():.3f}, {feature_data.max():.3f}]")
                    print(f"Feature data mean: {feature_data.mean():.3f}")
                    
                    # Test frequency band extraction
                    band_data = loader.get_frequency_band_data(feature_data, 'delta')
                    print(f"✅ Successfully extracted delta band data")
                    print(f"Delta band shape: {band_data.shape}")
                    print(f"Delta band range: [{band_data.min():.3f}, {band_data.max():.3f}]")
                    
                    return True
                else:
                    print("❌ Failed to extract feature data")
                    return False
            else:
                print("❌ Failed to load MATLAB file")
                return False
        else:
            print("❌ Not a MATLAB file")
            return False
            
    except Exception as e:
        print(f"❌ Error during MATLAB loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_raw_matlab_loading():
    """Test direct MATLAB file loading with scipy"""
    print("\n🧪 Testing Raw MATLAB File Loading")
    print("=" * 50)
    
    # Try to load Subject 1, Session 1 directly
    session_dir = SEED_IV_MAT_PATH / "1"
    if not session_dir.exists():
        print(f"❌ Session directory does not exist: {session_dir}")
        return False
    
    # Find Subject 1 file
    subject_files = list(session_dir.glob("1_*.mat"))
    if not subject_files:
        print("❌ No Subject 1 MATLAB file found")
        return False
    
    mat_file = subject_files[0]
    print(f"Testing file: {mat_file}")
    
    try:
        # Load with scipy
        mat_data = loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
        
        # Remove MATLAB metadata
        features = {key: value for key, value in mat_data.items() 
                   if not key.startswith('__')}
        
        print(f"✅ Successfully loaded with scipy")
        print(f"Available keys: {list(features.keys())[:10]}...")  # Show first 10
        
        # Look for de_LDS features
        de_features = [key for key in features.keys() if key.startswith('de_LDS')]
        print(f"DE_LDS features found: {len(de_features)}")
        
        if de_features:
            # Test first de_LDS feature
            feature_key = de_features[0]
            feature_data = features[feature_key]
            print(f"Testing feature: {feature_key}")
            print(f"Feature data type: {type(feature_data)}")
            print(f"Feature data shape: {feature_data.shape if hasattr(feature_data, 'shape') else 'No shape'}")
            
            if hasattr(feature_data, 'shape') and len(feature_data.shape) > 0:
                print(f"Feature data range: [{np.min(feature_data):.3f}, {np.max(feature_data):.3f}]")
                print(f"Feature data mean: {np.mean(feature_data):.3f}")
                print("✅ Raw MATLAB data looks good!")
                return True
            else:
                print("❌ Feature data has no shape or is empty")
                return False
        else:
            print("❌ No de_LDS features found")
            return False
            
    except Exception as e:
        print(f"❌ Error loading raw MATLAB file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 MATLAB Data Loading Test Suite")
    print("=" * 60)
    print("This tests if your website backend can load ACTUAL MATLAB data")
    print("instead of falling back to random number generation.")
    print()
    
    success = True
    
    # Test 1: File access
    if not test_matlab_file_access():
        success = False
    
    # Test 2: MATLAB loading through backend
    if not test_matlab_loading():
        success = False
    
    # Test 3: Raw MATLAB loading
    if not test_raw_matlab_loading():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Your backend should now load REAL MATLAB data instead of random numbers!")
        print("The website charts should now be consistent across reloads.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("The backend may still fall back to random data generation.")
        print("Check the error messages above to fix the issues.")
    print("=" * 60)

if __name__ == "__main__":
    main()
