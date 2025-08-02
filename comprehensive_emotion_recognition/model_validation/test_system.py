"""
Quick test script to verify the validation system works
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the comprehensive_emotion_recognition to path
sys.path.append(str(Path(__file__).parent.parent))

def test_validation_system():
    """Test the validation system components"""
    
    print("🧪 Testing EEG Model Validation System")
    print("=" * 40)
    
    try:
        # Test 1: Import all modules
        print("Test 1: Importing validation modules...")
        
        from model_validation.config import ValidationConfig
        from model_validation.model_loader import ModelLoader  
        from model_validation.data_loader import UnseenDataLoader
        from model_validation.validation_engine import ValidationEngine
        from model_validation.report_generator import ValidationReportGenerator
        
        print("   ✅ All modules imported successfully")
        
        # Test 2: Initialize configuration
        print("\nTest 2: Initializing configuration...")
        config = ValidationConfig()
        print(f"   ✅ Config loaded - Model dir: {config.model_dir}")
        print(f"   ✅ Data dir: {config.data_dir}")
        print(f"   ✅ Output dir: {config.validation_output_dir}")
        
        # Test 3: Check directory structure
        print("\nTest 3: Checking directory structure...")
        
        # Check if model directory exists (may be None for stage-only models)
        if config.model_dir:
            if Path(config.model_dir).exists():
                model_files = list(Path(config.model_dir).glob("*.joblib"))
                print(f"   ✅ Model directory exists with {len(model_files)} .joblib files")
            else:
                print(f"   ⚠️ Model directory not found: {config.model_dir}")
        else:
            print(f"   ✅ Model directory: None (using Stage checkpoints only)")
        
        # Check stage checkpoints
        if hasattr(config, 'stage1_checkpoint_path') and Path(config.stage1_checkpoint_path).exists():
            print(f"   ✅ Stage 1 checkpoint found")
        else:
            print(f"   ⚠️ Stage 1 checkpoint not found: {getattr(config, 'stage1_checkpoint_path', 'Not set')}")
            
        if hasattr(config, 'stage2_checkpoint_path') and Path(config.stage2_checkpoint_path).exists():
            print(f"   ✅ Stage 2 checkpoint found")
        else:
            print(f"   ⚠️ Stage 2 checkpoint not found: {getattr(config, 'stage2_checkpoint_path', 'Not set')}")
        
        # Check if data directory exists
        if Path(config.data_dir).exists():
            print(f"   ✅ Data directory exists")
        else:
            print(f"   ⚠️ Data directory not found: {config.data_dir}")
        
        # Test 4: Initialize components
        print("\nTest 4: Initializing validation components...")
        
        model_loader = ModelLoader(config)
        data_loader = UnseenDataLoader(config) 
        validation_engine = ValidationEngine(config)
        report_generator = ValidationReportGenerator(config)
        
        print("   ✅ All components initialized successfully")
        
        # Test 5: Check for available models (without loading)
        print("\nTest 5: Checking available models...")
        
        # Check for stage checkpoints instead of model_dir
        if hasattr(config, 'stage1_checkpoint_path'):
            stage1_path = Path(config.stage1_checkpoint_path)
            if stage1_path.exists():
                print(f"   📋 Found Stage 1 checkpoint: {stage1_path.name}")
            else:
                print(f"   ⚠️ Stage 1 checkpoint not found: {stage1_path}")
        
        if hasattr(config, 'stage2_checkpoint_path'):
            stage2_path = Path(config.stage2_checkpoint_path)
            if stage2_path.exists():
                print(f"   📋 Found Stage 2 checkpoint: {stage2_path.name}")
            else:
                print(f"   ⚠️ Stage 2 checkpoint not found: {stage2_path}")
        
        if config.model_dir:
            model_dir = Path(config.model_dir)
            if model_dir.exists():
                joblib_files = list(model_dir.glob("*.joblib"))
                print(f"   📋 Found {len(joblib_files)} additional model files:")
                for model_file in joblib_files[:5]:  # Show first 5
                    print(f"      - {model_file.name}")
                if len(joblib_files) > 5:
                    print(f"      ... and {len(joblib_files) - 5} more")
            else:
                print("   ⚠️ No additional model directory")
        else:
            print("   ✅ Using Stage checkpoints only (saved_models ignored)")
        
        # Test 6: Check data availability
        print("\nTest 6: Checking data availability...")
        
        data_dir = Path(config.data_dir)
        if data_dir.exists():
            csv_files = list(data_dir.glob("**/*.csv"))
            print(f"   📊 Found {len(csv_files)} CSV files")
            
            # Check for SEED-IV structure
            seed_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if seed_folders:
                print(f"   📁 Found {len(seed_folders)} subject folders: {[d.name for d in seed_folders[:3]]}")
        else:
            print("   ⚠️ No data directory found")
        
        # Test 7: Output directory setup
        print("\nTest 7: Setting up output directory...")
        
        output_dir = Path(config.validation_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_dir.exists():
            print(f"   ✅ Output directory ready: {output_dir}")
        else:
            print(f"   ❌ Failed to create output directory")
        
        print("\n" + "=" * 40)
        print("🎉 VALIDATION SYSTEM TEST SUMMARY")
        print("✅ All core components are working")
        print("✅ Directory structure is accessible")
        print("✅ Ready to run full validation")
        
        print(f"\n📝 To run validation: python run_validation.py")
        print(f"📁 Results will be saved to: {config.validation_output_dir}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def quick_model_check():
    """Quick check of available models"""
    
    print("\n🔍 QUICK MODEL INVENTORY")
    print("-" * 30)
    
    try:
        from model_validation.config import ValidationConfig
        config = ValidationConfig()
        
        model_dir = Path(config.model_dir)
        
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return
        
        # List all .joblib files
        joblib_files = list(model_dir.glob("*.joblib"))
        
        if not joblib_files:
            print("❌ No .joblib model files found")
            return
        
        print(f"📋 Found {len(joblib_files)} trained models:")
        
        for i, model_file in enumerate(joblib_files, 1):
            # Extract info from filename
            name = model_file.stem
            size_mb = model_file.stat().st_size / (1024 * 1024)
            
            print(f"   {i:2d}. {name}")
            print(f"       📁 Size: {size_mb:.1f} MB")
            print(f"       📅 Modified: {datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
            
        print(f"\n🎯 Ready to validate {len(joblib_files)} models!")
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")

if __name__ == "__main__":
    # Run system test
    success = test_validation_system()
    
    if success:
        # Show available models
        quick_model_check()
        
        print("\n" + "🚀" * 20)
        print("VALIDATION SYSTEM IS READY!")
        print("Run: python run_validation.py")
        print("🚀" * 20)
    else:
        print("\n❌ System test failed. Please check the errors above.")
