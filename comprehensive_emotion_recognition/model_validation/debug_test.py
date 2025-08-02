"""
Simple debug test for validation system
"""

print("=== VALIDATION SYSTEM DEBUG ===")

print("Step 1: Testing config import...")
try:
    from config import ValidationConfig
    print("✅ Config import OK")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    exit(1)

print("Step 2: Creating config instance...")
try:
    config = ValidationConfig()
    print("✅ Config instance created")
    print(f"   Data dir: {config.data_dir}")
    print(f"   MATLAB dir: {getattr(config, 'matlab_data_dir', 'Not set')}")
    print(f"   Stage 1 checkpoint: {config.stage1_checkpoint_path}")
    print(f"   Stage 2 checkpoint: {config.stage2_checkpoint_path}")
except Exception as e:
    print(f"❌ Config creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("Step 3: Testing data loader import...")
try:
    from data_loader import UnseenDataLoader
    print("✅ Data loader import OK")
except Exception as e:
    print(f"❌ Data loader import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("Step 4: Creating data loader instance...")
try:
    data_loader = UnseenDataLoader(config)
    print("✅ Data loader instance created")
except Exception as e:
    print(f"❌ Data loader creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("✅ All components working!")
print("=== DEBUG COMPLETE ===")
