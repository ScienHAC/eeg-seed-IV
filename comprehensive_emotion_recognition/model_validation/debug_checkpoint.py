#!/usr/bin/env python3
"""
Debug checkpoint loading - fix the models import
"""

import sys
import os
from pathlib import Path

# Add the main project directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add comprehensive_emotion_recognition to path
comp_root = Path(__file__).parent.parent
sys.path.insert(0, str(comp_root))

print(f"Python path includes:")
for p in sys.path[:5]:
    print(f"  {p}")

try:
    import joblib
    import numpy as np
    
    # Load Stage 1 checkpoint with proper imports
    stage1_path = Path(__file__).parent.parent / "csv_data" / "checkpoints" / "stage_1_checkpoint.joblib"
    print(f"📋 Loading: {stage1_path}")
    
    if not stage1_path.exists():
        print(f"❌ Not found: {stage1_path}")
        sys.exit(1)
    
    # Try to create the missing models module reference
    class MockModels:
        pass
    
    # Add to sys.modules to prevent import error
    sys.modules['models'] = MockModels()
    
    # Now try loading
    checkpoint_data = joblib.load(stage1_path)
    print(f"✅ Checkpoint loaded successfully!")
    
    print(f"Checkpoint keys: {list(checkpoint_data.keys())}")
    
    if 'model' in checkpoint_data:
        model = checkpoint_data['model']
        print(f"Model type: {type(model)}")
        print(f"Model: {model}")
    
    if 'result' in checkpoint_data:
        result = checkpoint_data['result']
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
