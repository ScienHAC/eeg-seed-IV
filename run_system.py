#!/usr/bin/env python3
"""
SEED-IV System Runner - Clean execution script
"""

import os
import sys
from pathlib import Path

def run_seed_iv_system():
    """Run the SEED-IV emotion recognition system"""
    
    # Change to the correct directory
    system_dir = Path(__file__).parent / "comprehensive_emotion_recognition"
    os.chdir(system_dir)
    
    print("🧠 SEED-IV Comprehensive Emotion Recognition System")
    print("=" * 60)
    print(f"📁 Working directory: {system_dir}")
    print(f"🐍 Python: {sys.executable}")
    print()
    
    try:
        # Import and run the main system
        from main import main
        
        print("🚀 Starting comprehensive analysis...")
        print("⏱️  Expected time: 15-25 minutes for stages 1-2")
        print("💾 Checkpoints will be saved automatically")
        print()
        
        # Run the system
        main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory")
        return False
        
    except Exception as e:
        print(f"❌ System error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_seed_iv_system()
    if success:
        print("\n✅ System completed successfully!")
    else:
        print("\n❌ System encountered errors")
