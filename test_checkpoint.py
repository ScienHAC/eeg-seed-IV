#!/usr/bin/env python3
"""
Test Checkpoint Functionality
=============================

Quick test script to verify checkpoint save/resume functionality
"""

import sys
from pathlib import Path

# Add the comprehensive emotion recognition to path
sys.path.insert(0, str(Path(__file__).parent / "comprehensive_emotion_recognition"))

from main import ComprehensiveEmotionRecognition

def test_checkpoints():
    """Test checkpoint functionality"""
    print("🧪 TESTING CHECKPOINT FUNCTIONALITY")
    print("=" * 50)
    
    # Initialize system
    system = ComprehensiveEmotionRecognition()
    
    # Show initial checkpoint status
    print("\n1. Initial checkpoint status:")
    system.show_checkpoint_status()
    
    # Run only Stage 1 first
    print("\n2. Running Stage 1 only...")
    results = system.run_all_stages(stages=[1], force_run=False)
    
    if results and 1 in results and 'error' not in results[1]:
        print("✅ Stage 1 completed successfully!")
        
        # Show checkpoint status after Stage 1
        print("\n3. Checkpoint status after Stage 1:")
        system.show_checkpoint_status()
        
        # Now run all stages - Stage 1 should be resumed from checkpoint
        print("\n4. Running all stages (Stage 1 should resume from checkpoint)...")
        results = system.run_all_stages(stages=[1, 2], force_run=False)
        
        if results:
            print("✅ All stages completed!")
            print("\n5. Final checkpoint status:")
            system.show_checkpoint_status()
            
            # Test clearing checkpoints
            print("\n6. Testing checkpoint clearing...")
            system.clear_checkpoints([1])
            system.show_checkpoint_status()
            
        else:
            print("❌ Failed to complete all stages")
    else:
        print("❌ Stage 1 failed")
        if results and 1 in results:
            print(f"Error: {results[1].get('error', 'Unknown error')}")

def quick_test():
    """Quick test without full training"""
    print("🚀 QUICK CHECKPOINT TEST")
    print("=" * 30)
    
    system = ComprehensiveEmotionRecognition()
    
    # Test checkpoint directory creation
    print(f"Checkpoint directory: {system.checkpoint_dir}")
    print(f"Directory exists: {system.checkpoint_dir.exists()}")
    
    # Test checkpoint status
    status = system.show_checkpoint_status()
    print(f"Found {sum(status.values())} existing checkpoints")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test checkpoint functionality')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        test_checkpoints()
