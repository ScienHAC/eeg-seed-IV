#!/usr/bin/env python3
"""
Resume Training Script
======================

Utility script to resume training from checkpoints
"""

import sys
from pathlib import Path

# Add the comprehensive emotion recognition to path
sys.path.insert(0, str(Path(__file__).parent / "comprehensive_emotion_recognition"))

from main import ComprehensiveEmotionRecognition

def resume_training(force_run=False, clear_checkpoints=False):
    """
    Resume training from existing checkpoints
    
    Parameters:
    -----------
    force_run : bool
        Force rerun all stages from scratch
    clear_checkpoints : bool
        Clear all existing checkpoints before running
    """
    print("🔄 RESUME TRAINING FROM CHECKPOINTS")
    print("=" * 50)
    
    # Initialize system
    system = ComprehensiveEmotionRecognition()
    
    if clear_checkpoints:
        print("🗑️ Clearing all checkpoints...")
        system.clear_checkpoints()
        print()
    
    # Show checkpoint status
    print("📋 Current checkpoint status:")
    checkpoint_status = system.show_checkpoint_status()
    available_checkpoints = [stage for stage, has_cp in checkpoint_status.items() if has_cp]
    print()
    
    if available_checkpoints and not force_run:
        print(f"✅ Found checkpoints for stages: {available_checkpoints}")
        print("   → These stages will be resumed from checkpoints")
        print("   → Only missing stages will be trained from scratch")
    elif force_run:
        print("🔄 Force run enabled - all stages will be trained from scratch")
    else:
        print("ℹ️ No checkpoints found - all stages will be trained from scratch")
    
    print()
    print("⏰ Estimated completion times:")
    print("   → Stage 1 (SVM): ~5-8 minutes")
    print("   → Stage 2 (Random Forest): ~10-15 minutes")
    print("   → Total: ~15-25 minutes")
    print()
    
    # Ask for confirmation
    if not force_run and available_checkpoints:
        response = input("Continue with resume? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    print("🚀 Starting training...")
    print()
    
    # Run stages
    results = system.run_all_stages(stages=[1, 2], force_run=force_run)
    
    if results:
        successful_stages = [stage for stage, result in results.items() if 'error' not in result]
        failed_stages = [stage for stage, result in results.items() if 'error' in result]
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED!")
        print("=" * 60)
        
        if successful_stages:
            print(f"✅ Successful stages: {successful_stages}")
            for stage in successful_stages:
                result = results[stage]
                accuracy = result.get('accuracy', 0) * 100
                print(f"   Stage {stage}: {accuracy:.2f}% accuracy")
        
        if failed_stages:
            print(f"❌ Failed stages: {failed_stages}")
            for stage in failed_stages:
                result = results[stage]
                print(f"   Stage {stage}: {result.get('error', 'Unknown error')}")
        
        print(f"\n📁 Results saved to: {system.config.data.csv_output_dir}")
        print(f"💾 Checkpoints saved to: {system.checkpoint_dir}")
        print("\n💡 To resume interrupted training, run this script again")
        
    else:
        print("\n❌ Training failed!")
        print("Check the logs for error details")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume training from checkpoints')
    parser.add_argument('--force', action='store_true', 
                       help='Force rerun all stages from scratch')
    parser.add_argument('--clear', action='store_true',
                       help='Clear all checkpoints before running')
    
    args = parser.parse_args()
    
    resume_training(force_run=args.force, clear_checkpoints=args.clear)
