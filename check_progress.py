#!/usr/bin/env python3
"""
Quick progress checker for the SEED-IV system
"""

import os
import time
from pathlib import Path

def check_system_progress():
    """Check if the system is making progress"""
    
    log_file = Path("comprehensive_emotion_recognition/comprehensive_emotion_recognition.log")
    
    if not log_file.exists():
        print("❌ Log file not found")
        return
    
    # Get file size and modification time
    stat = log_file.stat()
    file_size = stat.st_size
    mod_time = stat.st_mtime
    
    print(f"📊 System Progress Check")
    print(f"=" * 50)
    print(f"Log file size: {file_size:,} bytes")
    print(f"Last modified: {time.ctime(mod_time)}")
    print(f"Time since last update: {time.time() - mod_time:.1f} seconds")
    
    # Read last few lines
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"\n📝 Last 5 log entries:")
    print("-" * 30)
    for line in lines[-5:]:
        print(line.strip())
    
    # Check for checkpoints
    checkpoint_dir = Path("comprehensive_emotion_recognition/csv_data/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.joblib"))
        print(f"\n💾 Checkpoints found: {len(checkpoints)}")
        for cp in checkpoints:
            print(f"  - {cp.name}")
    else:
        print("\n💾 No checkpoints directory found")
    
    # Estimate what stage we're in
    content = ''.join(lines[-20:]) if len(lines) >= 20 else ''.join(lines)
    
    if "Pipeline created with" in content:
        print("\n🔄 STATUS: Training in progress (Cross-validation phase)")
        print("⏱️  This can take 10-20 minutes for SVM on 37K samples")
        print("✅ System is working normally - please wait")
    elif "Stage 1 failed" in content:
        print("\n❌ STATUS: Error encountered")
    elif "Training completed" in content:
        print("\n✅ STATUS: Training completed")
    else:
        print("\n❓ STATUS: Unknown state")

if __name__ == "__main__":
    check_system_progress()
