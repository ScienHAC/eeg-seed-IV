#!/usr/bin/env python3
"""
Explore MATLAB directory to understand file structure
"""

from pathlib import Path
import os

def explore_matlab_directory():
    """Explore the MATLAB directory to understand file naming"""
    matlab_dir = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth"
    
    print(f"🔍 Exploring MATLAB directory: {matlab_dir}")
    
    matlab_path = Path(matlab_dir)
    
    if not matlab_path.exists():
        print(f"❌ Directory not found: {matlab_path}")
        return
    
    print(f"✅ Directory exists!")
    
    # List all files
    try:
        all_files = list(matlab_path.iterdir())
        print(f"📁 Found {len(all_files)} items")
        
        # Filter .mat files
        mat_files = [f for f in all_files if f.suffix.lower() == '.mat']
        print(f"📄 Found {len(mat_files)} .mat files")
        
        # Show first 20 .mat files
        print(f"\n📋 First 20 .mat files:")
        for i, mat_file in enumerate(mat_files[:20]):
            print(f"   {i+1:2d}. {mat_file.name}")
        
        if len(mat_files) > 20:
            print(f"   ... and {len(mat_files) - 20} more files")
        
        # Look for files containing subjects 13, 14, 15
        print(f"\n🎯 Files for test subjects [13, 14, 15]:")
        test_files = []
        for subject in [13, 14, 15]:
            subject_files = [f for f in mat_files if str(subject) in f.name]
            print(f"   Subject {subject}: {len(subject_files)} files")
            for sf in subject_files[:5]:  # Show first 5
                print(f"      {sf.name}")
                test_files.append(sf)
            if len(subject_files) > 5:
                print(f"      ... and {len(subject_files) - 5} more")
        
        # Show directory structure if there are subdirectories
        subdirs = [f for f in all_files if f.is_dir()]
        if subdirs:
            print(f"\n📁 Subdirectories found:")
            for subdir in subdirs[:10]:
                print(f"   {subdir.name}/")
                # Check for .mat files in subdirectories
                try:
                    subdir_mats = list(subdir.glob("*.mat"))
                    if subdir_mats:
                        print(f"      Contains {len(subdir_mats)} .mat files")
                        for sm in subdir_mats[:3]:
                            print(f"         {sm.name}")
                except:
                    pass
    
    except Exception as e:
        print(f"❌ Error exploring directory: {e}")

if __name__ == "__main__":
    explore_matlab_directory()
