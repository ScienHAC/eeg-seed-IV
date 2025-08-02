#!/usr/bin/env python3
"""
Detailed exploration of MATLAB files to understand subject numbering
"""

from pathlib import Path

def detailed_matlab_exploration():
    """Get detailed view of MATLAB file structure"""
    matlab_dir = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth"
    matlab_path = Path(matlab_dir)
    
    print(f"🔍 Detailed MATLAB file exploration")
    print(f"📁 Directory: {matlab_dir}")
    
    for session in [1, 2, 3]:
        session_dir = matlab_path / str(session)
        print(f"\n📂 SESSION {session}:")
        
        if session_dir.exists():
            mat_files = list(session_dir.glob("*.mat"))
            print(f"   Found {len(mat_files)} .mat files")
            
            # Extract subject numbers from filenames
            subjects = []
            for mat_file in sorted(mat_files):
                # Extract subject number (assuming format: subject_date.mat)
                name_parts = mat_file.stem.split('_')
                if name_parts:
                    try:
                        subject_num = int(name_parts[0])
                        subjects.append(subject_num)
                    except ValueError:
                        pass
                
                print(f"      {mat_file.name}")
            
            if subjects:
                unique_subjects = sorted(set(subjects))
                print(f"   Subjects available: {unique_subjects}")
                
                # Check if we have subjects 13, 14, 15
                test_subjects = [13, 14, 15]
                available_test_subjects = [s for s in test_subjects if s in unique_subjects]
                missing_test_subjects = [s for s in test_subjects if s not in unique_subjects]
                
                print(f"   Test subjects available: {available_test_subjects}")
                if missing_test_subjects:
                    print(f"   Test subjects MISSING: {missing_test_subjects}")
                
                # Suggest alternative test subjects
                max_subject = max(unique_subjects) if unique_subjects else 0
                if max_subject < 13:
                    print(f"   ⚠️ Highest subject number is {max_subject}")
                    print(f"   💡 Suggested test subjects: {unique_subjects[-3:] if len(unique_subjects) >= 3 else unique_subjects}")
        else:
            print(f"   ❌ Session directory not found")

if __name__ == "__main__":
    detailed_matlab_exploration()
