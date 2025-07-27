"""
EEG Emotion Recognition - Main Runner
=====================================

Organized project structure for SEED-IV emotion recognition:
1. Sequential Feature Selection (models/sequential_feature_selection/)
2. Medical-Grade CNN (models/cnn_medical/)
3. Interactive Reports (index.html)

Usage: python main.py
"""

import os
import sys
from pathlib import Path

def show_project_structure():
    """Display organized project structure"""
    print("🗂️  EEG EMOTION RECOGNITION - ORGANIZED PROJECT")
    print("=" * 60)
    print("""
📁 Project Structure:
├── csv/                           # SEED-IV Dataset
├── saved_models/                  # Your SFS checkpoints (backup)
├── models/
│   ├── sequential_feature_selection/
│   │   ├── clean_eeg_classifier.py    # Your SFS pipeline ✅
│   │   ├── choose_approach.py         # Menu system
│   │   └── simple_cnn_classifier.py   # Backup CNN
│   ├── cnn_medical/
│   │   ├── cnn_medical_grade.py       # Medical CNN 🏥
│   │   ├── improved_cnn_fixed.py      # Improved CNN
│   │   └── advanced_ensemble.py       # Advanced Ensemble ⚡
│   └── home/
│       └── eeg_emotion_classifier.py  # Additional backup
├── reports/
│   ├── PROJECT_SUMMARY.md            # Full project status
│   └── README_MEDICAL_CNN.md         # CNN instructions
├── index.html                        # Interactive Report 📊
└── main.py                          # This runner
""")

def run_sequential_feature_selection():
    """Run your Sequential Feature Selection pipeline"""
    print("🎯 Running Sequential Feature Selection...")
    
    try:
        # Import with proper error handling
        import importlib.util
        import sys
        
        # Load the module dynamically but safely
        sfs_path = Path("models/sequential_feature_selection/clean_eeg_classifier.py").resolve()
        if not sfs_path.exists():
            print("❌ Sequential Feature Selection module not found")
            return
            
        spec = importlib.util.spec_from_file_location("clean_eeg_classifier", sfs_path)
        clean_eeg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clean_eeg_module)
        
        # Run the analysis
        results = clean_eeg_module.run_clean_eeg_analysis()
        if results:
            print("✅ Sequential Feature Selection completed!")
            print(f"📊 Results: {results}")
        else:
            print("❌ Sequential Feature Selection failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the CSV data is in the correct location")

def run_advanced_ensemble():
    """Run Advanced Ensemble that should beat your SFS results"""
    print("🚀 Running Advanced Ensemble (Proper Small Dataset Approach)...")
    
    try:
        # Import the advanced ensemble module dynamically
        import importlib.util
        
        ensemble_path = Path("models/cnn_medical/advanced_ensemble.py").resolve()
        if not ensemble_path.exists():
            print("❌ Advanced Ensemble module not found")
            return
            
        spec = importlib.util.spec_from_file_location("advanced_ensemble", ensemble_path)
        ensemble_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ensemble_module)
        
        # Run the advanced ensemble pipeline
        results = ensemble_module.run_advanced_ensemble_pipeline()
        if results:
            print("✅ Advanced Ensemble completed!")
            print(f"📊 Results: {results}")
            if results.get('beats_sfs', False):
                print("🎉 SUCCESS: Advanced Ensemble beats your SFS baseline!")
                print(f"🎯 Best approach: {results.get('best_approach', 'Unknown')}")
            else:
                print("🔧 Advanced ML is better approach than CNN for small datasets")
        else:
            print("❌ Advanced Ensemble failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Advanced ensemble uses proper ML for small datasets (not CNN)")

def run_improved_cnn():
    """Run Improved CNN that should beat your SFS results"""
    print("🚀 Running Improved CNN (Fixed Version)...")
    
    try:
        # Check if TensorFlow is available first
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
        
        # Import the improved CNN module dynamically
        import importlib.util
        
        cnn_path = Path("models/cnn_medical/improved_cnn_fixed.py").resolve()
        if not cnn_path.exists():
            print("❌ Improved CNN module not found")
            return
            
        spec = importlib.util.spec_from_file_location("improved_cnn_fixed", cnn_path)
        improved_cnn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(improved_cnn_module)
        
        # Run the improved CNN pipeline
        results = improved_cnn_module.run_improved_cnn_pipeline()
        if results:
            print("✅ Improved CNN completed!")
            print(f"📊 Results: {results}")
            if results.get('beats_sfs', False):
                print("🎉 SUCCESS: Improved CNN beats your SFS baseline!")
            else:
                print("🔧 Still needs work, but better approach than original CNN")
        else:
            print("❌ Improved CNN failed")
            
    except ImportError as e:
        if "tensorflow" in str(e).lower():
            print("❌ TensorFlow not installed")
            print("💡 Install with: pip install tensorflow")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_medical_cnn():
    """Run Medical-Grade CNN pipeline"""
    print("🏥 Running Medical-Grade CNN...")
    
    try:
        # Check if TensorFlow is available first
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
        
        # Import the medical CNN module dynamically
        import importlib.util
        
        cnn_path = Path("models/cnn_medical/cnn_medical_grade.py").resolve()
        if not cnn_path.exists():
            print("❌ Medical CNN module not found")
            return
            
        spec = importlib.util.spec_from_file_location("cnn_medical_grade", cnn_path)
        cnn_medical_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cnn_medical_module)
        
        # Run the medical grade pipeline
        results = cnn_medical_module.run_medical_grade_pipeline()
        if results:
            print("✅ Medical-Grade CNN completed!")
            print(f"📊 Results: {results}")
        else:
            print("❌ Medical-Grade CNN failed")
            
    except ImportError as e:
        if "tensorflow" in str(e).lower():
            print("❌ TensorFlow not installed")
            print("💡 Install with: pip install tensorflow")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def open_interactive_report():
    """Open the interactive HTML report"""
    print("📊 Opening interactive report...")
    import webbrowser
    
    html_path = Path("index.html").absolute()
    if html_path.exists():
        webbrowser.open(f"file://{html_path}")
        print("✅ Interactive report opened in browser")
    else:
        print("❌ index.html not found")

def show_current_results():
    """Show current project results"""
    print("📊 CURRENT PROJECT RESULTS")
    print("=" * 40)
    
    # Check Sequential Feature Selection results
    if Path("clean_eeg_results.csv").exists():
        import pandas as pd
        sfs_results = pd.read_csv("clean_eeg_results.csv")
        print("✅ Sequential Feature Selection Results:")
        print(f"   Feature type: {sfs_results['feature_type'].iloc[0]}")
        print(f"   Selected features: {sfs_results['best_features'].iloc[0]}")
        print(f"   Test accuracy: {sfs_results['test_accuracy'].iloc[0]:.3f}")
        print(f"   Total samples: {sfs_results['total_samples'].iloc[0]}")
    else:
        print("❌ Sequential Feature Selection not completed")
    
    # Check Medical CNN results
    cnn_results_path = Path("models/cnn_medical/medical_eeg_cnn_results.csv")
    if cnn_results_path.exists():
        import pandas as pd
        cnn_results = pd.read_csv(cnn_results_path)
        print("\n✅ Medical-Grade CNN Results (Original):")
        print(f"   CV accuracy: {cnn_results['cv_accuracy'].iloc[0]:.3f}")
        print(f"   Medical grade: {cnn_results['medical_grade'].iloc[0]}")
        print(f"   Improvement: {cnn_results['improvement_over_svm'].iloc[0]:+.1f}%")
        print(f"   Status: ❌ WORSE than SFS baseline")
    else:
        print("\n❌ Medical-Grade CNN (Original) not completed")
    
    # Check Improved CNN results
    improved_cnn_results_path = Path("improved_cnn_results.csv")
    if improved_cnn_results_path.exists():
        import pandas as pd
        improved_results = pd.read_csv(improved_cnn_results_path)
        print("\n✅ Improved CNN Results:")
        print(f"   CV accuracy: {improved_results['cv_accuracy'].iloc[0]:.3f}")
        print(f"   Beats SFS: {improved_results['beats_sfs'].iloc[0]}")
        print(f"   Improvement vs SFS: {improved_results['improvement_over_sfs'].iloc[0]:+.1f}%")
        print(f"   Status: {'✅ SUCCESS' if improved_results['beats_sfs'].iloc[0] else '🔧 NEEDS WORK'}")
    else:
        print("\n❌ Improved CNN not completed")
    
    # Check Advanced Ensemble results
    ensemble_results_path = Path("intensive_ensemble_results.csv")
    if ensemble_results_path.exists():
        import pandas as pd
        ensemble_results = pd.read_csv(ensemble_results_path)
        print("\n✅ Advanced Ensemble Results:")
        print(f"   Best approach: {ensemble_results['best_approach'].iloc[0]}")
        print(f"   Best accuracy: {ensemble_results['best_accuracy'].iloc[0]:.3f}")
        print(f"   Beats SFS: {ensemble_results['beats_sfs'].iloc[0]}")
        print(f"   Improvement vs SFS: {ensemble_results['improvement_over_sfs'].iloc[0]:+.1f}%")
        print(f"   Status: {'✅ SUCCESS' if ensemble_results['beats_sfs'].iloc[0] else '🔧 NEEDS MORE FEATURES'}")
    else:
        print("\n❌ Advanced Ensemble not completed")
    
    # Check saved models
    if Path("saved_models").exists():
        saved_files = list(Path("saved_models").glob("*.npy"))
        print(f"\n💾 Saved checkpoints: {len(saved_files)} feature sets")
    else:
        print("\n❌ No saved checkpoints found")

def main_menu():
    """Main interactive menu"""
    while True:
        print("\n🧠 EEG EMOTION RECOGNITION - MAIN MENU")
        print("=" * 50)
        print("1. 📊 Show Project Structure")
        print("2. 🎯 Run Sequential Feature Selection")
        print("3. 🏥 Run Medical-Grade CNN (Original)")
        print("4. 🚀 Run Improved CNN (Fixed)")
        print("5. ⚡ Run Advanced Ensemble (BEST)")
        print("6. 📈 Show Current Results")
        print("7. 🌐 Open Interactive Report")
        print("8. ❌ Exit")
        
        choice = input("\nChoose option (1-8): ").strip()
        
        if choice == '1':
            show_project_structure()
        elif choice == '2':
            run_sequential_feature_selection()
        elif choice == '3':
            run_medical_cnn()
        elif choice == '4':
            run_improved_cnn()
        elif choice == '5':
            run_advanced_ensemble()
        elif choice == '6':
            show_current_results()
        elif choice == '7':
            open_interactive_report()
        elif choice == '8':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🚀 Starting EEG Emotion Recognition System...")
    main_menu()
