"""
EEG Emotion Classification - Choose Your Approach
=================================================

This gives you 3 clean options to solve the accuracy problem:

1. CLEAN SVM + SEQUENTIAL FEATURE SELECTION (Recommended for understanding)
   - Compares de_LDS vs de_movingAve features  
   - Finds optimal feature set using sequential selection
   - Uses best SVM classifier for your data
   - Target: 70%+ accuracy

2. SIMPLE CNN (Recommended for deployment)
   - Uses 2D EEG spatial maps
   - Fast training and deployment
   - Target: 75%+ accuracy

3. FEATURE COMPARISON ONLY (Quick diagnostic)
   - Just compares feature stability
   - Helps choose best feature type

Usage: python choose_approach.py
"""

import sys
from pathlib import Path

def run_approach_1():
    """Clean SVM + Sequential Feature Selection"""
    print("🎯 Running Clean SVM + Sequential Feature Selection...")
    try:
        import clean_eeg_classifier
        return clean_eeg_classifier.run_clean_eeg_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def run_approach_2():
    """Simple CNN for deployment"""
    print("🧠 Running Simple CNN Classifier...")
    try:
        import simple_cnn_classifier
        return simple_cnn_classifier.run_simple_cnn_classification()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 If TensorFlow error, install with: pip install tensorflow")
        return None

def run_approach_3():
    """Feature comparison only"""
    print("🔍 Running Feature Comparison...")
    try:
        import clean_eeg_classifier
        return clean_eeg_classifier.compare_feature_types(
            csv_dir="d:/eeg-python-code/eeg-seed-IV/csv", 
            max_subjects=10
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🚀 EEG EMOTION CLASSIFICATION - CHOOSE YOUR APPROACH")
    print("=" * 60)
    print()
    print("Available approaches:")
    print("1. 🎯 CLEAN SVM + SEQUENTIAL FEATURE SELECTION")
    print("   • Compares de_LDS vs de_movingAve stability")
    print("   • Advanced sequential feature selection")
    print("   • Best classifier selection (SVM/RandomForest)")
    print("   • Target: 70%+ accuracy")
    print("   • Good for: Understanding your data")
    print()
    print("2. 🧠 SIMPLE CNN CLASSIFIER")
    print("   • 2D EEG spatial maps (62×5)")
    print("   • Lightweight CNN architecture")
    print("   • Ready for deployment")
    print("   • Target: 75%+ accuracy")
    print("   • Good for: Production deployment")
    print()
    print("3. 🔍 FEATURE COMPARISON ONLY")
    print("   • Quick de_LDS vs de_movingAve comparison")
    print("   • Stability analysis")
    print("   • Fast diagnostic")
    print("   • Good for: Quick data check")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1, 2, 3, or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                return
            
            choice = int(choice)
            if choice not in [1, 2, 3]:
                print("⚠️ Please enter 1, 2, 3, or 'q'")
                continue
            
            break
        except ValueError:
            print("⚠️ Please enter a valid number (1, 2, 3) or 'q'")
    
    print(f"\n{'='*60}")
    
    if choice == 1:
        print("🎯 APPROACH 1: CLEAN SVM + SEQUENTIAL FEATURE SELECTION")
        print("=" * 60)
        results = run_approach_1()
        
        if results:
            print(f"\n🏆 FINAL RESULTS:")
            print(f"   Best feature type: {results.get('feature_type', 'N/A')}")
            print(f"   Selected features: {results.get('best_features', 'N/A')}")
            print(f"   CV accuracy: {results.get('cv_accuracy', 0):.3f}")
            print(f"   Test accuracy: {results.get('test_accuracy', 0):.3f}")
            
            if results.get('test_accuracy', 0) >= 0.7:
                print("✅ SUCCESS! Target accuracy achieved!")
            else:
                print("🔄 Consider trying CNN approach for better results")
    
    elif choice == 2:
        print("🧠 APPROACH 2: SIMPLE CNN CLASSIFIER")
        print("=" * 40)
        results = run_approach_2()
        
        if results:
            print(f"\n🏆 CNN RESULTS:")
            print(f"   Test accuracy: {results.get('test_accuracy', 0):.3f}")
            print(f"   Model saved: eeg_emotion_cnn.h5")
            print(f"   Preprocessing: cnn_preprocessing.json")
            
            if results.get('test_accuracy', 0) >= 0.75:
                print("✅ EXCELLENT! CNN target achieved!")
            elif results.get('test_accuracy', 0) >= 0.7:
                print("✅ GOOD! Acceptable accuracy achieved!")
            else:
                print("🔄 Consider data preprocessing improvements")
    
    elif choice == 3:
        print("🔍 APPROACH 3: FEATURE COMPARISON")
        print("=" * 35)
        best_feature = run_approach_3()
        
        if best_feature:
            print(f"\n🏆 RECOMMENDATION:")
            print(f"   Use feature type: {best_feature}")
            print(f"   Run Approach 1 or 2 with this feature type")

if __name__ == "__main__":
    main()
