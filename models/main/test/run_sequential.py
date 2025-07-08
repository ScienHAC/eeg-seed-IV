"""
Simple Runner for EEG Sequential Feature Selection
=================================================

This script loads your SEED-IV EEG data and runs sequential feature selection.
Just run this file and it will automatically find the best features.

Usage: python run_sequential.py
"""

import sys
from pathlib import Path

# Add the current directory to path so we can import sequential_clean
sys.path.append(str(Path(__file__).parent))

from sequential_clean import load_eeg_data, run_sequential_feature_selection, plot_results
import numpy as np
import pandas as pd

def run_feature_selection():
    """
    Main function to run feature selection on your EEG data
    """
    print("🚀 SEED-IV EEG Feature Selection")
    print("=" * 50)
    
    # STEP 1: Load your EEG data
    print("\n📂 STEP 1: Loading EEG data from CSV folder...")
    X, y = load_eeg_data("../../../csv")  # Adjust path to your csv folder
    
    if X is None:
        print("❌ Could not load data. Trying different path...")
        X, y = load_eeg_data("csv")  # Try current directory
        
    if X is None:
        print("❌ Could not find CSV data. Please check your folder structure.")
        print("Expected structure: csv/1/1/de_LDS1.csv, csv/1/1/de_LDS2.csv, etc.")
        return None
    
    # STEP 2: Run feature selection  
    print("\n🔍 STEP 2: Running sequential feature selection...")
    print("This will test different numbers of features to find the optimal set.")
    
    results = run_sequential_feature_selection(
        X, y, 
        max_features=50,  # Test up to 50 features
        step=5            # Test 5, 10, 15, 20, ... features
    )
    
    # STEP 3: Show results
    print("\n📊 STEP 3: Visualizing results...")
    plot_results(results)
    
    # STEP 4: Save best features
    print("\n💾 STEP 4: Saving results...")
    
    # Save best feature indices
    best_indices = np.where(results['best_features'])[0]
    
    # Create summary
    summary = {
        'best_accuracy': results['best_accuracy'],
        'best_feature_count': results['best_count'],
        'total_features_tested': X.shape[1],
        'improvement_over_baseline': results['best_accuracy'] - results['accuracies'][0],
        'best_feature_indices': best_indices.tolist()
    }
    
    # Save to files
    pd.DataFrame({'feature_index': best_indices}).to_csv('best_features.csv', index=False)
    
    with open('feature_selection_summary.txt', 'w') as f:
        f.write("SEED-IV EEG Sequential Feature Selection Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best accuracy: {summary['best_accuracy']:.4f}\n")
        f.write(f"Best feature count: {summary['best_feature_count']}\n")
        f.write(f"Total features available: {summary['total_features_tested']}\n")
        f.write(f"Improvement: {summary['improvement_over_baseline']:.4f}\n\n")
        f.write(f"Best feature indices:\n{best_indices}\n")
    
    print(f"✅ Results saved:")
    print(f"   📄 best_features.csv - List of best feature indices")
    print(f"   📄 feature_selection_summary.txt - Summary report")
    
    # Print final summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   🏆 Best accuracy: {results['best_accuracy']:.4f}")
    print(f"   📊 Optimal features: {results['best_count']}")
    print(f"   📈 Improvement: +{summary['improvement_over_baseline']:.4f}")
    print(f"   💡 Selected {len(best_indices)} out of {X.shape[1]} features")
    
    return results

if __name__ == "__main__":
    results = run_feature_selection()
