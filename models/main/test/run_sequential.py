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
    
    # STEP 1: Load your EEG data with balanced sampling
    print("\n📂 STEP 1: Loading EEG data from CSV folder...")
    X, y = load_eeg_data(
        "../../../csv",  # Updated for Colab
        max_subjects=10,      # Use 10 subjects per session (instead of 15)
        trials_per_emotion=2  # Use 2 trials per emotion (balanced sampling)
    )
    
    if X is None:
        print("❌ Could not load data. Trying alternative path...")
        X, y = load_eeg_data(
            "csv", 
            max_subjects=10, 
            trials_per_emotion=2
        )
        
    if X is None:
        print("❌ Could not find CSV data. Please check your folder structure.")
        print("Expected structure: csv/1/1/de_LDS1.csv, csv/1/1/de_LDS2.csv, etc.")
        return None
    
    # STEP 1.5: Pre-filter features to make it manageable
    print(f"\n🔧 STEP 1.5: Pre-filtering features...")
    print(f"Original features: {X.shape[1]} (too many for sequential selection)")
    
    # Clean data: handle NaN values
    print("🧹 Cleaning data: handling missing values...")
    print(f"NaN values found: {X.isnull().sum().sum()}")
    
    # Fill NaN values with column means
    X_clean = X.fillna(X.mean())
    
    # Remove any remaining infinite values
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(0)
    
    print(f"✅ Data cleaned. NaN values after cleaning: {X_clean.isnull().sum().sum()}")
    
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Reduce from 17,050 to 200 most informative features
    print("🔍 Selecting top 200 features using F-test...")
    selector = SelectKBest(f_classif, k=200)
    X_filtered = selector.fit_transform(X_clean, y)  # Use cleaned data
    
    # Convert back to DataFrame with proper column names
    X_filtered = pd.DataFrame(X_filtered, columns=[f'filtered_feature_{i}' for i in range(200)])
    
    print(f"✅ Reduced to {X_filtered.shape[1]} features")
    print(f"📊 New dataset shape: {X_filtered.shape}")
    
    # STEP 2: Run feature selection on pre-filtered data
    print("\n🔍 STEP 2: Running sequential feature selection...")
    print("This will test different numbers of features to find the optimal set.")
    
    results = run_sequential_feature_selection(
        X_filtered, y,  # Use pre-filtered data instead of X
        max_features=25,  # Optimal: 25 features (sweet spot)
        step=5           # Test 5, 10, 15, 20, 25 features
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
        'total_features_tested': X_filtered.shape[1],  # Use filtered shape
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
    print(f"   💡 Selected {len(best_indices)} out of {X_filtered.shape[1]} features")
    
    return results

if __name__ == "__main__":
    results = run_feature_selection()
