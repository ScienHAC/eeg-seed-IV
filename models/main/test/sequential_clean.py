"""
Simple Sequential Feature Selection for EEG Emotion Recognition
==============================================================

Clean implementation for SEED-IV dataset feature selection.
Author: AI Assistant
Date: July 8, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_eeg_data(csv_dir="csv"):
    """
    Load EEG data from SEED-IV CSV files
    
    Returns:
    --------
    X : pd.DataFrame - Feature matrix
    y : np.array - Emotion labels (0=Neutral, 1=Sad, 2=Fear, 3=Happy)
    """
    print("🔄 Loading SEED-IV EEG dataset...")
    
    csv_path = Path(csv_dir)
    
    # SEED-IV emotion labels for each trial
    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }
    
    all_data = []
    file_count = 0
    
    # Load data from all sessions/subjects
    for session in range(1, 4):  # Sessions 1, 2, 3
        for subject in range(1, 16):  # Subjects 1-15
            session_path = csv_path / str(session) / str(subject)
            
            if not session_path.exists():
                continue
                
            print(f"📁 Loading Session {session}, Subject {subject}...")
            
            for trial in range(1, 25):  # Trials 1-24
                emotion_label = session_labels[session][trial-1]
                
                # Load LDS features (main differential entropy features)
                file_path = session_path / f"de_LDS{trial}.csv"
                
                if file_path.exists():
                    try:
                        data = pd.read_csv(file_path)
                        
                        # Flatten the EEG data into feature vector
                        features = data.values.flatten()
                        
                        # Create feature names
                        if file_count == 0:  # First file
                            feature_names = [f"feature_{i}" for i in range(len(features))]
                        
                        # Add metadata
                        row_data = {
                            'emotion': emotion_label,
                            'session': session,
                            'subject': subject,
                            'trial': trial
                        }
                        
                        # Add all features
                        for i, feature in enumerate(features):
                            row_data[f'feature_{i}'] = feature
                        
                        all_data.append(row_data)
                        file_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Error loading {file_path}: {e}")
    
    print(f"✅ Loaded {file_count} files successfully")
    
    if not all_data:
        print("❌ No data found! Check your CSV folder path.")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols]
    y = df['emotion'].values
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Emotion distribution: {np.bincount(y)}")
    print(f"   0=Neutral: {np.sum(y==0)}, 1=Sad: {np.sum(y==1)}, 2=Fear: {np.sum(y==2)}, 3=Happy: {np.sum(y==3)}")
    
    return X, y

def run_sequential_feature_selection(X, y, max_features=50, step=5):
    """
    Run sequential feature selection on EEG data
    
    Parameters:
    -----------
    X : DataFrame - Features
    y : array - Labels 
    max_features : int - Maximum number of features to test
    step : int - Step size for feature count
    
    Returns:
    --------
    dict - Results with best features and accuracies
    """
    print(f"\n🔍 Running Sequential Feature Selection...")
    print(f"   Testing up to {max_features} features in steps of {step}")
    
    # Standardize features
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Setup classifier and cross-validation
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test different numbers of features
    feature_counts = list(range(step, min(max_features + 1, X.shape[1] + 1), step))
    results = {
        'feature_counts': [],
        'accuracies': [],
        'std_deviations': [],
        'selected_features': []
    }
    
    best_accuracy = 0
    best_features = None
    best_count = 0
    
    for n_features in feature_counts:
        print(f"\n🧪 Testing {n_features} features...")
        
        # Create Sequential Feature Selector
        sfs = SequentialFeatureSelector(
            classifier, 
            n_features_to_select=n_features,
            direction='forward',
            scoring='accuracy',
            cv=5,
            n_jobs=-1
        )
        
        # Fit and get results
        sfs.fit(X_scaled, y)
        X_selected = sfs.transform(X_scaled)
        
        # Cross-validate performance
        cv_scores = cross_val_score(classifier, X_selected, y, cv=cv, scoring='accuracy')
        mean_acc = cv_scores.mean()
        std_acc = cv_scores.std()
        
        # Store results
        results['feature_counts'].append(n_features)
        results['accuracies'].append(mean_acc)
        results['std_deviations'].append(std_acc)
        results['selected_features'].append(sfs.get_support())
        
        print(f"   ✅ Accuracy: {mean_acc:.4f} (±{std_acc:.4f})")
        
        # Track best
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_features = sfs.get_support()
            best_count = n_features
    
    # Store best results
    results['best_accuracy'] = best_accuracy
    results['best_features'] = best_features
    results['best_count'] = best_count
    results['scaler'] = scaler
    
    print(f"\n🏆 BEST RESULTS:")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    print(f"   Best feature count: {best_count}")
    print(f"   Selected feature indices: {np.where(best_features)[0][:20]}...")
    
    return results

def plot_results(results):
    """Plot feature selection results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy vs features
    ax1.errorbar(results['feature_counts'], results['accuracies'], 
                yerr=results['std_deviations'],
                marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.axvline(x=results['best_count'], color='red', linestyle='--', 
                label=f'Best: {results["best_count"]} features')
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Cross-Validation Accuracy')
    ax1.set_title('Sequential Feature Selection Results')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot accuracy improvement
    accuracies = results['accuracies']
    accuracy_diff = np.diff([0] + accuracies)
    ax2.bar(results['feature_counts'], accuracy_diff, alpha=0.7,
            color=['green' if x > 0 else 'red' for x in accuracy_diff])
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Accuracy Change')
    ax2.set_title('Accuracy Change at Each Step')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run feature selection on SEED-IV data"""
    print("🚀 EEG Sequential Feature Selection for SEED-IV Dataset")
    print("=" * 60)
    
    # Load data
    X, y = load_eeg_data("csv")  # Change path if needed
    
    if X is None:
        print("❌ Failed to load data. Please check your CSV folder path.")
        return
    
    # Run feature selection
    results = run_sequential_feature_selection(X, y, max_features=100, step=10)
    
    # Plot results
    plot_results(results)
    
    # Save best features
    best_feature_indices = np.where(results['best_features'])[0]
    feature_df = pd.DataFrame({
        'feature_index': best_feature_indices,
        'feature_name': [f'feature_{i}' for i in best_feature_indices]
    })
    
    feature_df.to_csv('best_features.csv', index=False)
    print(f"\n💾 Saved {len(best_feature_indices)} best features to 'best_features.csv'")
    
    return results

if __name__ == "__main__":
    results = main()
