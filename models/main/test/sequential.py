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

def advanced_sequential_feature_selection(X, y, classifier=None, 
                                         feature_range=(5, 100, 5),
                                         cv_folds=5, 
                                         direction='forward',
                                         scoring='accuracy',
                                         verbose=True,
                                         plot_results=True):
    """
    Advanced Sequential Feature Selection with full transparency and control.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix with samples as rows and features as columns
    y : array-like
        Target labels
    classifier : sklearn classifier, optional
        Classifier to use for feature selection. If None, uses RandomForestClassifier
    feature_range : tuple
        (start, stop, step) for number of features to test
    cv_folds : int
        Number of cross-validation folds
    direction : str
        'forward' or 'backward' for sequential feature selection
    scoring : str
        Scoring metric for cross-validation
    verbose : bool
        Whether to print detailed information
    plot_results : bool
        Whether to plot accuracy vs number of features
        
    Returns:
    --------
    dict : Contains results including best features, accuracies, and selection history
    """
    
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Initialize results storage
    results = {
        'feature_counts': [],
        'accuracies': [],
        'std_deviations': [],
        'selected_features': [],
        'feature_names': [],
        'cv_scores_detail': [],
        'selection_history': []
    }
    
    # Get feature names
    if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_array = X
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Generate feature count range
    start, stop, step = feature_range
    feature_counts = list(range(start, min(stop + 1, X.shape[1] + 1), step))
    
    if verbose:
        print(f"=== Advanced Sequential Feature Selection ===")
        print(f"Dataset shape: {X.shape}")
        print(f"Classifier: {type(classifier).__name__}")
        print(f"Direction: {direction}")
        print(f"CV folds: {cv_folds}")
        print(f"Feature counts to test: {feature_counts}")
        print(f"Scoring: {scoring}")
        print("=" * 50)
    
    best_accuracy = 0
    best_features = None
    best_feature_count = 0
    
    for n_features in feature_counts:
        if verbose:
            print(f"\n🔍 Testing {n_features} features...")
        
        # Create Sequential Feature Selector
        sfs = SequentialFeatureSelector(
            classifier, 
            n_features_to_select=n_features,
            direction=direction,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1
        )
        
        # Fit the selector
        sfs.fit(X_array, y)
        
        # Get selected features
        selected_mask = sfs.get_support()
        selected_indices = np.where(selected_mask)[0]
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        # Transform data to selected features
        X_selected = sfs.transform(X_array)
        
        # Perform cross-validation on selected features
        cv_scores = cross_val_score(classifier, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
        mean_accuracy = cv_scores.mean()
        std_accuracy = cv_scores.std()
        
        # Store results
        results['feature_counts'].append(n_features)
        results['accuracies'].append(mean_accuracy)
        results['std_deviations'].append(std_accuracy)
        results['selected_features'].append(selected_indices.tolist())
        results['feature_names'].append(selected_feature_names)
        results['cv_scores_detail'].append(cv_scores.tolist())
        
        # Track best performance
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_features = selected_indices
            best_feature_count = n_features
        
        if verbose:
            print(f"   ✅ Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
            print(f"   📊 CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
            print(f"   🎯 Selected feature indices: {selected_indices.tolist()[:10]}{'...' if len(selected_indices) > 10 else ''}")
            if len(selected_feature_names) <= 10:
                print(f"   📝 Selected features: {selected_feature_names}")
            else:
                print(f"   📝 First 10 features: {selected_feature_names[:10]}...")
    
    # Store final results
    results['best_accuracy'] = best_accuracy
    results['best_features'] = best_features
    results['best_feature_count'] = best_feature_count
    results['best_feature_names'] = [feature_names[i] for i in best_features] if best_features is not None else []
    
    if verbose:
        print(f"\n🏆 BEST RESULTS:")
        print(f"   Best accuracy: {best_accuracy:.4f}")
        print(f"   Best feature count: {best_feature_count}")
        print(f"   Best feature indices: {best_features.tolist() if best_features is not None else []}")
    
    # Plot results
    if plot_results:
        plot_sfs_results(results, title=f"Sequential Feature Selection - {type(classifier).__name__}")
    
    return results

def plot_sfs_results(results, title="Sequential Feature Selection Results"):
    """
    Plot the results of sequential feature selection.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs Number of Features
    ax1 = axes[0, 0]
    feature_counts = results['feature_counts']
    accuracies = results['accuracies']
    std_devs = results['std_deviations']
    
    ax1.errorbar(feature_counts, accuracies, yerr=std_devs, 
                marker='o', linewidth=2, markersize=6, capsize=5)
    ax1.axvline(x=results['best_feature_count'], color='red', linestyle='--', 
                label=f'Best: {results["best_feature_count"]} features')
    ax1.axhline(y=results['best_accuracy'], color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Cross-Validation Accuracy')
    ax1.set_title('Accuracy vs Number of Features')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Accuracy Improvement
    ax2 = axes[0, 1]
    accuracy_diff = np.diff([0] + accuracies)
    ax2.bar(feature_counts, accuracy_diff, alpha=0.7, 
            color=['green' if x > 0 else 'red' for x in accuracy_diff])
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Accuracy Change')
    ax2.set_title('Accuracy Change at Each Step')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Plot 3: Standard Deviation
    ax3 = axes[1, 0]
    ax3.plot(feature_counts, std_devs, marker='s', linewidth=2, markersize=6, color='orange')
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Model Stability (Lower is Better)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Selection Heatmap (first 20 features)
    ax4 = axes[1, 1]
    if len(results['selected_features']) > 0:
        max_features_to_show = min(20, max(len(feat) for feat in results['selected_features']))
        heatmap_data = np.zeros((len(feature_counts), max_features_to_show))
        
        for i, selected in enumerate(results['selected_features']):
            for feature_idx in selected[:max_features_to_show]:
                if feature_idx < max_features_to_show:
                    heatmap_data[i, feature_idx] = 1
        
        sns.heatmap(heatmap_data, ax=ax4, cmap='RdYlBu_r', cbar=True,
                   xticklabels=[f'F{i}' for i in range(max_features_to_show)],
                   yticklabels=feature_counts)
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Number of Features Selected')
        ax4.set_title(f'Feature Selection Pattern (First {max_features_to_show} features)')
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(X, y, selected_features, feature_names=None, classifier=None):
    """
    Analyze the importance of selected features.
    """
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Get data for selected features only
    X_selected = X[:, selected_features] if isinstance(X, np.ndarray) else X.iloc[:, selected_features]
    
    # Fit classifier to get feature importance
    classifier.fit(X_selected, y)
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        selected_feature_names = [feature_names[i] for i in selected_features]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature_index': selected_features,
            'feature_name': selected_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        top_n = min(20, len(importance_df))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature_name')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        
        # Plot cumulative importance
        plt.subplot(2, 1, 2)
        cumulative_importance = np.cumsum(importance_df['importance'].values)
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, marker='o')
        plt.xlabel('Number of Features (Ranked by Importance)')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print(f"Classifier {type(classifier).__name__} does not provide feature importance scores.")
        return None

def compare_classifiers_sfs(X, y, classifiers=None, feature_range=(5, 50, 5)):
    """
    Compare different classifiers using sequential feature selection.
    """
    if classifiers is None:
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
    
    results_comparison = {}
    
    plt.figure(figsize=(12, 8))
    
    for name, classifier in classifiers.items():
        print(f"\n{'='*20} {name} {'='*20}")
        results = advanced_sequential_feature_selection(
            X, y, classifier=classifier, 
            feature_range=feature_range,
            verbose=False, plot_results=False
        )
        results_comparison[name] = results
        
        # Plot results
        plt.errorbar(results['feature_counts'], results['accuracies'], 
                    yerr=results['std_deviations'],
                    label=f"{name} (Best: {results['best_accuracy']:.4f})",
                    marker='o', linewidth=2, markersize=6, capsize=3)
        
        print(f"Best accuracy: {results['best_accuracy']:.4f}")
        print(f"Best feature count: {results['best_feature_count']}")
    
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Classifier Comparison - Sequential Feature Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results_comparison

# Demo function with synthetic data
def demo_sequential_feature_selection():
    """
    Demonstration of sequential feature selection with synthetic data.
    """
    print("🚀 Demo: Advanced Sequential Feature Selection")
    print("=" * 50)
    
    # Create synthetic dataset
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=15,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"Dataset created: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Classes: {np.unique(y, return_counts=True)}")
    
    # Test with RandomForest
    print("\n🌲 Testing with Random Forest...")
    rf_results = advanced_sequential_feature_selection(
        X_df, y, 
        classifier=RandomForestClassifier(n_estimators=100, random_state=42),
        feature_range=(5, 30, 5),
        verbose=True,
        plot_results=True
    )
    
    # Analyze best features
    print("\n🔍 Analyzing best feature importance...")
    importance_df = analyze_feature_importance(
        X_df.values, y, 
        rf_results['best_features'], 
        feature_names,
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    
    # Compare classifiers
    print("\n⚖️ Comparing classifiers...")
    comparison_results = compare_classifiers_sfs(X_df, y, feature_range=(5, 25, 5))
    
    return rf_results, importance_df, comparison_results

if __name__ == "__main__":
    # Run demo
    demo_results = demo_sequential_feature_selection()
