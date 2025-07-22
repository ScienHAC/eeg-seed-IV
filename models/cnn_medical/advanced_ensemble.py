"""
Advanced Ensemble Classifier for SEED-IV Dataset
================================================

Since CNNs fail due to small dataset (1080 samples), let's create an
advanced ensemble that builds on your successful Sequential Feature Selection.

This approach uses:
1. Your successful 25-feature selection
2. Multiple advanced classifiers (not CNN)
3. Ensemble voting for better accuracy
4. Proper techniques for small datasets

Goal: Beat your 50% SFS by using advanced ML (not deep learning)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_your_successful_setup():
    """Load your successful Sequential Feature Selection results"""
    print("🎯 Loading your successful SFS setup...")
    
    # Load your saved 25 features
    saved_features_path = Path("saved_models/selected_features_25.npy")
    if not saved_features_path.exists():
        print("❌ Your 25-feature selection not found!")
        return None
    
    selected_features = np.load(str(saved_features_path))
    print(f"✅ Loaded your 25 optimal features")
    return selected_features

def load_data_with_successful_preprocessing():
    """Load data using your successful preprocessing"""
    print("🔄 Loading data with your successful preprocessing...")
    
    # Import your successful SFS module
    import sys
    import importlib.util
    
    sfs_path = Path("models/sequential_feature_selection/clean_eeg_classifier.py").resolve()
    if not sfs_path.exists():
        print(f"❌ SFS module not found")
        return None, None
    
    spec = importlib.util.spec_from_file_location("clean_eeg_classifier", sfs_path)
    clean_eeg_classifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clean_eeg_classifier)
    
    # Load data using your successful method
    X, y, metadata = clean_eeg_classifier.load_clean_seed_iv_data("csv", "de_LDS", max_subjects=15)
    
    if len(X) == 0:
        print("❌ Could not load SEED-IV data")
        return None, None
    
    print(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def create_advanced_ensemble():
    """
    Create advanced ensemble of multiple classifiers
    Optimized for small datasets (better than CNN for 1080 samples)
    """
    print("🚀 Creating advanced ensemble classifiers...")
    
    # Base classifiers optimized for small datasets
    classifiers = {
        'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42),
        'svm_poly': SVC(kernel='poly', degree=3, probability=True, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'extra_trees': ExtraTreesClassifier(n_estimators=200, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=7),
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=list(classifiers.items()),
        voting='soft'  # Use probability voting
    )
    
    print(f"✅ Created ensemble with {len(classifiers)} advanced classifiers")
    return ensemble, classifiers

def hyperparameter_optimization(X, y):
    """
    Optimize hyperparameters for best classifiers
    """
    print("🔧 Optimizing hyperparameters...")
    
    # Define parameter grids for top performers
    param_grids = {
        'RandomForest': {
            'classifier': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'classifier': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'SVM': {
            'classifier': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
    }
    
    best_models = {}
    
    for name, config in param_grids.items():
        print(f"   Optimizing {name}...")
        
        grid_search = GridSearchCV(
            config['classifier'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        best_models[name] = grid_search.best_estimator_
        
        print(f"   Best {name} score: {grid_search.best_score_:.3f}")
    
    return best_models

def evaluate_advanced_ensemble(ensemble, X, y):
    """Comprehensive evaluation with cross-validation"""
    print("📊 Evaluating advanced ensemble...")
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
    
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)
    
    print(f"✅ Cross-validation results:")
    print(f"   Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"   Individual folds: {cv_scores}")
    
    return mean_accuracy, std_accuracy, cv_scores

def compare_all_approaches(X, y, selected_features):
    """Compare all approaches: SFS baseline vs individual vs ensemble"""
    print("🏆 Comprehensive comparison of all approaches...")
    
    # Use EXACT same preprocessing as successful SFS
    # Step 1: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Pre-filter to 200 features (same as SFS)
    selector = SelectKBest(f_classif, k=200)
    X_filtered = selector.fit_transform(X_scaled, y)
    
    # Step 3: Apply the EXACT boolean mask from SFS
    # selected_features is a boolean mask of shape (200,)
    X_selected = X_filtered[:, selected_features]
    
    print(f"📊 Using exact SFS preprocessing:")
    print(f"   Original features: {X.shape[1]}")
    print(f"   After filtering: {X_filtered.shape[1]}")
    print(f"   Selected features: {np.sum(selected_features)} out of {len(selected_features)}")
    print(f"   Final feature matrix: {X_selected.shape}")
    
    # Test individual classifiers
    individual_results = {}
    ensemble, classifiers = create_advanced_ensemble()
    
    print("\n🔍 Testing individual classifiers:")
    for name, classifier in classifiers.items():
        scores = cross_val_score(classifier, X_selected, y, cv=5, scoring='accuracy')
        individual_results[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
        print(f"   {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # Test ensemble
    print("\n🚀 Testing advanced ensemble:")
    ensemble_mean, ensemble_std, ensemble_scores = evaluate_advanced_ensemble(ensemble, X_selected, y)
    
    # Hyperparameter optimization for best performers
    print("\n🔧 Hyperparameter optimization:")
    best_models = hyperparameter_optimization(X_selected, y)
    
    optimized_results = {}
    for name, model in best_models.items():
        scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
        optimized_results[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
        print(f"   Optimized {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    return individual_results, ensemble_mean, ensemble_std, optimized_results

def plot_comprehensive_comparison(individual_results, ensemble_mean, optimized_results, sfs_baseline=0.50):
    """Plot comprehensive comparison of all approaches"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Individual classifiers
    names = list(individual_results.keys())
    means = [individual_results[name]['mean'] for name in names]
    stds = [individual_results[name]['std'] for name in names]
    
    bars1 = ax1.bar(range(len(names)), means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
    ax1.axhline(y=sfs_baseline, color='red', linestyle='--', label=f'SFS Baseline ({sfs_baseline:.1%})')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Individual Classifiers vs SFS Baseline')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bar, mean in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Optimized vs Ensemble
    opt_names = list(optimized_results.keys()) + ['Ensemble']
    opt_means = [optimized_results[name]['mean'] for name in optimized_results.keys()] + [ensemble_mean]
    opt_colors = ['green', 'orange', 'purple', 'blue']
    
    bars2 = ax2.bar(opt_names, opt_means, color=opt_colors, alpha=0.7)
    ax2.axhline(y=sfs_baseline, color='red', linestyle='--', label=f'SFS Baseline ({sfs_baseline:.1%})')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Optimized Models vs Ensemble')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for bar, mean in zip(bars2, opt_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Best approach summary
    best_individual = max(individual_results.items(), key=lambda x: x[1]['mean'])
    best_optimized = max(optimized_results.items(), key=lambda x: x[1]['mean'])
    
    all_approaches = [
        ('SFS Baseline', sfs_baseline),
        ('Best Individual', best_individual[1]['mean']),
        ('Best Optimized', best_optimized[1]['mean']),
        ('Ensemble', ensemble_mean)
    ]
    
    approach_names, approach_scores = zip(*all_approaches)
    colors = ['red', 'blue', 'green', 'purple']
    
    bars3 = ax3.bar(approach_names, approach_scores, color=colors, alpha=0.7)
    ax3.axhline(y=sfs_baseline, color='red', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Overall Comparison')
    ax3.grid(True, alpha=0.3)
    
    for bar, score in zip(bars3, approach_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Improvement summary
    best_score = max(approach_scores)
    improvement = (best_score - sfs_baseline) / sfs_baseline * 100
    
    ax4.text(0.5, 0.7, f'Best Advanced Approach', ha='center', fontsize=16, fontweight='bold')
    ax4.text(0.5, 0.5, f'{best_score:.1%}', ha='center', fontsize=24, fontweight='bold', 
             color='green' if best_score > sfs_baseline else 'red')
    ax4.text(0.5, 0.3, f'{improvement:+.1f}% vs SFS', ha='center', fontsize=14,
             color='green' if improvement > 0 else 'red')
    
    status = "SUCCESS" if best_score > sfs_baseline else "NEEDS MORE WORK"
    ax4.text(0.5, 0.1, status, ha='center', fontsize=12, 
             color='green' if best_score > sfs_baseline else 'red')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('advanced_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_advanced_ensemble_pipeline():
    """
    Main pipeline for advanced ensemble approach
    Better than CNN for small datasets
    """
    print("🚀 ADVANCED ENSEMBLE PIPELINE - PROPER SMALL DATASET APPROACH")
    print("=" * 70)
    print("🎯 Goal: Beat 50% SFS with advanced ML (not deep learning)")
    
    # Step 1: Load your successful setup
    selected_features = load_your_successful_setup()
    if selected_features is None:
        return None
    
    # Step 2: Load data with your successful preprocessing
    X, y = load_data_with_successful_preprocessing()
    if X is None:
        return None
    
    # Step 3: Comprehensive comparison
    print("\n🔍 Running comprehensive comparison...")
    individual_results, ensemble_mean, ensemble_std, optimized_results = compare_all_approaches(X, y, selected_features)
    
    # Step 4: Find best approach
    best_individual = max(individual_results.items(), key=lambda x: x[1]['mean'])
    best_optimized = max(optimized_results.items(), key=lambda x: x[1]['mean'])
    
    all_scores = [
        ('SFS Baseline', 0.50),
        ('Best Individual', best_individual[1]['mean']),
        ('Best Optimized', best_optimized[1]['mean']),
        ('Ensemble', ensemble_mean)
    ]
    
    best_approach, best_score = max(all_scores, key=lambda x: x[1])
    
    # Step 5: Plot results
    print(f"\n📊 Creating comprehensive comparison plots...")
    plot_comprehensive_comparison(individual_results, ensemble_mean, optimized_results)
    
    # Step 6: Save results
    improvement = (best_score - 0.50) / 0.50 * 100
    
    results_summary = {
        'method': 'Advanced Ensemble',
        'best_approach': best_approach,
        'best_accuracy': best_score,
        'beats_sfs': best_score > 0.50,
        'improvement_over_sfs': improvement,
        'sfs_baseline': 0.50,
        'total_samples': len(X),
        'selected_features': 25,
        'individual_best': f"{best_individual[0]}: {best_individual[1]['mean']:.3f}",
        'optimized_best': f"{best_optimized[0]}: {best_optimized[1]['mean']:.3f}",
        'ensemble_score': ensemble_mean
    }
    
    pd.DataFrame([results_summary]).to_csv('advanced_ensemble_results.csv', index=False)
    
    print(f"\n🏆 ADVANCED ENSEMBLE RESULTS:")
    print(f"   Best Approach: {best_approach}")
    print(f"   Best Accuracy: {best_score:.3f}")
    print(f"   SFS Baseline: 50.0%")
    print(f"   Beats SFS: {'✅ YES' if best_score > 0.50 else '❌ NO'}")
    print(f"   Improvement: {improvement:+.1f}% vs SFS")
    print(f"   Why this works: Advanced ML optimized for small datasets (1080 samples)")
    print(f"   Status: {'🎯 SUCCESS' if best_score > 0.50 else '🔧 NEEDS MORE FEATURES'}")
    
    return results_summary

if __name__ == "__main__":
    results = run_advanced_ensemble_pipeline()
