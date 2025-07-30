"""
Advanced Feature Selection for EEG Emotion Recognition
=====================================================

This module implements sophisticated feature selection techniques specifically
designed for EEG emotion recognition tasks. It provides multiple selection
methods and automatic optimization to find the best subset of features.

Key Features:
- Multiple selection algorithms (Random Forest, Mutual Information, F-test, etc.)
- Automatic feature count optimization  
- Cross-validation for robust selection
- Feature importance analysis and visualization
- Joblib integration for saving/loading selected features

Author: AI Assistant
Date: July 28, 2025
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

# Scikit-learn imports
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureSelector:
    """
    Advanced feature selection with multiple methods and optimization.
    
    This class provides comprehensive feature selection capabilities:
    - Multiple selection algorithms
    - Automatic feature count optimization
    - Cross-validation for robust results
    - Feature importance analysis
    - Results saving/loading with joblib
    """
    
    def __init__(self, 
                 output_dir: str = "feature_selection_results",
                 random_state: int = 42):
        """
        Initialize the AdvancedFeatureSelector.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results and selected features
        random_state : int
            Random state for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # Results storage
        self.selection_results = {}
        self.feature_rankings = {}
        self.optimal_features = {}
        self.best_method = None
        self.best_k = None
        self.best_score = 0.0
        
        # Available selection methods
        self.selection_methods = {
            'random_forest_importance': self._select_rf_importance,
            'mutual_info': self._select_mutual_info,
            'f_classif': self._select_f_classif,
            'chi2': self._select_chi2,
            'rfe_rf': self._select_rfe_rf,
            'lasso': self._select_lasso,
            'extra_trees': self._select_extra_trees
        }
        
        logger.info(f"AdvancedFeatureSelector initialized")
        logger.info(f"Output directory: {self.output_dir.resolve()}")
        logger.info(f"Available methods: {list(self.selection_methods.keys())}")
    
    def select_best_features(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           methods: List[str] = None,
                           k_range: List[int] = None,
                           cv_folds: int = 5,
                           resume_from_checkpoint: bool = True) -> Dict:
        """
        Find the best feature selection method and optimal number of features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (samples x features)
        y : np.ndarray  
            Target labels
        methods : List[str], optional
            Methods to test. If None, tests all available methods
        k_range : List[int], optional
            Range of feature counts to test. If None, uses default range
        cv_folds : int
            Number of cross-validation folds
        resume_from_checkpoint : bool
            Whether to resume from previous checkpoint if available
            
        Returns:
        --------
        Dict : Results containing best method, features, and performance
        """
        logger.info("🚀 Starting comprehensive feature selection...")
        logger.info(f"Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        if methods is None:
            methods = list(self.selection_methods.keys())
        
        if k_range is None:
            max_features = min(X.shape[1], 50)  # Don't test more than 50 features
            k_range = [10, 15, 20, 25, 30, 35, 40, max_features]
            k_range = [k for k in k_range if k <= X.shape[1]]
        
        logger.info(f"Testing methods: {methods}")
        logger.info(f"Testing feature counts: {k_range}")
        
        # Check for existing checkpoint
        checkpoint_file = self.output_dir / "checkpoint.joblib"
        results = []
        completed_combinations = set()
        
        if resume_from_checkpoint and checkpoint_file.exists():
            try:
                checkpoint_data = joblib.load(checkpoint_file)
                results = checkpoint_data.get('results', [])
                completed_combinations = set(checkpoint_data.get('completed_combinations', []))
                logger.info(f"📂 Resumed from checkpoint: {len(results)} results loaded")
                logger.info(f"🔄 Skipping {len(completed_combinations)} completed combinations")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                results = []
                completed_combinations = set()
        
        # Test all combinations of methods and feature counts
        total_combinations = len(methods) * len(k_range)
        current_combination = len(completed_combinations)
        
        logger.info(f"📊 Progress: {current_combination}/{total_combinations} combinations completed")
        
        try:
            for method in methods:
                logger.info(f"\n📊 Testing method: {method}")
                method_results = self._test_method_with_k_range(
                    X, y, method, k_range, cv_folds, completed_combinations
                )
                results.extend(method_results)
                
                # Save checkpoint after each method
                self._save_checkpoint(results, completed_combinations, X.shape)
                
        except KeyboardInterrupt:
            logger.info("🛑 INTERRUPTED BY USER!")
            logger.info("💾 Saving checkpoint...")
            self._save_checkpoint(results, completed_combinations, X.shape)
            
            if not results:
                logger.error("No results available yet. Please run longer next time.")
                return None
        
        if not results:
            logger.error("No results obtained!")
            return None
        
        # Find best combination
        best_result = max(results, key=lambda x: x['cv_score'])
        self.best_method = best_result['method']
        self.best_k = best_result['k']
        self.best_score = best_result['cv_score']
        
        logger.info(f"\n🏆 BEST RESULT:")
        logger.info(f"Method: {self.best_method}")
        logger.info(f"Features: {self.best_k}")
        logger.info(f"CV Score: {self.best_score:.4f}")
        
        # Get the actual best features
        best_features = self.selection_methods[self.best_method](X, y, self.best_k)
        
        # Save results
        self._save_results(X, y, best_features, results)
        
        # Clean up checkpoint file
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("🗑️  Checkpoint file cleaned up")
        
        return {
            'best_method': self.best_method,
            'best_k': self.best_k,
            'best_score': self.best_score,
            'selected_features': best_features,
            'all_results': results,
            'feature_names': [f'feature_{i}' for i in best_features]
        }
    
    def _test_method_with_k_range(self, 
                                  X: np.ndarray, 
                                  y: np.ndarray,
                                  method: str,
                                  k_range: List[int],
                                  cv_folds: int,
                                  completed_combinations: set = None) -> List[Dict]:
        """Test a selection method with different feature counts."""
        method_results = []
        
        for k in k_range:
            if k > X.shape[1]:
                continue
                
            combination_key = (method, k)
            if completed_combinations and combination_key in completed_combinations:
                logger.info(f"  ⏭️  Skipping k={k:2d} (already completed)")
                continue
                
            try:
                logger.info(f"  Testing k={k:2d}...")
                
                # Select features
                selected_features = self.selection_methods[method](X, y, k)
                X_selected = X[:, selected_features]
                
                # Evaluate with cross-validation using Random Forest
                # NOTE: This is the model used for accuracy-based feature selection
                rf_classifier = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                cv_scores = cross_val_score(
                    rf_classifier, X_selected, y, 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                     random_state=self.random_state),
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                result = {
                    'method': method,
                    'k': k,
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'selected_features': selected_features
                }
                
                method_results.append(result)
                
                # Update completed combinations
                if completed_combinations is not None:
                    completed_combinations.add(combination_key)
                
                logger.info(f"  k={k:2d}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.warning(f"  k={k:2d}: Failed - {str(e)}")
                continue
        
        return method_results
    
    def _save_checkpoint(self, results: List[Dict], completed_combinations: set, data_shape: tuple):
        """Save checkpoint data for resuming interrupted experiments."""
        checkpoint_file = self.output_dir / "checkpoint.joblib"
        
        checkpoint_data = {
            'results': results,
            'completed_combinations': list(completed_combinations),  # Convert set to list for JSON serialization
            'data_shape': data_shape,
            'timestamp': datetime.now().isoformat(),
            'experiment_info': {
                'random_state': self.random_state,
                'output_dir': str(self.output_dir)
            }
        }
        
        try:
            joblib.dump(checkpoint_data, checkpoint_file)
            logger.info(f"💾 Checkpoint saved: {len(results)} results, {len(completed_combinations)} completed combinations")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _select_rf_importance(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Select features based on Random Forest importance."""
        rf = RandomForestClassifier(
            n_estimators=500, 
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importance_indices = np.argsort(rf.feature_importances_)[::-1]
        return importance_indices[:k]
    
    def _select_mutual_info(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Select features based on mutual information."""
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        return selector.get_support(indices=True)
    
    def _select_f_classif(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Select features based on ANOVA F-test."""
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        return selector.get_support(indices=True)
    
    def _select_chi2(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Select features based on chi-squared test."""
        # Ensure non-negative features for chi2
        X_positive = X - X.min() + 1e-8
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(X_positive, y)
        return selector.get_support(indices=True)
    
    def _select_rfe_rf(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Select features using Recursive Feature Elimination with Random Forest."""
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        selector = RFE(estimator=rf, n_features_to_select=k)
        selector.fit(X, y)
        return selector.get_support(indices=True)
    
    def _select_lasso(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Select features using Lasso regularization."""
        # Scale features for Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use LassoCV to find optimal alpha
        lasso = LassoCV(cv=5, random_state=self.random_state, n_jobs=-1)
        
        # Convert to binary classification problem for Lasso
        from sklearn.multiclass import OneVsRestClassifier
        ovr_classifier = OneVsRestClassifier(lasso)
        ovr_classifier.fit(X_scaled, y)
        
        # Get feature importance (average across all classifiers)
        coef_importance = np.mean([np.abs(est.coef_) for est in ovr_classifier.estimators_], axis=0)
        top_indices = np.argsort(coef_importance)[::-1][:k]
        
        return top_indices
    
    def _select_extra_trees(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Select features using Extra Trees importance."""
        et = ExtraTreesClassifier(
            n_estimators=500, 
            random_state=self.random_state,
            n_jobs=-1
        )
        et.fit(X, y)
        
        importance_indices = np.argsort(et.feature_importances_)[::-1]
        return importance_indices[:k]
    
    def _save_results(self, 
                      X: np.ndarray, 
                      y: np.ndarray, 
                      selected_features: np.ndarray,
                      all_results: List[Dict]):
        """Save feature selection results and selected features."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save selected features (joblib)
        features_file = self.output_dir / f"selected_features_{timestamp}.joblib"
        joblib.dump({
            'selected_features': selected_features,
            'method': self.best_method,
            'k': self.best_k,
            'cv_score': self.best_score,
            'feature_names': [f'feature_{i}' for i in selected_features],
            'timestamp': timestamp
        }, features_file)
        
        # Save detailed results (joblib)
        results_file = self.output_dir / f"selection_results_{timestamp}.joblib"
        joblib.dump({
            'all_results': all_results,
            'best_method': self.best_method,
            'best_k': self.best_k,
            'best_score': self.best_score,
            'input_shape': X.shape,
            'n_classes': len(np.unique(y)),
            'timestamp': timestamp
        }, results_file)
        
        # NEW: Save selected features as JSON for easy reading
        import json
        json_file = self.output_dir / f"selected_features_{timestamp}.json"
        json_data = {
            'selected_features': selected_features.tolist(),
            'method': self.best_method,
            'k': self.best_k,
            'cv_score': float(self.best_score),
            'feature_names': [f'feature_{i}' for i in selected_features],
            'timestamp': timestamp,
            'total_features': int(X.shape[1]),
            'reduction_percentage': float((1 - self.best_k / X.shape[1]) * 100),
            'metadata': {
                'input_shape': list(X.shape),
                'n_classes': int(len(np.unique(y))),
                'cv_folds': 5,
                'selection_date': datetime.now().isoformat()
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # NEW: Save comprehensive results as JSON
        json_results_file = self.output_dir / f"all_results_{timestamp}.json"
        json_all_results = {
            'best_result': {
                'method': self.best_method,
                'k': self.best_k,
                'cv_score': float(self.best_score)
            },
            'all_results': [
                {
                    'method': r['method'],
                    'k': int(r['k']),
                    'cv_score': float(r['cv_score']),
                    'cv_std': float(r['cv_std']),
                    'selected_features': r['selected_features'].tolist() if hasattr(r['selected_features'], 'tolist') else list(r['selected_features'])
                } for r in all_results
            ],
            'summary': {
                'total_combinations_tested': len(all_results),
                'methods_tested': list(set(r['method'] for r in all_results)),
                'k_values_tested': sorted(list(set(r['k'] for r in all_results))),
                'best_score_overall': float(max(r['cv_score'] for r in all_results)),
                'experiment_metadata': {
                    'timestamp': timestamp,
                    'input_shape': list(X.shape),
                    'n_classes': int(len(np.unique(y)))
                }
            }
        }
        
        with open(json_results_file, 'w') as f:
            json.dump(json_all_results, f, indent=2)
        
        # Create summary report
        self._create_summary_report(all_results, timestamp)
        
        logger.info(f"✅ Results saved:")
        logger.info(f"  Features: {features_file}")
        logger.info(f"  Results: {results_file}")
    
    def _create_summary_report(self, all_results: List[Dict], timestamp: str):
        """Create a human-readable summary report."""
        report_file = self.output_dir / f"feature_selection_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("EEG FEATURE SELECTION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("BEST RESULT:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Method: {self.best_method}\n")
            f.write(f"Features Selected: {self.best_k}\n")
            f.write(f"Cross-Validation Score: {self.best_score:.4f}\n\n")
            
            f.write("ALL RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            # Group by method
            methods = {}
            for result in all_results:
                method = result['method']
                if method not in methods:
                    methods[method] = []
                methods[method].append(result)
            
            for method, results in methods.items():
                f.write(f"\n{method.upper()}:\n")
                for result in sorted(results, key=lambda x: x['k']):
                    f.write(f"  k={result['k']:2d}: {result['cv_score']:.4f} ± {result['cv_std']:.4f}\n")
        
        logger.info(f"  Report: {report_file}")
    
    def load_selected_features(self, features_file: str) -> Dict:
        """Load previously selected features."""
        return joblib.load(features_file)
    
    def visualize_results(self, all_results: List[Dict] = None):
        """Create visualization of feature selection results."""
        if all_results is None:
            logger.warning("No results to visualize")
            return
        
        # Create results DataFrame
        df = pd.DataFrame(all_results)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Performance by method and k
        plt.subplot(2, 2, 1)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            plt.plot(method_data['k'], method_data['cv_score'], 
                    marker='o', label=method, linewidth=2)
        
        plt.xlabel('Number of Features (k)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Feature Selection Performance Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Best score for each method
        plt.subplot(2, 2, 2)
        method_best = df.groupby('method')['cv_score'].max().sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(method_best)))
        bars = plt.barh(range(len(method_best)), method_best.values, color=colors)
        plt.yticks(range(len(method_best)), method_best.index)
        plt.xlabel('Best Cross-Validation Accuracy')
        plt.title('Best Performance by Method')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        # Plot 3: Feature count vs performance distribution
        plt.subplot(2, 2, 3)
        plt.scatter(df['k'], df['cv_score'], alpha=0.6, c=df['cv_score'], 
                   cmap='viridis', s=50)
        plt.colorbar(label='CV Score')
        plt.xlabel('Number of Features (k)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Feature Count vs Performance')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Method performance distribution
        plt.subplot(2, 2, 4)
        method_scores = [df[df['method'] == method]['cv_score'].values 
                        for method in df['method'].unique()]
        plt.boxplot(method_scores, labels=df['method'].unique())
        plt.xticks(rotation=45)
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Performance Distribution by Method')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"feature_selection_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"  Visualization: {plot_file}")
        
        plt.show()

def compare_selection_methods(X: np.ndarray, 
                            y: np.ndarray,
                            output_dir: str = "feature_comparison") -> Dict:
    """
    Quick comparison of different feature selection methods.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    Dict : Comparison results
    """
    selector = AdvancedFeatureSelector(output_dir=output_dir)
    return selector.select_best_features(X, y)

def optimize_feature_count(X: np.ndarray, 
                          y: np.ndarray,
                          method: str = 'random_forest_importance',
                          max_features: int = 50) -> Dict:
    """
    Optimize the number of features for a specific selection method.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    method : str
        Feature selection method to use
    max_features : int
        Maximum number of features to test
        
    Returns:
    --------
    Dict : Optimization results
    """
    selector = AdvancedFeatureSelector()
    k_range = list(range(5, min(max_features, X.shape[1]) + 1, 5))
    
    return selector.select_best_features(
        X, y, 
        methods=[method], 
        k_range=k_range
    )
