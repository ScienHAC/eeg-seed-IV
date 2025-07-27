"""
SEED-IV Comprehensive Emotion Recognition - Stage 1: Traditional Baseline

This module implements the traditional machine learning baseline using Support Vector Machine (SVM).
Target accuracy: 70-75%

Stage 1 Features:
- Basic DE (Differential Entropy) features from EEG data
- Standard preprocessing and normalization
- SVM classifier with RBF kernel
- Cross-validation for robust evaluation
- Comprehensive visualization and reporting

Author: GitHub Copilot
Date: 2024
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from datetime import datetime
import warnings
import joblib

# Scientific computing
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, learning_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

try:
    from config import Stage1Config
    from data_processing.seed_iv_loader import SeedIVLoader
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TraditionalBaseline:
    """
    Traditional machine learning baseline using Support Vector Machine
    
    This class implements Stage 1 of the comprehensive emotion recognition pipeline,
    focusing on achieving 70-75% accuracy using traditional ML approaches.
    """
    
    def __init__(self, config: Optional[Stage1Config] = None, random_state: int = 42):
        """
        Initialize the traditional baseline model
        
        Parameters:
        -----------
        config : Stage1Config, optional
            Configuration for Stage 1 model
        random_state : int
            Random seed for reproducibility
        """
        self.stage_config = config or Stage1Config()
        self.random_state = random_state
        self.target_accuracy = 0.725  # Target: 70-75%
        
        # Model components
        self.pipeline = None
        self.scaler = None
        self.feature_selector = None
        self.classifier = None
        
        # Results storage
        self.results = {}
        
        logger.info(f"TraditionalBaseline initialized with target accuracy: {self.target_accuracy:.1%}")
    
    def create_model(self) -> Pipeline:
        """
        Create the complete ML pipeline
        
        Returns:
        --------
        Pipeline : Scikit-learn pipeline with preprocessing and SVM
        """
        logger.info("Creating traditional baseline pipeline...")
        
        # Feature selection
        if self.stage_config.use_feature_selection:
            feature_selector = SelectKBest(
                score_func=f_classif,
                k=self.stage_config.n_selected_features
            )
        else:
            feature_selector = None
        
        # Scaler
        if self.stage_config.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.stage_config.scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # SVM Classifier
        svm_classifier = SVC(
            kernel=self.stage_config.svm_kernel,
            C=self.stage_config.svm_C,
            gamma=self.stage_config.svm_gamma,
            random_state=self.random_state,
            probability=True  # Enable probability estimates
        )
        
        # Create pipeline
        pipeline_steps = [('scaler', scaler)]
        
        if feature_selector is not None:
            pipeline_steps.append(('feature_selector', feature_selector))
        
        pipeline_steps.append(('classifier', svm_classifier))
        
        self.pipeline = Pipeline(pipeline_steps)
        
        logger.info(f"Pipeline created with {len(pipeline_steps)} steps")
        return self.pipeline
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              validation_strategy: str = 'cross_val') -> Dict[str, Any]:
        """
        Train the traditional baseline model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        validation_strategy : str
            Validation strategy ('cross_val', 'hold_out', or 'none')
            
        Returns:
        --------
        Dict[str, Any] : Training results and metrics
        """
        logger.info(f"Training traditional baseline model...")
        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        start_time = time.time()
        
        # Create model if not exists
        if self.pipeline is None:
            self.create_model()
        
        # Hyperparameter optimization if enabled
        if self.stage_config.use_grid_search:
            logger.info("Performing hyperparameter optimization...")
            self.pipeline = self._optimize_hyperparameters(X_train, y_train)
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Validation
        validation_results = {}
        if validation_strategy == 'cross_val':
            validation_results = self._cross_validation(X_train, y_train)
        elif validation_strategy == 'hold_out':
            validation_results = self._hold_out_validation(X_train, y_train)
        
        # Store training results
        train_results = {
            'training_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'training_time': training_time,
            'validation_strategy': validation_strategy,
            'model_params': self.pipeline.get_params(),
            **validation_results
        }
        
        self.results['training'] = train_results
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        if 'cv_mean_accuracy' in validation_results:
            logger.info(f"CV Mean Accuracy: {validation_results['cv_mean_accuracy']:.4f} ± {validation_results['cv_std_accuracy']:.4f}")
        
        return train_results
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """
        Optimize hyperparameters using GridSearchCV
        """
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'classifier__kernel': ['rbf', 'linear', 'poly']
        }
        
        if self.stage_config.use_feature_selection:
            param_grid['feature_selector__k'] = [100, 200, 300, 'all']
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Accuracy scores
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        
        # F1 scores
        f1_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1_macro')
        
        return {
            'cv_scores': cv_scores,
            'cv_mean_accuracy': np.mean(cv_scores),
            'cv_std_accuracy': np.std(cv_scores),
            'cv_f1_scores': f1_scores,
            'cv_mean_f1': np.mean(f1_scores),
            'cv_std_f1': np.std(f1_scores)
        }
    
    def _hold_out_validation(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Perform hold-out validation
        """
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Create and train model on training portion
        temp_pipeline = self.pipeline
        temp_pipeline.fit(X_train_val, y_train_val)
        
        # Evaluate on validation set
        y_val_pred = temp_pipeline.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        return {
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_samples': len(y_val)
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        Dict[str, Any] : Evaluation results and metrics
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info(f"Evaluating model on {X_test.shape[0]} test samples...")
        
        start_time = time.time()
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        
        evaluation_time = time.time() - start_time
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        eval_results = {
            'test_accuracy': test_accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_accuracy': per_class_accuracy,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'evaluation_time': evaluation_time,
            'test_samples': X_test.shape[0]
        }
        
        self.results['evaluation'] = eval_results
        
        # Log results
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
        logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
        logger.info(f"Target Achievement: {'✅' if test_accuracy >= self.target_accuracy else '❌'}")
        
        return eval_results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of results
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if 'evaluation' not in self.results:
            logger.warning("No evaluation results available for plotting")
            return
        
        eval_results = self.results['evaluation']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SEED-IV Stage 1: Traditional Baseline Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = eval_results['confusion_matrix']
        emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotions, yticklabels=emotions, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_ylabel('True Label')
        
        # 2. Per-class Performance
        per_class_acc = eval_results['per_class_accuracy']
        class_report = eval_results['classification_report']
        
        x_pos = np.arange(len(emotions))
        accuracies = per_class_acc
        f1_scores = [class_report[str(i)]['f1-score'] for i in range(len(emotions))]
        
        x_pos_acc = x_pos - 0.2
        x_pos_f1 = x_pos + 0.2
        
        axes[0, 1].bar(x_pos_acc, accuracies, 0.4, label='Accuracy', alpha=0.8)
        axes[0, 1].bar(x_pos_f1, f1_scores, 0.4, label='F1-Score', alpha=0.8)
        axes[0, 1].set_xlabel('Emotion Classes')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Per-Class Performance')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(emotions)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # 3. Model Summary
        axes[1, 0].axis('off')
        
        # Prepare summary text
        train_results = self.results.get('training', {})
        accuracy = eval_results['test_accuracy']
        target_status = "✅ Achieved" if accuracy >= self.target_accuracy else "❌ Not Achieved"
        
        summary_text = f"""MODEL SUMMARY
        
        Stage: 1 - Traditional Baseline
        Model: Support Vector Machine (SVM)
        Target Accuracy: {self.target_accuracy:.1%}
        
        PERFORMANCE:
        • Test Accuracy: {accuracy:.4f} ({accuracy:.1%})
        • F1 (Macro): {eval_results['f1_macro']:.4f}
        • F1 (Weighted): {eval_results['f1_weighted']:.4f}
        • Target Status: {target_status}
        
        TRAINING:
        • Training Samples: {train_results.get('training_samples', 'N/A')}
        • Features: {train_results.get('n_features', 'N/A')}
        • Training Time: {train_results.get('training_time', 0):.2f}s
        • Validation: {train_results.get('validation_strategy', 'N/A')}
        
        MODEL PARAMETERS:
        • Kernel: {self.stage_config.svm_kernel}
        • C: {self.stage_config.svm_C}
        • Gamma: {self.stage_config.svm_gamma}
        """
        
        axes[1, 0].text(0.1, 0.9, summary_text, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 4. Learning Progress (if cross-validation results available)
        if 'cv_scores' in train_results:
            cv_scores = train_results['cv_scores']
            axes[1, 1].plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', linewidth=2, markersize=8)
            axes[1, 1].axhline(y=self.target_accuracy, color='r', linestyle='--', 
                              label=f'Target ({self.target_accuracy:.1%})')
            axes[1, 1].axhline(y=np.mean(cv_scores), color='g', linestyle='-', 
                              label=f'Mean ({np.mean(cv_scores):.3f})')
            axes[1, 1].set_xlabel('CV Fold')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Cross-Validation Performance')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        else:
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, 'Cross-validation\nresults not available', 
                           transform=axes[1, 1].transAxes, ha='center', va='center',
                           fontsize=12, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model and results
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.pipeline is None:
            logger.warning("No trained model to save")
            return
        
        save_data = {
            'model': self.pipeline,
            'results': self.results,
            'config': self.stage_config.__dict__,
            'random_state': self.random_state
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        try:
            save_data = joblib.load(filepath)
            
            self.pipeline = save_data['model']
            self.results = save_data.get('results', {})
            
            logger.info(f"Model loaded from: {filepath}")
            
            if 'evaluation' in self.results:
                accuracy = self.results['evaluation']['test_accuracy']
                logger.info(f"Loaded model accuracy: {accuracy:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (not directly available for SVM)
        
        Returns:
        --------
        Optional[np.ndarray] : Feature importance scores
        """
        logger.warning("Feature importance not directly available for SVM")
        logger.info("Consider using permutation importance or SHAP values for interpretability")
        return None
    
    def run_complete_pipeline(self, data_config, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete Stage 1 pipeline
        
        Parameters:
        -----------
        data_config : DataConfig
            Data configuration object
        save_results : bool
            Whether to save results to files
            
        Returns:
        --------
        Dict[str, Any] : Complete results dictionary
        """
        logger.info("Running Stage 1: Traditional Baseline Pipeline")
        
        # Load data
        loader = SeedIVLoader(data_config)
        features, labels, subjects = loader.load_all_subjects()
        
        if len(features) == 0:
            logger.error("No data loaded!")
            return {'error': 'No data available'}
        
        logger.info(f"Loaded {len(features)} samples from {len(subjects)} subjects")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=labels
        )
        
        # Train model
        train_results = self.train(X_train, y_train, validation_strategy='cross_val')
        
        # Evaluate model
        eval_results = self.evaluate(X_test, y_test)
        
        # Create comprehensive results
        complete_results = {
            'stage': 1,
            'model_name': 'Traditional Baseline (SVM)',
            'model_type': 'Support Vector Machine',
            'accuracy': eval_results['test_accuracy'],
            'f1_score': eval_results['f1_macro'],
            'target_accuracy': self.target_accuracy,
            'target_achieved': eval_results['test_accuracy'] >= self.target_accuracy,
            'subjects': subjects,
            'training': train_results,
            'evaluation': eval_results,
            'processing_time': train_results.get('training_time', 0) + eval_results.get('evaluation_time', 0)
        }
        
        # Save results if requested
        if save_results:
            output_dir = data_config.output_dir / "stage1_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.save_model(output_dir / "traditional_baseline_model.joblib")
            
            # Save plot
            self.plot_results(save_path=output_dir / "stage1_results.png")
        
        return complete_results


def main():
    """
    Main function for Stage 1 demonstration
    """
    print("🧠 Stage 1: Traditional Baseline Model")
    print("=" * 50)
    print("Target: 70-75% accuracy using SVM")
    print("Features: Basic DE features (310 dimensions)")
    print()
    
    # Initialize model
    baseline = TraditionalBaseline()
    
    # This would run the complete pipeline if data path is provided
    print("⚠️  Please configure data path in config.py to run the complete pipeline")
    
    return baseline


if __name__ == "__main__":
    main()
