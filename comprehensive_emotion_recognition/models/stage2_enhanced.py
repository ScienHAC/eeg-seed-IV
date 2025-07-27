"""
SEED-IV Comprehensive Emotion Recognition - Stage 2: Enhanced Features

This module implements enhanced feature engineering with Random Forest classifier.
Target accuracy: 75-80%

Stage 2 Features:
- Multi-domain feature engineering (spatial, temporal, frequency, connectivity)
- Advanced feature selection techniques
- Random Forest classifier with hyperparameter optimization
- Comprehensive visualization and analysis

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel,
    SequentialFeatureSelector
)

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

try:
    from config import Stage2Config
    from data_processing.seed_iv_loader import SeedIVLoader
    from data_processing.feature_engineering import AdvancedFeatureEngineer
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


class EnhancedFeaturesModel:
    """
    Enhanced features model using Random Forest with advanced feature engineering
    
    This class implements Stage 2 of the comprehensive emotion recognition pipeline,
    focusing on achieving 75-80% accuracy using enhanced multi-domain features.
    """
    
    def __init__(self, config: Optional[Stage2Config] = None, random_state: int = 42):
        """
        Initialize the enhanced features model
        
        Parameters:
        -----------
        config : Stage2Config, optional
            Configuration for Stage 2 model
        random_state : int
            Random seed for reproducibility
        """
        self.stage_config = config or Stage2Config()
        self.random_state = random_state
        self.target_accuracy = 0.775  # Target: 75-80%
        
        # Model components
        self.feature_engineer = None
        self.scaler = None
        self.feature_selector = None
        self.model = None
        
        # Results storage
        self.results = {}
        self.selected_indices = None
        
        logger.info(f"EnhancedFeaturesModel initialized with target accuracy: {self.target_accuracy:.1%}")
    
    def extract_enhanced_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract enhanced multi-domain features
        
        Parameters:
        -----------
        X : np.ndarray
            Input EEG features
            
        Returns:
        --------
        np.ndarray : Enhanced feature matrix
        """
        logger.info("Extracting enhanced multi-domain features...")
        
        if self.feature_engineer is None:
            self.feature_engineer = AdvancedFeatureEngineer(
                n_channels=62,
                n_frequency_bands=5
            )
        
        # Extract enhanced features
        enhanced_features = self.feature_engineer.extract_all_features(X)
        
        logger.info(f"Enhanced features shape: {enhanced_features.shape}")
        logger.info(f"Feature expansion: {X.shape[1]} -> {enhanced_features.shape[1]} ({enhanced_features.shape[1]/X.shape[1]:.1f}x)")
        
        return enhanced_features
    
    def select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform feature selection
        
        Parameters:
        -----------
        X : np.ndarray
            Enhanced features
        y : np.ndarray
            Labels
            
        Returns:
        --------
        np.ndarray : Selected features
        """
        logger.info(f"Feature selection using {self.stage_config.feature_selection_method}...")
        
        if self.stage_config.feature_selection_method == 'select_k_best':
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.stage_config.n_selected_features, X.shape[1])
            )
        elif self.stage_config.feature_selection_method == 'rfe':
            base_estimator = RandomForestClassifier(
                n_estimators=50, 
                random_state=self.random_state
            )
            self.feature_selector = RFE(
                estimator=base_estimator,
                n_features_to_select=min(self.stage_config.n_selected_features, X.shape[1])
            )
        elif self.stage_config.feature_selection_method == 'select_from_model':
            base_estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            )
            self.feature_selector = SelectFromModel(
                estimator=base_estimator,
                max_features=min(self.stage_config.n_selected_features, X.shape[1])
            )
        else:
            logger.warning(f"Unknown feature selection method: {self.stage_config.feature_selection_method}")
            logger.info("Using SelectKBest with f_classif as fallback")
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.stage_config.n_selected_features, X.shape[1])
            )
        
        # Fit and transform
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Store selected feature indices
        if hasattr(self.feature_selector, 'get_support'):
            self.selected_indices = self.feature_selector.get_support(indices=True)
        
        logger.info(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        return X_selected
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """
        Optimize Random Forest hyperparameters
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
            
        Returns:
        --------
        RandomForestClassifier : Optimized model
        """
        logger.info("Optimizing Random Forest hyperparameters...")
        
        # Parameter grid for optimization
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Base model
        base_model = RandomForestClassifier(random_state=self.random_state)
        
        # Use RandomizedSearch for efficiency
        if self.stage_config.use_randomized_search:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=50,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        else:
            # Use GridSearch (more thorough but slower)
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        
        # Fit the search
        search.fit(X, y)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train the enhanced features model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
            
        Returns:
        --------
        Dict[str, Any] : Training results and metrics
        """
        logger.info("Training enhanced features model...")
        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        start_time = time.time()
        
        # Step 1: Extract enhanced features
        enhanced_features = self.extract_enhanced_features(X_train)
        
        # Step 2: Scale features
        if self.stage_config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.stage_config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        enhanced_features_scaled = self.scaler.fit_transform(enhanced_features)
        
        # Step 3: Feature selection
        selected_features = self.select_features(enhanced_features_scaled, y_train)
        
        # Step 4: Model training
        if optimize_hyperparams:
            self.model = self.optimize_hyperparameters(selected_features, y_train)
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.stage_config.n_estimators,
                max_depth=self.stage_config.max_depth,
                min_samples_split=self.stage_config.min_samples_split,
                min_samples_leaf=self.stage_config.min_samples_leaf,
                max_features=self.stage_config.max_features,
                random_state=self.random_state
            )
            self.model.fit(selected_features, y_train)
        
        training_time = time.time() - start_time
        
        # Cross-validation evaluation
        cv_results = self._cross_validation(selected_features, y_train)
        
        # Store training results
        train_results = {
            'training_samples': X_train.shape[0],
            'original_features': X_train.shape[1],
            'enhanced_features': enhanced_features.shape[1],
            'selected_features': selected_features.shape[1],
            'training_time': training_time,
            'model_params': self.model.get_params(),
            **cv_results
        }
        
        self.results['training'] = train_results
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Feature pipeline: {X_train.shape[1]} -> {enhanced_features.shape[1]} -> {selected_features.shape[1]}")
        
        if 'cv_mean' in cv_results:
            logger.info(f"CV Mean Accuracy: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        
        return train_results
    
    def _cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Accuracy scores
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        # F1 scores
        f1_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_macro')
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_f1_scores': f1_scores,
            'cv_f1_mean': np.mean(f1_scores),
            'cv_f1_std': np.std(f1_scores)
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
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info(f"Evaluating model on {X_test.shape[0]} test samples...")
        
        start_time = time.time()
        
        # Transform test data through the same pipeline
        enhanced_features = self.feature_engineer.extract_all_features(X_test)
        enhanced_features_scaled = self.scaler.transform(enhanced_features)
        
        # Apply feature selection if available
        if self.feature_selector is not None:
            selected_features = self.feature_selector.transform(enhanced_features_scaled)
        else:
            logger.warning("No feature selector available, using all features")
            selected_features = enhanced_features_scaled
        
        # Predictions
        y_pred = self.model.predict(selected_features)
        y_pred_proba = self.model.predict_proba(selected_features)
        
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
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        
        eval_results = {
            'test_accuracy': test_accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_accuracy': per_class_accuracy,
            'feature_importance': feature_importance,
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
        train_results = self.results.get('training', {})
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('SEED-IV Stage 2: Enhanced Features Results', fontsize=16, fontweight='bold')
        
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
        
        # 3. Feature Importance (Top 20)
        feature_importance = eval_results['feature_importance']
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        
        axes[0, 2].barh(range(len(top_importance)), top_importance)
        axes[0, 2].set_xlabel('Importance')
        axes[0, 2].set_ylabel('Feature Index')
        axes[0, 2].set_title('Top 20 Feature Importance')
        axes[0, 2].set_yticks(range(len(top_importance)))
        axes[0, 2].set_yticklabels([f'F{idx}' for idx in top_indices])
        
        # 4. Feature Engineering Pipeline
        axes[1, 0].axis('off')
        pipeline_text = f"""FEATURE ENGINEERING PIPELINE
        
        Original Features: {train_results.get('original_features', 'N/A')}
        Enhanced Features: {train_results.get('enhanced_features', 'N/A')}
        Selected Features: {train_results.get('selected_features', 'N/A')}
        
        Feature Types:
        • Spatial: {'✅' if self.stage_config.use_spatial_features else '❌'}
        • Temporal: {'✅' if self.stage_config.use_temporal_features else '❌'}
        • Connectivity: {'✅' if self.stage_config.use_connectivity_features else '❌'}
        • Frequency: ✅
        
        Selection Method: {self.stage_config.feature_selection_method}
        Target Features: {self.stage_config.n_selected_features}
        """
        
        axes[1, 0].text(0.1, 0.9, pipeline_text, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 5. Model Summary
        axes[1, 1].axis('off')
        accuracy = eval_results['test_accuracy']
        target_status = "✅ Achieved" if accuracy >= self.target_accuracy else "❌ Not Achieved"
        
        summary_text = f"""MODEL SUMMARY
        
        Stage: 2 - Enhanced Features
        Model: Random Forest
        Target Accuracy: {self.target_accuracy:.1%}
        
        PERFORMANCE:
        • Test Accuracy: {accuracy:.4f} ({accuracy:.1%})
        • F1 (Macro): {eval_results['f1_macro']:.4f}
        • F1 (Weighted): {eval_results['f1_weighted']:.4f}
        • Target Status: {target_status}
        
        TRAINING:
        • Training Samples: {train_results.get('training_samples', 'N/A')}
        • Training Time: {train_results.get('training_time', 0):.2f}s
        • CV Score: {train_results.get('cv_mean', 0):.4f} ± {train_results.get('cv_std', 0):.4f}
        
        MODEL PARAMETERS:
        • Estimators: {self.stage_config.n_estimators}
        • Max Depth: {self.stage_config.max_depth}
        • Min Samples Split: {self.stage_config.min_samples_split}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 6. Cross-Validation Results
        if 'cv_scores' in train_results:
            cv_scores = train_results['cv_scores']
            axes[1, 2].plot(range(1, len(cv_scores) + 1), cv_scores, 'go-', 
                           linewidth=2, markersize=8, label='CV Scores')
            axes[1, 2].axhline(y=self.target_accuracy, color='r', linestyle='--', 
                              label=f'Target ({self.target_accuracy:.1%})')
            axes[1, 2].axhline(y=np.mean(cv_scores), color='b', linestyle='-', 
                              label=f'Mean ({np.mean(cv_scores):.3f})')
            axes[1, 2].set_xlabel('CV Fold')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].set_title('Cross-Validation Performance')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim([0, 1])
        else:
            axes[1, 2].axis('off')
            axes[1, 2].text(0.5, 0.5, 'CV results\nnot available', 
                           transform=axes[1, 2].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the complete model pipeline
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        save_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_selector': self.feature_selector,
            'scaler': self.scaler,
            'selected_indices': getattr(self, 'selected_indices', None),
            'results': self.results,
            'config': self.stage_config.__dict__,
            'random_state': self.random_state
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Enhanced features model saved to: {filepath}")
    
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
            
            self.model = save_data['model']
            self.feature_engineer = save_data['feature_engineer']
            self.feature_selector = save_data['feature_selector']
            self.scaler = save_data['scaler']
            self.selected_indices = save_data.get('selected_indices')
            self.results = save_data.get('results', {})
            
            logger.info(f"Enhanced features model loaded from: {filepath}")
            
            if 'evaluation' in self.results:
                accuracy = self.results['evaluation']['test_accuracy']
                logger.info(f"Loaded model accuracy: {accuracy:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
    
    def run_complete_pipeline(self, data_config, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete Stage 2 pipeline
        
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
        logger.info("Running Stage 2: Enhanced Features Pipeline")
        
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
        train_results = self.train(X_train, y_train, optimize_hyperparams=True)
        
        # Evaluate model
        eval_results = self.evaluate(X_test, y_test)
        
        # Create comprehensive results
        complete_results = {
            'stage': 2,
            'model_name': 'Enhanced Features (Random Forest)',
            'model_type': 'Random Forest with Enhanced Features',
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
            output_dir = Path(data_config.csv_output_dir) / "stage2_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.save_model(output_dir / "enhanced_features_model.joblib")
            
            # Save plot
            self.plot_results(save_path=output_dir / "stage2_results.png")
        
        return complete_results


def main():
    """
    Main function for Stage 2 demonstration
    """
    print("🧠 Stage 2: Enhanced Features Model")
    print("=" * 50)
    print("Target: 75-80% accuracy using Random Forest")
    print("Features: Multi-domain enhanced features")
    print("Pipeline: Feature engineering + selection + optimization")
    print()
    
    # Initialize model
    enhanced_model = EnhancedFeaturesModel()
    
    # This would run the complete pipeline if data path is provided
    print("⚠️  Please configure data path in config.py to run the complete pipeline")
    
    return enhanced_model


if __name__ == "__main__":
    main()
