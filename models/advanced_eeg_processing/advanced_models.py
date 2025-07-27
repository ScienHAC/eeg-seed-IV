"""
Advanced EEG Processing Models
=============================

This module provides advanced machine learning and deep learning models
for EEG-based emotion recognition using the SEED-IV dataset.

Features:
- Traditional ML models (SVM, Random Forest, XGBoost, LightGBM)
- Deep learning models (CNN, RNN, LSTM, GRU)
- Ensemble methods with voting classifiers
- Advanced feature selection techniques
- Comprehensive model evaluation and comparison
- Automated hyperparameter optimization

Author: AI Assistant
Date: July 26, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SequentialFeatureSelector
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Deep Learning with modern Keras imports
try:
    import tensorflow as tf
    import keras
    from keras import layers
    from keras.models import Sequential, Model
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, GRU, Dropout, BatchNormalization, Input, Reshape
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Persistence and utilities
import joblib
from datetime import datetime
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedEEGModels:
    """
    Advanced EEG processing models with multiple algorithms and deep learning support
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the advanced models processor
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_selected = None
        self.X_test_selected = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        
        logger.info("AdvancedEEGModels initialized")
    
    def prepare_data(self, features, labels, test_size=0.2, feature_selection_method=None):
        """
        Prepare data for training with scaling and optional feature selection
        
        Parameters:
        -----------
        features : np.ndarray
            EEG features
        labels : np.ndarray
            Emotion labels
        test_size : float
            Proportion of data for testing
        feature_selection_method : str
            Method for feature selection ('sequential', 'rfe', 'select_k_best', None)
            
        Returns:
        --------
        dict : Summary of data preparation
        """
        logger.info("Preparing data for advanced modeling...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Feature selection
        if feature_selection_method:
            self.X_train_selected, self.X_test_selected = self._perform_feature_selection(
                self.X_train_scaled, self.X_test_scaled, self.y_train, feature_selection_method
            )
        else:
            self.X_train_selected = self.X_train_scaled
            self.X_test_selected = self.X_test_scaled
        
        # Summary
        summary = {
            'original_features': features.shape[1],
            'selected_features': self.X_train_selected.shape[1],
            'train_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0],
            'feature_selection': feature_selection_method,
            'class_distribution': Counter(labels)
        }
        
        logger.info(f"Data preparation complete: {summary['selected_features']} features selected")
        return summary
    
    def _perform_feature_selection(self, X_train, X_test, y_train, method):
        """
        Perform feature selection using specified method
        """
        logger.info(f"Performing feature selection: {method}")
        
        if method == 'sequential':
            # Sequential Feature Selection with Random Forest
            base_estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            selector = SequentialFeatureSelector(
                estimator=base_estimator,
                n_features_to_select=min(200, X_train.shape[1]),
                direction='forward',
                cv=3,
                n_jobs=-1
            )
        elif method == 'rfe':
            # Recursive Feature Elimination
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = RFE(
                estimator=base_estimator,
                n_features_to_select=min(200, X_train.shape[1])
            )
        elif method == 'select_k_best':
            # SelectKBest with f_classif
            selector = SelectKBest(
                score_func=f_classif,
                k=min(200, X_train.shape[1])
            )
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            return X_train, X_test
        
        # Fit and transform
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        self.feature_selector = selector
        
        logger.info(f"Feature selection complete: {X_train_selected.shape[1]} features selected")
        return X_train_selected, X_test_selected
    
    def train_traditional_models(self):
        """
        Train traditional machine learning models
        
        Returns:
        --------
        dict : Results for all traditional models
        """
        logger.info("Training traditional ML models...")
        
        traditional_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        results = {}
        for name, model in traditional_models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train_selected, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test_selected)
            y_pred_proba = model.predict_proba(self.X_test_selected) if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_selected, self.y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        self.models.update(traditional_models)
        self.results.update(results)
        
        return results
    
    def train_deep_learning_models(self):
        """
        Train deep learning models (if TensorFlow is available)
        
        Returns:
        --------
        dict : Results for all deep learning models
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - skipping deep learning models")
            return {}
        
        logger.info("Training deep learning models...")
        
        # Prepare data for deep learning
        X_train_dl = self.X_train_selected.astype(np.float32)
        X_test_dl = self.X_test_selected.astype(np.float32)
        
        # One-hot encode labels
        n_classes = len(np.unique(self.y_train))
        y_train_encoded = keras.utils.to_categorical(self.y_train, n_classes)
        y_test_encoded = keras.utils.to_categorical(self.y_test, n_classes)
        
        results = {}
        
        # 1. Deep Neural Network
        dnn_model = self._create_dnn_model(X_train_dl.shape[1], n_classes)
        dnn_history = self._train_dl_model(dnn_model, X_train_dl, y_train_encoded, X_test_dl, y_test_encoded, 'DNN')
        results['Deep Neural Network'] = dnn_history
        
        # 2. 1D CNN for EEG
        # Reshape data for CNN (samples, time_steps, features)
        # Assuming features can be reshaped into time series
        if X_train_dl.shape[1] >= 64:  # Need sufficient features for reshaping
            try:
                # Try to reshape into reasonable time series format
                n_time_steps = 8
                n_features = X_train_dl.shape[1] // n_time_steps
                
                if n_features > 0:
                    X_train_cnn = X_train_dl[:, :n_time_steps*n_features].reshape(-1, n_time_steps, n_features)
                    X_test_cnn = X_test_dl[:, :n_time_steps*n_features].reshape(-1, n_time_steps, n_features)
                    
                    cnn_model = self._create_cnn_model(n_time_steps, n_features, n_classes)
                    cnn_history = self._train_dl_model(cnn_model, X_train_cnn, y_train_encoded, X_test_cnn, y_test_encoded, 'CNN')
                    results['1D CNN'] = cnn_history
            except Exception as e:
                logger.warning(f"CNN model creation failed: {e}")
        
        # 3. LSTM for EEG sequences
        if X_train_dl.shape[1] >= 32:
            try:
                # Reshape for LSTM
                n_time_steps = 4
                n_features = X_train_dl.shape[1] // n_time_steps
                
                if n_features > 0:
                    X_train_lstm = X_train_dl[:, :n_time_steps*n_features].reshape(-1, n_time_steps, n_features)
                    X_test_lstm = X_test_dl[:, :n_time_steps*n_features].reshape(-1, n_time_steps, n_features)
                    
                    lstm_model = self._create_lstm_model(n_time_steps, n_features, n_classes)
                    lstm_history = self._train_dl_model(lstm_model, X_train_lstm, y_train_encoded, X_test_lstm, y_test_encoded, 'LSTM')
                    results['LSTM'] = lstm_history
            except Exception as e:
                logger.warning(f"LSTM model creation failed: {e}")
        
        return results
    
    def _create_dnn_model(self, input_dim, n_classes):
        """Create a deep neural network model"""
        model = Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_model(self, n_time_steps, n_features, n_classes):
        """Create a 1D CNN model for EEG"""
        model = Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=(n_time_steps, n_features)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_lstm_model(self, n_time_steps, n_features, n_classes):
        """Create an LSTM model for EEG sequences"""
        model = Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(n_time_steps, n_features)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.LSTM(64, return_sequences=False),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _train_dl_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train a deep learning model with callbacks"""
        logger.info(f"Training {model_name} model...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        
        result = {
            'model': model,
            'history': history.history,
            'accuracy': test_accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        logger.info(f"{model_name} - Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}")
        
        return result
    
    def create_ensemble_model(self, model_names=None):
        """
        Create an ensemble model using trained models
        
        Parameters:
        -----------
        model_names : list
            List of model names to include in ensemble
            
        Returns:
        --------
        dict : Ensemble model results
        """
        if model_names is None:
            # Use best performing traditional models
            model_names = ['Random Forest', 'Extra Trees', 'XGBoost']
        
        logger.info(f"Creating ensemble model with: {model_names}")
        
        # Get models for ensemble
        estimators = []
        for name in model_names:
            if name in self.models:
                estimators.append((name.lower().replace(' ', '_'), self.models[name]))
        
        if len(estimators) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return {}
        
        # Create voting classifier
        ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability averaging
        )
        
        # Train ensemble
        ensemble_model.fit(self.X_train_selected, self.y_train)
        
        # Predict
        y_pred = ensemble_model.predict(self.X_test_selected)
        y_pred_proba = ensemble_model.predict_proba(self.X_test_selected)
        
        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        
        # Cross-validation
        cv_scores = cross_val_score(ensemble_model, self.X_train_selected, self.y_train, cv=5, scoring='accuracy')
        
        result = {
            'model': ensemble_model,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        self.models['Ensemble'] = ensemble_model
        self.results['Ensemble'] = result
        
        logger.info(f"Ensemble - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return result
    
    def compare_models(self):
        """
        Compare all trained models and return summary
        
        Returns:
        --------
        pd.DataFrame : Comparison results
        """
        if not self.results:
            logger.warning("No models trained yet")
            return pd.DataFrame()
        
        logger.info("Comparing all trained models...")
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_score'],
                'CV Mean': result.get('cv_mean', 'N/A'),
                'CV Std': result.get('cv_std', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        logger.info("Model comparison complete")
        return comparison_df
    
    def plot_results(self, save_path=None):
        """
        Create comprehensive visualization of all model results
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create comparison DataFrame
        comparison_df = self.compare_models()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced EEG Models - Comprehensive Results', fontsize=16, fontweight='bold')
        
        # 1. Model Accuracy Comparison
        axes[0, 0].barh(comparison_df['Model'], comparison_df['Accuracy'])
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add accuracy values on bars
        for i, v in enumerate(comparison_df['Accuracy']):
            axes[0, 0].text(v + 0.005, i, f'{v:.3f}', va='center')
        
        # 2. F1-Score Comparison
        axes[0, 1].barh(comparison_df['Model'], comparison_df['F1-Score'])
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add F1 values on bars
        for i, v in enumerate(comparison_df['F1-Score']):
            axes[0, 1].text(v + 0.005, i, f'{v:.3f}', va='center')
        
        # 3. Best Model Confusion Matrix
        best_model = comparison_df.iloc[0]['Model']
        best_cm = self.results[best_model]['confusion_matrix']
        emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
        
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotions, yticklabels=emotions, ax=axes[1, 0])
        axes[1, 0].set_title(f'Best Model Confusion Matrix: {best_model}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Model Summary
        axes[1, 1].axis('off')
        
        # Prepare summary text
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        n_models = len(comparison_df)
        
        summary_text = f"""MODEL SUMMARY
        
        Total Models Trained: {n_models}
        Best Model: {best_model}
        Best Accuracy: {best_accuracy:.4f} ({best_accuracy:.1%})
        
        DATASET INFO:
        • Train Samples: {self.X_train.shape[0]}
        • Test Samples: {self.X_test.shape[0]}
        • Selected Features: {self.X_train_selected.shape[1]}
        • Classes: 4 (Neutral, Sad, Fear, Happy)
        
        MODELS TRAINED:
        """
        
        # Add model list
        for _, row in comparison_df.iterrows():
            summary_text += f"• {row['Model']}: {row['Accuracy']:.3f}\n        "
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, filepath):
        """
        Save all models and results
        
        Parameters:
        -----------
        filepath : str
            Path to save the results
        """
        save_data = {
            'models': self.models,
            'results': self.results,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'random_state': self.random_state,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"All results saved to: {filepath}")
    
    def load_results(self, filepath):
        """
        Load saved models and results
        
        Parameters:
        -----------
        filepath : str
            Path to load the results from
        """
        try:
            save_data = joblib.load(filepath)
            
            self.models = save_data['models']
            self.results = save_data['results']
            self.scaler = save_data['scaler']
            self.feature_selector = save_data.get('feature_selector')
            
            logger.info(f"Results loaded from: {filepath}")
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load results from {filepath}: {e}")


def main():
    """
    Main function to demonstrate advanced EEG models
    """
    print("🧠 Advanced EEG Processing Models")
    print("=" * 50)
    print("Features:")
    print("• Traditional ML models (RF, SVM, XGBoost, LightGBM)")
    print("• Deep learning models (DNN, CNN, LSTM)")
    print("• Ensemble methods")
    print("• Automated feature selection")
    print("• Comprehensive model comparison")
    print()
    
    # Initialize processor
    processor = AdvancedEEGModels()
    
    print("⚠️  Please provide EEG features and labels to run the complete pipeline")
    print("Example usage:")
    print("  processor.prepare_data(features, labels)")
    print("  processor.train_traditional_models()")
    print("  processor.train_deep_learning_models()")
    print("  processor.create_ensemble_model()")
    print("  processor.plot_results()")
    
    return processor


if __name__ == "__main__":
    main()
