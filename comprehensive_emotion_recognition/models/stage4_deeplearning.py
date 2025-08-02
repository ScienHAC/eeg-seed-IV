"""
Stage 4: Deep Learning Foundation (CNN, LSTM, CNN-LSTM)
Target Accuracy: 85-88%

This stage implements fundamental deep learning architectures optimized for EEG
emotion recognition, with advanced regularization to overcome overfitting.

Author: GitHub Copilot
Date: August 2, 2025
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
PYTORCH_AVAILABLE = True

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Standard ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Data loading
from data_processing.seed_iv_loader import SeedIVLoader

logger = logging.getLogger(__name__)

class EEGCNNModel(nn.Module):
    """
    CNN model for EEG emotion recognition with regularization
    """
    
    def __init__(self, input_dim: int = 310, num_classes: int = 4, dropout_rate: float = 0.5):
        super(EEGCNNModel, self).__init__()
        
        # Reshape input to 2D: (batch, 1, 62, 5) for spatial-spectral processing
        self.input_dim = input_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Reshape to 2D: (batch, 1, 62, 5)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 62, 5)
        
        # Conv layers with regularization
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = self.dropout(x)
        
        # Flatten and FC layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class EEGLSTMModel(nn.Module):
    """
    LSTM model for EEG temporal pattern recognition
    """
    
    def __init__(self, input_dim: int = 310, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 4, dropout_rate: float = 0.5):
        super(EEGLSTMModel, self).__init__()
        
        # Reshape input to sequence: (batch, seq_len, features)
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add sequence dimension by reshaping features
        # Convert (batch, 310) to (batch, 62, 5) for channel-frequency sequence
        x = x.view(batch_size, 62, 5)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        x = torch.mean(attn_out, dim=1)
        
        # Final classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class EEGCNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model combining spatial and temporal processing
    """
    
    def __init__(self, input_dim: int = 310, num_classes: int = 4, dropout_rate: float = 0.5):
        super(EEGCNNLSTMModel, self).__init__()
        
        # CNN branch for spatial features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # LSTM branch for temporal features
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            dropout=dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        
        # Fusion layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 2, 256)  # bidirectional LSTM
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch, 1, 62, 5)
        x = x.view(batch_size, 1, 62, 5)
        
        # CNN processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1, 3)  # (batch, 62, 64, 5)
        x = x.contiguous().view(batch_size, 62, -1)  # (batch, 62, 64*5)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        x = torch.mean(lstm_out, dim=1)
        
        # Final classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class DeepLearningModel:
    """
    Stage 4: Deep Learning Foundation with multiple architectures
    """
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
    
    def load_and_prepare_data(self, data_config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load and prepare data for deep learning
        """
        logger.info("Loading SEED-IV data for Stage 4...")
        
        try:
            loader = SeedIVLoader(data_config)
            
            # Load data using the correct method
            features, labels, subjects = loader.load_all_subjects(
                feature_type='de_LDS'
            )
            
            if len(features) == 0:
                raise ValueError("No data loaded")
            
            X = features
            y = labels
            # subjects is already available from the load_all_subjects call
            
            logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Subject-independent split
            unique_subjects = np.unique(subjects)
            n_subjects = len(unique_subjects)
            n_train_subjects = max(1, int(0.7 * n_subjects))  # Ensure at least 1 training subject
            
            train_subjects = unique_subjects[:n_train_subjects]
            test_subjects = unique_subjects[n_train_subjects:]
            
            train_mask = np.isin(subjects, train_subjects)
            test_mask = np.isin(subjects, test_subjects)
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            logger.info(f"Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")
            
            return X_train, X_test, y_train, y_test, subjects
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def create_models(self) -> Dict[str, nn.Module]:
        """
        Create deep learning models
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot create deep learning models")
            return {}
        
        models = {
            'CNN_2D': EEGCNNModel(
                input_dim=310,
                num_classes=4,
                dropout_rate=self.config.dropout_rate
            ),
            'LSTM_Attention': EEGLSTMModel(
                input_dim=310,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                num_classes=4,
                dropout_rate=self.config.dropout_rate
            ),
            'CNN_LSTM_Hybrid': EEGCNNLSTMModel(
                input_dim=310,
                num_classes=4,
                dropout_rate=self.config.dropout_rate
            )
        }
        
        logger.info(f"Created {len(models)} deep learning models")
        return models
    
    def train_pytorch_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Train a PyTorch model with advanced techniques
        """
        logger.info(f"Training {model_name}...")
        
        # Move model to device
        model = model.to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer with weight decay
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with early stopping
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_accuracy = val_correct / val_total
            avg_train_loss = total_train_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_accuracy)
            
            # Early stopping check
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                logger.info(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                           f'Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # Early stopping
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'best_val_accuracy': best_val_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'epochs_trained': epoch + 1
        }
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a trained PyTorch model
        """
        model.eval()
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, all_predictions)
        test_f1 = f1_score(y_test, all_predictions, average='weighted')
        
        return {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'predictions': np.array(all_predictions),
            'prediction_probabilities': np.array(all_probabilities)
        }
    
    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate all deep learning models
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot train deep learning models")
            return {}
        
        logger.info("Training Stage 4: Deep Learning Models")
        
        # Data preprocessing
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create validation split from training data
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create models
        models = self.create_models()
        results = {}
        
        for name, model in models.items():
            try:
                start_time = time.time()
                
                # Train model
                train_result = self.train_pytorch_model(
                    model, X_train_split, y_train_split, X_val_split, y_val_split, name
                )
                
                # Evaluate on test set
                eval_result = self.evaluate_model(train_result['model'], X_test_scaled, y_test)
                
                # Combine results
                results[name] = {
                    **train_result,
                    **eval_result,
                    'training_time': time.time() - start_time
                }
                
                logger.info(f"{name} - Val: {train_result['best_val_accuracy']:.4f}, "
                           f"Test: {eval_result['test_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                continue
        
        self.results = results
        
        # Select best model
        if results:
            best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
            self.best_model = (best_name, results[best_name])
            
            logger.info(f"Stage 4 best model: {best_name} "
                       f"(Accuracy: {results[best_name]['test_accuracy']:.4f})")
        
        return results
    
    def run_complete_pipeline(self, data_config, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete Stage 4 pipeline
        """
        logger.info("Starting Stage 4: Deep Learning Pipeline")
        start_time = time.time()
        
        try:
            # Check PyTorch availability
            if not PYTORCH_AVAILABLE:
                return {
                    'stage_num': 4,
                    'stage_name': 'Deep Learning Foundation',
                    'error': 'PyTorch not available - cannot run deep learning models',
                    'processing_time': time.time() - start_time,
                    'target_accuracy': self.config.target_accuracy,
                    'accuracy': 0.0
                }
            
            # Load and prepare data
            X_train, X_test, y_train, y_test, subjects = self.load_and_prepare_data(data_config)
            
            # Train and evaluate models
            model_results = self.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            if not model_results:
                raise ValueError("No models trained successfully")
            
            # Get best model results
            best_name, best_result = self.best_model
            
            # Compile final results
            final_results = {
                'stage_num': 4,
                'stage_name': 'Deep Learning Foundation',
                'target_accuracy': self.config.target_accuracy,
                'model_type': best_name,
                'accuracy': best_result['test_accuracy'],
                'f1_score': best_result['test_f1'],
                'processing_time': time.time() - start_time,
                'subjects': np.unique(subjects).tolist(),
                'model_results': model_results,
                'data_shape': {
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'n_features': X_train.shape[1]
                },
                'device_used': str(self.device)
            }
            
            # Check target achievement
            target_achieved = final_results['accuracy'] >= self.config.target_accuracy
            final_results['target_achieved'] = target_achieved
            
            status = "ACHIEVED" if target_achieved else "NOT ACHIEVED"
            logger.info(f"Stage 4 Target {status}: {final_results['accuracy']:.1%} vs {self.config.target_accuracy:.1%}")
            
            # Save results if requested
            if save_results:
                self.save_results(final_results, data_config)
            
            logger.info(f"Stage 4 completed successfully in {final_results['processing_time']:.1f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Stage 4 pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'stage_num': 4,
                'stage_name': 'Deep Learning Foundation',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'target_accuracy': self.config.target_accuracy,
                'accuracy': 0.0
            }
    
    def save_results(self, results: Dict[str, Any], data_config) -> None:
        """
        Save Stage 4 results
        """
        try:
            output_dir = Path(data_config.csv_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results (without model objects)
            import json
            results_file = output_dir / "stage4_deep_learning_results.json"
            
            # Clean results for JSON serialization
            clean_results = {}
            for key, value in results.items():
                if key == 'model_results':
                    # Remove model objects from nested results
                    clean_model_results = {}
                    for model_name, model_result in value.items():
                        clean_model_result = {k: v for k, v in model_result.items() 
                                            if k != 'model' and not k.endswith('_losses')}
                        if hasattr(clean_model_result.get('predictions'), 'tolist'):
                            clean_model_result['predictions'] = clean_model_result['predictions'].tolist()
                        clean_model_results[model_name] = clean_model_result
                    clean_results[key] = clean_model_results
                elif hasattr(value, 'tolist'):
                    clean_results[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    clean_results[key] = float(value)
                else:
                    clean_results[key] = value
            
            with open(results_file, 'w') as f:
                json.dump(clean_results, f, indent=2, default=str)
            
            logger.info(f"Stage 4 results saved to: {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save Stage 4 results: {e}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data
        """
        if not self.best_model:
            raise ValueError("No trained model available")
        
        logger.info("Evaluating Stage 4 model...")
        
        # Apply same preprocessing as training
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get best model
        best_name, best_result = self.best_model
        model = best_result['model']
        
        # Evaluate
        eval_results = self.evaluate_model(model, X_test_scaled, y_test)
        eval_results['model_name'] = best_name
        
        logger.info(f"Stage 4 evaluation - Accuracy: {eval_results['test_accuracy']:.1%}")
        
        return eval_results
