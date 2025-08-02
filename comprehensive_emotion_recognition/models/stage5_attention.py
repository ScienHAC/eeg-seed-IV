"""
Stage 5: Advanced Deep Learning (Attention, Multi-head, Transformer)
Target Accuracy: 88-92%

This stage implements state-of-the-art attention mechanisms and advanced
architectures for superior EEG emotion recognition performance.

Author: GitHub Copilot
Date: August 2, 2025
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
import math

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
PYTORCH_AVAILABLE = True

# Standard ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Data loading
from data_processing.seed_iv_loader import SeedIVLoader

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for EEG data
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.output_projection(context)
        
        return output, attention_weights

class EEGTransformerModel(nn.Module):
    """
    Transformer model for EEG emotion recognition with spatial-temporal attention
    """
    
    def __init__(self, input_dim=310, d_model=256, num_heads=8, num_layers=6, 
                 num_classes=4, dropout=0.1):
        super(EEGTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(5, d_model)  # Each channel-frequency token has 5 features
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to create sequence: (batch, 62, 5) for channel-frequency
        x = x.view(batch_size, 62, 5)
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, 62, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
        # Global average pooling over sequence dimension
        pooled_output = transformer_output.mean(dim=0)  # (batch, d_model)
        
        # Classification
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        
        return output

class DualBranchAttentionModel(nn.Module):
    """
    Dual-branch model with spatial and temporal attention mechanisms
    """
    
    def __init__(self, input_dim=310, num_classes=4, dropout=0.1):
        super(DualBranchAttentionModel, self).__init__()
        
        # Spatial branch (CNN with attention)
        self.spatial_conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.spatial_conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.spatial_attention = MultiHeadSelfAttention(128, num_heads=8, dropout=dropout)
        
        # Temporal branch (LSTM with attention)  
        self.temporal_projection = nn.Linear(input_dim, 128)
        self.temporal_lstm = nn.LSTM(128, 128, num_layers=2, bidirectional=True, 
                                   dropout=dropout, batch_first=True)
        self.temporal_attention = MultiHeadSelfAttention(256, num_heads=8, dropout=dropout)
        
        # Fusion and classification
        self.fusion = nn.Linear(128 + 256, 256)  # Spatial + Temporal
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Spatial branch
        spatial_x = x.view(batch_size, 1, 62, 5)  # Reshape for CNN
        spatial_x = F.relu(self.spatial_conv1(spatial_x))
        spatial_x = F.relu(self.spatial_conv2(spatial_x))  # (batch, 128, 62, 5)
        
        # Reshape for attention: (batch, seq, features)
        spatial_x = spatial_x.permute(0, 2, 3, 1).contiguous()  # (batch, 62, 5, 128)
        spatial_x = spatial_x.view(batch_size, 62 * 5, 128)     # (batch, 310, 128)
        
        spatial_attended, _ = self.spatial_attention(spatial_x)
        spatial_pooled = spatial_attended.mean(dim=1)  # (batch, 128)
        
        # Temporal branch
        temporal_x = self.temporal_projection(x)  # (batch, 128)
        temporal_x = temporal_x.unsqueeze(1).repeat(1, 62, 1)  # (batch, 62, 128)
        
        temporal_lstm_out, _ = self.temporal_lstm(temporal_x)  # (batch, 62, 256)
        temporal_attended, _ = self.temporal_attention(temporal_lstm_out)
        temporal_pooled = temporal_attended.mean(dim=1)  # (batch, 256)
        
        # Fusion
        fused = torch.cat([spatial_pooled, temporal_pooled], dim=1)  # (batch, 384)
        fused = F.relu(self.fusion(fused))  # (batch, 256)
        
        # Classification
        output = self.dropout(fused)
        output = self.classifier(output)
        
        return output

class MultiScaleEEGModel(nn.Module):
    """
    Multi-scale CNN with attention for capturing features at different resolutions
    """
    
    def __init__(self, input_dim=310, num_classes=4, dropout=0.1):
        super(MultiScaleEEGModel, self).__init__()
        
        # Multi-scale convolution branches
        self.scale1_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.scale2_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.scale3_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Attention mechanism for scale fusion
        self.scale_attention = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),  # 64*3 = 192
            nn.Sigmoid()
        )
        
        # Global processing
        self.global_conv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1))
        )
        
        # Classification
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 62, 5)
        
        # Multi-scale feature extraction
        scale1_features = self.scale1_conv(x)
        scale2_features = self.scale2_conv(x)
        scale3_features = self.scale3_conv(x)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)
        
        # Apply attention for scale fusion
        attention_weights = self.scale_attention(multi_scale_features)
        attended_features = multi_scale_features * attention_weights
        
        # Global processing
        global_features = self.global_conv(attended_features)
        global_features = global_features.view(batch_size, -1)
        
        # Classification
        output = self.dropout(global_features)
        output = self.classifier(output)
        
        return output

class AdvancedDeepLearningModel:
    """
    Stage 5: Advanced Deep Learning with Attention Mechanisms
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
        Load and prepare data for advanced deep learning
        """
        logger.info("Loading SEED-IV data for Stage 5...")
        
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
        Create advanced deep learning models
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return {}
        
        models = {
            'EEG_Transformer': EEGTransformerModel(
                input_dim=310,
                d_model=256,
                num_heads=self.config.attention_heads,
                num_layers=6,
                num_classes=4,
                dropout=self.config.attention_dropout
            ),
            'Dual_Branch_Attention': DualBranchAttentionModel(
                input_dim=310,
                num_classes=4,
                dropout=0.1
            ),
            'Multi_Scale_CNN': MultiScaleEEGModel(
                input_dim=310,
                num_classes=4,
                dropout=0.1
            )
        }
        
        logger.info(f"Created {len(models)} advanced deep learning models")
        return models
    
    def train_advanced_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Train advanced models with sophisticated techniques
        """
        logger.info(f"Training {model_name}...")
        
        model = model.to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Advanced optimizer with scheduling
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0001,  # Lower learning rate for complex models
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training with advanced techniques
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(150):  # More epochs for complex models
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
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
            scheduler.step()
            
            # Early stopping with longer patience
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 25 == 0:
                logger.info(f'Epoch [{epoch+1}/150], Loss: {total_train_loss/len(train_loader):.4f}, '
                           f'Val Acc: {val_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
            
            if patience_counter >= 25:  # Longer patience for complex models
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'best_val_accuracy': best_val_acc,
            'epochs_trained': epoch + 1
        }
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate trained model
        """
        model.eval()
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
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
        
        test_accuracy = accuracy_score(y_test, all_predictions)
        test_f1 = f1_score(y_test, all_predictions, average='weighted')
        
        return {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'predictions': np.array(all_predictions),
            'prediction_probabilities': np.array(all_probabilities)
        }
    
    def run_complete_pipeline(self, data_config, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete Stage 5 pipeline
        """
        logger.info("Starting Stage 5: Advanced Deep Learning Pipeline")
        start_time = time.time()
        
        try:
            if not PYTORCH_AVAILABLE:
                return {
                    'stage_num': 5,
                    'stage_name': 'Advanced Deep Learning',
                    'error': 'PyTorch not available',
                    'processing_time': time.time() - start_time,
                    'target_accuracy': self.config.target_accuracy,
                    'accuracy': 0.0
                }
            
            # Load data
            X_train, X_test, y_train, y_test, subjects = self.load_and_prepare_data(data_config)
            
            # Data preprocessing
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create validation split
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Create and train models
            models = self.create_models()
            results = {}
            
            for name, model in models.items():
                try:
                    start_time_model = time.time()
                    
                    # Train model
                    train_result = self.train_advanced_model(
                        model, X_train_split, y_train_split, X_val_split, y_val_split, name
                    )
                    
                    # Evaluate
                    eval_result = self.evaluate_model(train_result['model'], X_test_scaled, y_test)
                    
                    results[name] = {
                        **train_result,
                        **eval_result,
                        'training_time': time.time() - start_time_model
                    }
                    
                    logger.info(f"{name} - Val: {train_result['best_val_accuracy']:.4f}, "
                               f"Test: {eval_result['test_accuracy']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    continue
            
            if not results:
                raise ValueError("No models trained successfully")
            
            # Select best model
            best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
            self.best_model = (best_name, results[best_name])
            best_result = results[best_name]
            
            # Compile final results
            final_results = {
                'stage_num': 5,
                'stage_name': 'Advanced Deep Learning',
                'target_accuracy': self.config.target_accuracy,
                'model_type': best_name,
                'accuracy': best_result['test_accuracy'],
                'f1_score': best_result['test_f1'],
                'processing_time': time.time() - start_time,
                'subjects': np.unique(subjects).tolist(),
                'model_results': results,
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
            logger.info(f"Stage 5 Target {status}: {final_results['accuracy']:.1%} vs {self.config.target_accuracy:.1%}")
            
            if save_results:
                self.save_results(final_results, data_config)
            
            logger.info(f"Stage 5 completed in {final_results['processing_time']:.1f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Stage 5 pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'stage_num': 5,
                'stage_name': 'Advanced Deep Learning', 
                'error': str(e),
                'processing_time': time.time() - start_time,
                'target_accuracy': self.config.target_accuracy,
                'accuracy': 0.0
            }
    
    def save_results(self, results, data_config):
        """Save Stage 5 results"""
        try:
            output_dir = Path(data_config.csv_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            results_file = output_dir / "stage5_advanced_dl_results.json"
            
            # Clean for JSON serialization
            clean_results = {}
            for key, value in results.items():
                if key == 'model_results':
                    clean_model_results = {}
                    for model_name, model_result in value.items():
                        clean_model_result = {k: v for k, v in model_result.items() if k != 'model'}
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
            
            logger.info(f"Stage 5 results saved to: {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save Stage 5 results: {e}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained model"""
        if not self.best_model:
            raise ValueError("No trained model available")
        
        X_test_scaled = self.scaler.transform(X_test)
        best_name, best_result = self.best_model
        model = best_result['model']
        
        eval_results = self.evaluate_model(model, X_test_scaled, y_test)
        eval_results['model_name'] = best_name
        
        return eval_results
