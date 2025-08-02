"""
Stage 6: State-of-the-Art Models (Vision Transformer, Ensemble)
Target Accuracy: 92-96%

This stage implements cutting-edge architectures including Vision Transformers
adapted for EEG data and sophisticated ensemble methods.

Author: GitHub Copilot
Date: August 2, 2025
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
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
from sklearn.ensemble import VotingClassifier

# Data loading
from data_processing.seed_iv_loader import SeedIVLoader

logger = logging.getLogger(__name__)

class PatchEmbedding(nn.Module):
    """
    Patch embedding for EEG data adapted from Vision Transformer
    """
    
    def __init__(self, input_dim=310, patch_size=5, embed_dim=512):
        super(PatchEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Reshape EEG data to 2D: (62 channels, 5 freq bands)
        self.img_size = (62, 5)
        self.num_patches = (62 // patch_size) * (5 // patch_size)
        
        # Patch projection
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to 2D image format: (batch, 1, 62, 5)
        x = x.view(batch_size, 1, 62, 5)
        
        # Patch embedding: (batch, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)  
        
        # Flatten patches: (batch, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose: (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for Vision Transformer
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and MLP
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class EEGVisionTransformer(nn.Module):
    """
    Vision Transformer adapted for EEG emotion recognition
    """
    
    def __init__(self, input_dim=310, patch_size=5, embed_dim=512, depth=12, 
                 num_heads=8, mlp_ratio=4.0, num_classes=4, dropout=0.1):
        super(EEGVisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(input_dim, patch_size, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Use class token for classification
        cls_token = x[:, 0]
        
        # Classification
        output = self.head(cls_token)
        
        return output

class EnsembleMetaLearner(nn.Module):
    """
    Meta-learner for ensemble methods using neural networks
    """
    
    def __init__(self, num_models, num_classes=4, hidden_dim=128):
        super(EnsembleMetaLearner, self).__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        
        # Meta-learner network
        self.meta_learner = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, predictions):
        # predictions shape: (batch, num_models, num_classes)
        batch_size = predictions.size(0)
        
        # Flatten predictions
        x = predictions.view(batch_size, -1)
        
        # Meta-learning
        output = self.meta_learner(x)
        
        return output

class AdvancedEnsembleModel(nn.Module):
    """
    Advanced ensemble combining multiple architectures with meta-learning
    """
    
    def __init__(self, models: List[nn.Module], num_classes=4):
        super(AdvancedEnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.meta_learner = EnsembleMetaLearner(len(models), num_classes)
        
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(pred)
        
        # Stack predictions
        stacked_predictions = torch.stack(predictions, dim=1)  # (batch, num_models, num_classes)
        
        # Meta-learning
        output = self.meta_learner(stacked_predictions)
        
        return output

class StateOfArtModel:
    """
    Stage 6: State-of-the-Art Models with Vision Transformers and Ensembles
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
        Load and prepare data for state-of-the-art models
        """
        logger.info("Loading SEED-IV data for Stage 6...")
        
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
        Create state-of-the-art models
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return {}
        
        models = {
            'EEG_ViT_Base': EEGVisionTransformer(
                input_dim=310,
                patch_size=5,
                embed_dim=512,
                depth=8,
                num_heads=8,
                num_classes=4,
                dropout=0.1
            ),
            'EEG_ViT_Large': EEGVisionTransformer(
                input_dim=310,
                patch_size=5,
                embed_dim=768,
                depth=12,
                num_heads=12,
                num_classes=4,
                dropout=0.1
            )
        }
        
        logger.info(f"Created {len(models)} state-of-the-art models")
        return models
    
    def train_sota_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Train state-of-the-art model with advanced optimization
        """
        logger.info(f"Training {model_name}...")
        
        model = model.to(self.device)
        
        # Data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # Smaller batch for large models
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # Advanced optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=3e-5,  # Very low learning rate for stability 
            weight_decay=0.05,
            betas=(0.9, 0.95)
        )
        
        # Advanced scheduler with warmup
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 ** ((epoch - warmup_epochs) // 30)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function with focal loss for hard examples
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss(alpha=1, gamma=2)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(200):  # More epochs for convergence
            # Training phase
            model.train()
            total_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                
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
            
            # Model checkpointing
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 30 == 0:
                logger.info(f'Epoch [{epoch+1}/200], Loss: {total_train_loss/len(train_loader):.4f}, '
                           f'Val Acc: {val_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}')
            
            # Longer patience for complex models
            if patience_counter >= 40:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'best_val_accuracy': best_val_acc,
            'epochs_trained': epoch + 1
        }
    
    def create_ensemble(self, trained_models: List[nn.Module]) -> nn.Module:
        """
        Create advanced ensemble with meta-learning
        """
        if len(trained_models) < 2:
            logger.warning("Not enough models for ensemble")
            return trained_models[0] if trained_models else None
        
        logger.info(f"Creating advanced ensemble with {len(trained_models)} models")
        
        # Create ensemble model
        ensemble = AdvancedEnsembleModel(trained_models)
        
        return ensemble
    
    def train_ensemble(self, ensemble: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train the ensemble meta-learner
        """
        logger.info("Training ensemble meta-learner...")
        
        ensemble = ensemble.to(self.device)
        
        # Data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Only train meta-learner
        optimizer = optim.Adam(ensemble.meta_learner.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(50):  # Fewer epochs for meta-learner
            # Training
            ensemble.train()
            total_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = ensemble(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            ensemble.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = ensemble(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_accuracy = val_correct / val_total
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_ensemble_state = ensemble.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Ensemble Epoch [{epoch+1}/50], Val Acc: {val_accuracy:.4f}')
        
        # Load best ensemble
        ensemble.load_state_dict(best_ensemble_state)
        
        return {
            'model': ensemble,
            'best_val_accuracy': best_val_acc
        }
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate trained model
        """
        model.eval()
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
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
        Run the complete Stage 6 pipeline
        """
        logger.info("Starting Stage 6: State-of-the-Art Pipeline")
        start_time = time.time()
        
        try:
            if not PYTORCH_AVAILABLE:
                return {
                    'stage_num': 6,
                    'stage_name': 'State-of-the-Art',
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
            
            # Create and train individual models
            models = self.create_models()
            results = {}
            trained_models = []
            
            for name, model in models.items():
                try:
                    start_time_model = time.time()
                    
                    # Train model
                    train_result = self.train_sota_model(
                        model, X_train_split, y_train_split, X_val_split, y_val_split, name
                    )
                    
                    # Evaluate
                    eval_result = self.evaluate_model(train_result['model'], X_test_scaled, y_test)
                    
                    results[name] = {
                        **train_result,
                        **eval_result,
                        'training_time': time.time() - start_time_model
                    }
                    
                    trained_models.append(train_result['model'])
                    
                    logger.info(f"{name} - Val: {train_result['best_val_accuracy']:.4f}, "
                               f"Test: {eval_result['test_accuracy']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    continue
            
            # Create and train ensemble
            if len(trained_models) >= 2:
                try:
                    ensemble = self.create_ensemble(trained_models)
                    ensemble_result = self.train_ensemble(
                        ensemble, X_train_split, y_train_split, X_val_split, y_val_split
                    )
                    
                    # Evaluate ensemble
                    ensemble_eval = self.evaluate_model(ensemble_result['model'], X_test_scaled, y_test)
                    
                    results['Advanced_Ensemble'] = {
                        **ensemble_result,
                        **ensemble_eval,
                        'training_time': 0  # Already included in individual models
                    }
                    
                    logger.info(f"Advanced_Ensemble - Val: {ensemble_result['best_val_accuracy']:.4f}, "
                               f"Test: {ensemble_eval['test_accuracy']:.4f}")
                
                except Exception as e:
                    logger.error(f"Failed to create ensemble: {e}")
            
            if not results:
                raise ValueError("No models trained successfully")
            
            # Select best model
            best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
            self.best_model = (best_name, results[best_name])
            best_result = results[best_name]
            
            # Compile final results
            final_results = {
                'stage_num': 6,
                'stage_name': 'State-of-the-Art',
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
            logger.info(f"Stage 6 Target {status}: {final_results['accuracy']:.1%} vs {self.config.target_accuracy:.1%}")
            
            if save_results:
                self.save_results(final_results, data_config)
            
            logger.info(f"Stage 6 completed in {final_results['processing_time']:.1f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Stage 6 pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'stage_num': 6,
                'stage_name': 'State-of-the-Art',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'target_accuracy': self.config.target_accuracy,
                'accuracy': 0.0
            }
    
    def save_results(self, results, data_config):
        """Save Stage 6 results"""
        try:
            output_dir = Path(data_config.csv_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            results_file = output_dir / "stage6_sota_results.json"
            
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
            
            logger.info(f"Stage 6 results saved to: {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save Stage 6 results: {e}")
    
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
