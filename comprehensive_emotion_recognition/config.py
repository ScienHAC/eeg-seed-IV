"""
Configuration file for SEED-IV Emotion Recognition System
========================================================

Comprehensive configuration for all stages of emotion recognition:
- Data paths and preprocessing parameters
- Model hyperparameters for each stage
- Training configurations
- Evaluation settings

Author: AI Assistant
Date: July 26, 2025
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Dataset configuration"""
    # Data paths
    seed_iv_base_path: str = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth"
    csv_output_dir: str = "csv_data"
    processed_data_dir: str = "processed_data"
    cache_dir: str = "cache"
    
    # Dataset properties
    n_subjects: int = 15
    n_sessions: int = 3
    n_trials: int = 24
    n_channels: int = 62
    n_frequency_bands: int = 5
    sampling_rate: int = 200
    
    # Emotion labels
    emotions: Dict[int, str] = None
    n_classes: int = 4
    
    def __post_init__(self):
        if self.emotions is None:
            self.emotions = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}

@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    # Filtering
    low_freq: float = 1.0
    high_freq: float = 75.0
    notch_freq: float = 50.0
    
    # Artifact removal
    use_ica: bool = True
    n_ica_components: int = 20
    
    # Feature extraction
    features_to_extract: List[str] = None
    smoothing_methods: List[str] = None
    
    # Normalization
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust"
    
    def __post_init__(self):
        if self.features_to_extract is None:
            self.features_to_extract = ["de_LDS", "de_movingAve"]
        if self.smoothing_methods is None:
            self.smoothing_methods = ["LDS", "movingAve"]

@dataclass
class Stage1Config:
    """Stage 1: Traditional Baseline (70-75% accuracy)"""
    name: str = "Traditional_Baseline"
    target_accuracy: float = 0.75
    
    # Model parameters
    model_type: str = "SVM"
    svm_kernel: str = "linear"  # Linear is much faster than RBF
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    
    # Feature parameters
    feature_dim: int = 310  # 62 channels × 5 freq bands
    use_basic_features: bool = True
    use_feature_selection: bool = False  # Traditional baseline without selection
    scaler_type: str = "standard"  # "standard", "minmax", "robust"
    use_grid_search: bool = False  # Grid search configuration

@dataclass
class Stage2Config:
    """Stage 2: Enhanced Features (75-80% accuracy)"""
    name: str = "Enhanced_Features"
    target_accuracy: float = 0.80
    
    # Model parameters
    model_type: str = "RandomForest"
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    
    # Feature engineering
    use_spatial_features: bool = True
    use_temporal_features: bool = True
    use_connectivity_features: bool = True
    feature_selection_method: str = "optimized_medical"  # Use optimized 60 features
    n_selected_features: int = 60  # Medical-grade feature count
    
    # Hyperparameter optimization
    use_randomized_search: bool = True
    n_iter: int = 50
    
    # Preprocessing
    scaler_type: str = "standard"  # "standard", "minmax", "robust"

@dataclass
class Stage3Config:
    """Stage 3: Advanced ML (80-85% accuracy)"""
    name: str = "Advanced_ML"
    target_accuracy: float = 0.85
    
    # Model parameters
    model_type: str = "XGBoost"
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Hyperparameter optimization
    use_hyperopt: bool = True
    n_hyperopt_trials: int = 100
    
    # Feature selection
    feature_selection_methods: List[str] = None
    ensemble_feature_selection: bool = True
    
    def __post_init__(self):
        if self.feature_selection_methods is None:
            self.feature_selection_methods = ["mutual_info", "rfe", "lasso"]

@dataclass
class Stage4Config:
    """Stage 4: Deep Learning Foundation (85-88% accuracy)"""
    name: str = "Deep_Learning_Foundation"
    target_accuracy: float = 0.88
    
    # Model architectures
    architectures: List[str] = None
    
    # CNN parameters
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # LSTM parameters
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    dropout_rate: float = 0.5
    
    # Regularization
    weight_decay: float = 1e-4
    use_batch_norm: bool = True
    use_early_stopping: bool = True
    patience: int = 10
    
    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ["CNN_2D", "CNN_3D", "LSTM", "CNN_LSTM"]
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 5, 7]

@dataclass
class Stage5Config:
    """Stage 5: Advanced DL Models (88-92% accuracy)"""
    name: str = "Advanced_Deep_Learning"
    target_accuracy: float = 0.92
    
    # Advanced architectures
    use_attention: bool = True
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Hybrid models
    use_dual_branch: bool = True
    spatial_branch_type: str = "CNN"
    temporal_branch_type: str = "LSTM"
    
    # Training strategies
    use_progressive_learning: bool = True
    use_transfer_learning: bool = True
    use_domain_adaptation: bool = True
    
    # Multi-scale features
    use_multi_scale: bool = True
    scale_factors: List[int] = None
    
    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [1, 2, 4, 8]

@dataclass
class Stage6Config:
    """Stage 6: State-of-Art Models (92-96% accuracy)"""
    name: str = "State_of_Art"
    target_accuracy: float = 0.96
    
    # Vision Transformer
    use_vision_transformer: bool = True
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_num_heads: int = 12
    vit_num_layers: int = 12
    
    # Multi-modal fusion
    use_multimodal: bool = True
    modalities: List[str] = None
    
    # Ensemble methods
    ensemble_models: List[str] = None
    ensemble_method: str = "weighted_voting"  # "voting", "stacking", "bagging"
    
    # Advanced optimization
    use_neural_architecture_search: bool = False
    use_knowledge_distillation: bool = True
    use_uncertainty_quantification: bool = True
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["EEG", "EOG"]
        if self.ensemble_models is None:
            self.ensemble_models = ["ViT", "CNN_LSTM_Attention", "TransformerModel"]

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Cross-validation
    cv_strategy: str = "subject_independent"  # "subject_independent", "session_independent", "within_subject"
    n_folds: int = 5
    
    # Data splitting
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Training parameters
    random_seed: int = 42
    device: str = "cuda"  # "cuda", "cpu", "auto"
    num_workers: int = 4
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    save_best_only: bool = True
    
    # Logging
    log_frequency: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics
    primary_metric: str = "accuracy"
    metrics: List[str] = None
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_learning_curves: bool = True
    plot_feature_importance: bool = True
    
    # Statistical analysis
    perform_statistical_tests: bool = True
    significance_level: float = 0.05
    
    # Output
    save_predictions: bool = True
    save_models: bool = True
    generate_report: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]

class ComprehensiveConfig:
    """Main configuration class combining all stage configurations"""
    
    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.stage1 = Stage1Config()
        self.stage2 = Stage2Config()
        self.stage3 = Stage3Config()
        self.stage4 = Stage4Config()
        self.stage5 = Stage5Config()
        self.stage6 = Stage6Config()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        base_dir = Path(".")
        dirs_to_create = [
            self.data.csv_output_dir,
            self.data.processed_data_dir,
            self.data.cache_dir,
            "logs",
            "checkpoints",
            "results",
            "reports"
        ]
        
        for dir_name in dirs_to_create:
            dir_path = base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
    
    def get_stage_config(self, stage_number: int):
        """Get configuration for specific stage"""
        stage_configs = {
            1: self.stage1,
            2: self.stage2,
            3: self.stage3,
            4: self.stage4,
            5: self.stage5,
            6: self.stage6
        }
        return stage_configs.get(stage_number)
    
    def update_data_path(self, new_path: str):
        """Update the base data path"""
        self.data.seed_iv_base_path = new_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'data': self.data.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'stage1': self.stage1.__dict__,
            'stage2': self.stage2.__dict__,
            'stage3': self.stage3.__dict__,
            'stage4': self.stage4.__dict__,
            'stage5': self.stage5.__dict__,
            'stage6': self.stage6.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__
        }

# Global configuration instance
config = ComprehensiveConfig()

# Constants
EMOTION_LABELS = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Stage progression mapping
STAGE_PROGRESSION = {
    1: {"name": "Traditional Baseline", "target_accuracy": 0.75, "duration_weeks": 2},
    2: {"name": "Enhanced Features", "target_accuracy": 0.80, "duration_weeks": 3},
    3: {"name": "Advanced ML", "target_accuracy": 0.85, "duration_weeks": 4},
    4: {"name": "Deep Learning Foundation", "target_accuracy": 0.88, "duration_weeks": 6},
    5: {"name": "Advanced Deep Learning", "target_accuracy": 0.92, "duration_weeks": 8},
    6: {"name": "State-of-Art Models", "target_accuracy": 0.96, "duration_weeks": 12}
}

def print_config_summary():
    """Print a summary of the current configuration"""
    print("🧠 SEED-IV Emotion Recognition - Configuration Summary")
    print("=" * 60)
    print(f"Data Path: {config.data.seed_iv_base_path}")
    print(f"Subjects: {config.data.n_subjects}")
    print(f"Sessions: {config.data.n_sessions}")
    print(f"Trials: {config.data.n_trials}")
    print(f"Channels: {config.data.n_channels}")
    print(f"Classes: {config.data.n_classes}")
    print(f"Cross-validation: {config.training.cv_strategy}")
    print(f"Random Seed: {config.training.random_seed}")
    print("\nStage Targets:")
    for stage_num, stage_info in STAGE_PROGRESSION.items():
        stage_config = config.get_stage_config(stage_num)
        print(f"  Stage {stage_num}: {stage_info['name']} "
              f"(Target: {stage_info['target_accuracy']:.1%}, "
              f"Duration: {stage_info['duration_weeks']} weeks)")

if __name__ == "__main__":
    print_config_summary()
