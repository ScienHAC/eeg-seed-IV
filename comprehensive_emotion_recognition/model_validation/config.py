"""
Configuration for Model Validation System
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

class ValidationConfig:
    """Configuration for model validation"""
    
    def __init__(self):
        # Paths
        self.base_dir = str(Path(__file__).parent.parent)
        self.csv_data_dir = "csv_data"
        self.checkpoints_dir = "csv_data/checkpoints"
        
        # Data directory - MATLAB files location
        # User's MATLAB files are in: C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth
        self.matlab_data_dir = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth"
        
        # For validation, we can also use the converted CSV files if available
        project_root = Path(self.base_dir).parent  # Go up to main project directory
        self.data_dir = str(project_root / "csv")  # Fallback to CSV if needed
        
        # csv_data contains processed results and checkpoints
        self.csv_data_dir = "csv_data"
        self.checkpoints_dir = "csv_data/checkpoints"
        
        
        # DO NOT use saved_models - low accuracy (54%), disposed
        # Only use Stage 1 (77.64%) and Stage 2 (97.7%) checkpoints
        self.model_dir = None  # Not using saved_models
        
        # Checkpoint paths for Stage 1 and Stage 2 models
        self.stage1_checkpoint_path = str(Path(self.base_dir) / "csv_data" / "checkpoints" / "stage_1_checkpoint.joblib")
        self.stage2_checkpoint_path = str(Path(self.base_dir) / "csv_data" / "checkpoints" / "stage_2_checkpoint.joblib")
        
        self.validation_output_dir = str(Path(self.base_dir) / "model_validation" / "results")
        
        # Model validation settings
        self.test_subjects = [13, 14, 15]  # Use unseen subjects
        self.validation_split = 0.3
        self.random_state = 42
        self.n_jobs = -1  # Use all available cores
        
        # Feature settings
        self.n_features_to_test = [30, 60, 100, 150]
        self.feature_type = "de_LDS"  # or "de_movingAve"
        
        # Evaluation settings
        self.cv_folds = 5
        self.confidence_threshold = 0.5
        
        # Visualization settings
        self.plot_style = "default"
        self.figure_size = (12, 8)
        self.dpi = 300
        
        # Create output directory
        output_path = Path(self.validation_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    def get_test_subjects_for_validation(self) -> List[int]:
        """
        Get subjects that should be used for validation testing
        These are subjects NOT used in training
        """
        return self.test_subjects
    
    def get_model_save_path(self, model_name: str) -> str:
        """Get path for saving a specific model"""
        return os.path.join(self.model_dir, f"{model_name}.joblib")
    
    def get_results_dir(self, create: bool = True) -> str:
        """Get results directory path"""
        if create:
            os.makedirs(self.validation_output_dir, exist_ok=True)
        return self.validation_output_dir
