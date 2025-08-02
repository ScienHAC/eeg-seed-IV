"""
Model Loading and Feature Extraction for Validation
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Loads trained models and extracts features for validation
    """
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.feature_selectors = {}
        self.scalers = {}
        
    def load_stage_model(self, stage: int) -> Dict[str, Any]:
        """
        Load a trained model from specific stage
        
        Parameters:
        -----------
        stage : int
            Stage number (1 or 2)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing model, feature_selector, scaler, and metadata
        """
        try:
            if stage == 1:
                checkpoint_path = self.config.stage1_checkpoint_path
            elif stage == 2:
                checkpoint_path = self.config.stage2_checkpoint_path
            else:
                raise ValueError(f"Stage {stage} not supported")
            
            if not Path(checkpoint_path).exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
            
            # Load checkpoint
            checkpoint_data = joblib.load(checkpoint_path)
            logger.info(f"Loaded Stage {stage} checkpoint from {checkpoint_path}")
            
            # Extract components
            model_data = {
                'model': checkpoint_data.get('model'),
                'stage_num': checkpoint_data.get('stage_num'),
                'result': checkpoint_data.get('result', {}),
                'timestamp': checkpoint_data.get('timestamp'),
                'config': checkpoint_data.get('config', {})
            }
            
            # Store for later use
            self.models[f'stage_{stage}'] = model_data
            
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load Stage {stage} model: {e}")
            return None
    
    def extract_model_components(self, stage: int) -> Tuple[Any, Any, Any, Dict]:
        """
        Extract individual components from loaded model
        
        Parameters:
        -----------
        stage : int
            Stage number
            
        Returns:
        --------
        Tuple[Any, Any, Any, Dict]
            (trained_model, feature_selector, scaler, metadata)
        """
        model_data = self.load_stage_model(stage)
        if not model_data:
            return None, None, None, {}
        
        # For Stage 2, the model might be wrapped with feature selection
        if stage == 2:
            # Try to extract from the trained model
            trained_model = model_data['model']
            
            # Feature selector and scaler might be embedded in the pipeline
            # We'll need to reconstruct based on saved results
            feature_selector = None
            scaler = StandardScaler()  # Default scaler
            
            metadata = {
                'accuracy': model_data['result'].get('accuracy', 0),
                'f1_score': model_data['result'].get('f1_score', 0),
                'n_features': model_data['result'].get('n_features_selected', 60),
                'model_type': model_data['result'].get('model_type', 'Unknown'),
                'training_subjects': model_data['result'].get('subjects', []),
                'timestamp': model_data.get('timestamp')
            }
            
        elif stage == 1:
            trained_model = model_data['model']
            feature_selector = None  # Stage 1 uses all features
            scaler = StandardScaler()
            
            metadata = {
                'accuracy': model_data['result'].get('accuracy', 0),
                'f1_score': model_data['result'].get('f1_score', 0),
                'n_features': 310,  # All DE features
                'model_type': model_data['result'].get('model_type', 'SVM'),
                'training_subjects': model_data['result'].get('subjects', []),
                'timestamp': model_data.get('timestamp')
            }
        
        logger.info(f"Extracted Stage {stage} components:")
        logger.info(f"  - Model type: {metadata['model_type']}")
        logger.info(f"  - Training accuracy: {metadata['accuracy']:.1%}")
        logger.info(f"  - Features used: {metadata['n_features']}")
        
        return trained_model, feature_selector, scaler, metadata
    
    def get_available_models(self) -> List[int]:
        """
        Get list of available trained models
        
        Returns:
        --------
        List[int]
            List of available stage numbers
        """
        available_stages = []
        
        for stage in [1, 2]:
            if stage == 1 and Path(self.config.stage1_checkpoint_path).exists():
                available_stages.append(stage)
            elif stage == 2 and Path(self.config.stage2_checkpoint_path).exists():
                available_stages.append(stage)
        
        logger.info(f"Available trained models: Stage {available_stages}")
        return available_stages
    
    def load_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all available trained models
        ONLY loads Stage 1 and Stage 2 checkpoints (NOT saved_models)
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary mapping model names to model data
        """
        all_models = {}
        
        # DO NOT load saved_models - they have low accuracy (54%), disposed
        # Only load Stage 1 (77.64% SVM) and Stage 2 (97.7% RF) checkpoints
        
        available_stages = self.get_available_models()
        logger.info(f"Loading only Stage models (NOT saved_models): {available_stages}")
        
        for stage in available_stages:
            try:
                logger.info(f"Attempting to load Stage {stage} model...")
                trained_model, feature_selector, scaler, metadata = self.extract_model_components(stage)
                if trained_model is not None:
                    stage_name = f"stage_{stage}_model"
                    all_models[stage_name] = {
                        'model': trained_model,
                        'feature_selector': feature_selector,
                        'scaler': scaler,
                        'metadata': metadata,
                        'model_type': metadata.get('model_type', 'Unknown'),
                        'stage': stage
                    }
                    logger.info(f"Loaded Stage {stage} model: {metadata.get('model_type')} ({metadata.get('accuracy', 0):.1%})")
                else:
                    logger.warning(f"Failed to load Stage {stage} model - returned None")
            except ImportError as e:
                logger.error(f"Import error loading Stage {stage} model: {e}")
                logger.error(f"Skipping Stage {stage} due to missing dependencies")
                continue
            except Exception as e:
                logger.error(f"Failed to load stage {stage} model: {e}")
                continue
        
        if not all_models:
            logger.warning("⚠️ No models loaded! Check checkpoint paths:")
            logger.warning(f"   Stage 1: {self.config.stage1_checkpoint_path}")
            logger.warning(f"   Stage 2: {self.config.stage2_checkpoint_path}")
        
        logger.info(f"Total models loaded: {len(all_models)} (Stage models only)")
        return all_models
    
    def validate_model_compatibility(self, model_data: Dict[str, Any], test_features: np.ndarray) -> bool:
        """
        Validate that the model is compatible with test features
        
        Parameters:
        -----------
        model_data : Dict[str, Any]
            Loaded model data
        test_features : np.ndarray
            Test feature array
            
        Returns:
        --------
        bool
            True if compatible, False otherwise
        """
        try:
            model = model_data['model']
            
            # Check feature dimensionality
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                actual_features = test_features.shape[1]
                
                if expected_features != actual_features:
                    logger.error(f"Feature mismatch: model expects {expected_features}, got {actual_features}")
                    return False
            
            # Try a prediction on a small sample
            test_sample = test_features[:1]
            _ = model.predict(test_sample)
            
            logger.info("Model compatibility validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model compatibility check failed: {e}")
            return False
