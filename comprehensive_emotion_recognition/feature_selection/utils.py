"""
Feature Selection Utilities
===========================

Utility functions to work with selected features:
- Load saved feature selections
- Apply feature selection to new data
- Integrate with existing models

Author: AI Assistant
Date: July 28, 2025
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def load_latest_selection(selection_dir: str = "feature_selection_results") -> Dict:
    """
    Load the most recent feature selection results.
    
    Parameters:
    -----------
    selection_dir : str
        Directory containing feature selection results
        
    Returns:
    --------
    Dict : Selected features information
    """
    selection_path = Path(selection_dir)
    if not selection_path.exists():
        raise FileNotFoundError(f"Feature selection directory not found: {selection_dir}")
    
    # Find the most recent features file
    features_files = list(selection_path.glob("selected_features_*.joblib"))
    if not features_files:
        raise FileNotFoundError("No feature selection results found")
    
    # Get the most recent file
    latest_file = max(features_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Loading feature selection from: {latest_file}")
    selection_data = joblib.load(latest_file)
    
    logger.info(f"Selected method: {selection_data['method']}")
    logger.info(f"Selected features: {selection_data['k']}")
    logger.info(f"CV score: {selection_data['cv_score']:.4f}")
    
    return selection_data

def apply_feature_selection(X: np.ndarray, 
                          selection_data: Dict) -> np.ndarray:
    """
    Apply feature selection to data using saved selection.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix to apply selection to
    selection_data : Dict
        Feature selection information from load_latest_selection()
        
    Returns:
    --------
    np.ndarray : Selected features
    """
    selected_features = selection_data['selected_features']
    
    if X.shape[1] < max(selected_features) + 1:
        raise ValueError(f"Input has {X.shape[1]} features, but selection requires {max(selected_features) + 1}")
    
    X_selected = X[:, selected_features]
    
    logger.info(f"Applied feature selection: {X.shape[1]} → {X_selected.shape[1]} features")
    
    return X_selected

def get_feature_info(selection_data: Dict) -> Dict:
    """
    Get detailed information about selected features.
    
    Parameters:
    -----------
    selection_data : Dict
        Feature selection information
        
    Returns:
    --------
    Dict : Feature information
    """
    return {
        'method': selection_data['method'],
        'n_selected': selection_data['k'],
        'cv_score': selection_data['cv_score'],
        'selected_indices': selection_data['selected_features'],
        'feature_names': selection_data.get('feature_names', []),
        'timestamp': selection_data.get('timestamp', 'unknown')
    }

def create_feature_mask(total_features: int, 
                       selected_features: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask for feature selection.
    
    Parameters:
    -----------
    total_features : int
        Total number of features
    selected_features : np.ndarray
        Indices of selected features
        
    Returns:
    --------
    np.ndarray : Boolean mask
    """
    mask = np.zeros(total_features, dtype=bool)
    mask[selected_features] = True
    return mask
