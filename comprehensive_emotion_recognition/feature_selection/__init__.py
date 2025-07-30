"""
Feature Selection Module
=======================

Advanced feature selection techniques for EEG emotion recognition.
This module provides various methods to select the most relevant features
from the enhanced feature set to improve model performance.

"""

from .feature_selector import (
    AdvancedFeatureSelector,
    compare_selection_methods,
    optimize_feature_count
)

from .utils import (
    load_latest_selection,
    apply_feature_selection,
    get_feature_info,
    create_feature_mask
)

__all__ = [
    'AdvancedFeatureSelector',
    'compare_selection_methods', 
    'optimize_feature_count',
    'load_latest_selection',
    'apply_feature_selection',
    'get_feature_info',
    'create_feature_mask'
]
