"""
Model Validation Package for EEG Emotion Recognition

This package provides comprehensive validation tools to test the generalizability
and robustness of trained emotion classification models on unseen SEED-IV data.

Key Features:
- Load and test existing .joblib models without modification
- Validate on unseen subjects/sessions
- Comprehensive overfitting analysis
- Generate detailed validation reports
- Visualizations and statistical analysis

Author: Research Team
Date: August 2025
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .config import ValidationConfig
from .model_loader import ModelLoader
from .data_loader import UnseenDataLoader
from .validation_engine import ValidationEngine
from .report_generator import ValidationReportGenerator

__all__ = [
    'ValidationConfig',
    'ModelLoader', 
    'UnseenDataLoader',
    'ValidationEngine',
    'ValidationReportGenerator'
]
