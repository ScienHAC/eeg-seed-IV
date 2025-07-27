"""
SEED-IV Dataset Loader and Preprocessor
======================================

Comprehensive data loading and preprocessing for SEED-IV dataset:
- Load .mat files with scipy.io
- Extract DE features (de_LDS, de_movingAve)
- Organize data by subject/session/trial
- Advanced preprocessing pipeline
- Feature engineering and selection

Author: AI Assistant
Date: July 27, 2025
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Union
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from config import DataConfig, EMOTION_LABELS

logger = logging.getLogger(__name__)

class SeedIVLoader:
    """
    Comprehensive SEED-IV dataset loader and preprocessor
    """
    
    def __init__(self, data_config: DataConfig = None, cache_dir: str = "cache"):
        """
        Initialize SEED-IV loader
        
        Parameters:
        -----------
        data_config : DataConfig
            Configuration object containing data parameters
        cache_dir : str
            Directory for caching processed data
        """
        self.data_config = data_config or DataConfig()
        self.base_path = Path(self.data_config.seed_iv_base_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Dataset properties
        self.n_subjects = self.data_config.n_subjects
        self.n_sessions = self.data_config.n_sessions
        self.n_trials = self.data_config.n_trials
        self.n_channels = self.data_config.n_channels
        self.n_frequency_bands = self.data_config.n_frequency_bands
        self.emotions = self.data_config.emotions
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.metadata = {}
        
        # Scalers
        self.scalers = {}
        
        logger.info(f"SeedIVLoader initialized with base path: {self.base_path}")
    
    def scan_dataset(self) -> Dict[str, List[str]]:
        """
        Scan the dataset directory and catalog all .mat files
        
        Returns:
        --------
        Dict[str, List[str]] : Dictionary mapping sessions to file paths
        """
        logger.info("Scanning SEED-IV dataset...")
        
        file_catalog = {}
        
        # Scan each session directory
        for session in range(1, self.n_sessions + 1):
            session_dir = self.base_path / str(session)
            if not session_dir.exists():
                logger.warning(f"Session directory not found: {session_dir}")
                continue
            
            session_files = []
            for mat_file in session_dir.glob("*.mat"):
                session_files.append(str(mat_file))
            
            file_catalog[f"session_{session}"] = sorted(session_files)
            logger.info(f"Session {session}: Found {len(session_files)} .mat files")
        
        total_files = sum(len(files) for files in file_catalog.values())
        logger.info(f"Total files found: {total_files}")
        
        self.metadata['file_catalog'] = file_catalog
        return file_catalog
    
    def load_mat_file(self, mat_file_path: str) -> Dict[str, np.ndarray]:
        """
        Load a single .mat file and extract features
        
        Parameters:
        -----------
        mat_file_path : str
            Path to the .mat file
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary of extracted features
        """
        try:
            # Load .mat file
            mat_data = loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
            
            # Remove MATLAB metadata
            features = {key: value for key, value in mat_data.items() 
                       if not key.startswith('__')}
            
            # Extract subject and session info from filename
            file_path = Path(mat_file_path)
            session = file_path.parent.name
            filename = file_path.stem
            
            # Extract subject ID (assumes format like "1_20160518")
            subject_match = re.match(r'(\d+)', filename)
            subject_id = int(subject_match.group(1)) if subject_match else None
            
            logger.debug(f"Loaded {mat_file_path}: Subject {subject_id}, Session {session}")
            logger.debug(f"Available features: {list(features.keys())}")
            
            return {
                'features': features,
                'subject_id': subject_id,
                'session': session,
                'filename': filename,
                'file_path': mat_file_path
            }
            
        except Exception as e:
            logger.error(f"Failed to load {mat_file_path}: {e}")
            return None
    
    def extract_de_features(self, mat_data: Dict, feature_types: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract Differential Entropy (DE) features from loaded .mat data
        
        Parameters:
        -----------
        mat_data : Dict
            Loaded .mat file data
        feature_types : List[str]
            Types of DE features to extract (e.g., ['de_LDS', 'de_movingAve'])
            
        Returns:
        --------
        Dict[str, np.ndarray] : Extracted DE features
        """
        if feature_types is None:
            feature_types = ['de_LDS', 'de_movingAve']  # Default feature types
        
        extracted_features = {}
        features = mat_data['features']
        
        for feature_type in feature_types:
            # Find all features matching the type (e.g., de_LDS1, de_LDS2, ...)
            matching_features = [key for key in features.keys() 
                               if key.startswith(feature_type)]
            
            if not matching_features:
                logger.warning(f"No features found for type: {feature_type}")
                continue
            
            # Extract and organize features by trial
            feature_data = {}
            for feature_key in matching_features:
                # Extract trial number (e.g., de_LDS1 -> 1)
                trial_match = re.search(r'(\d+)$', feature_key)
                if trial_match:
                    trial_num = int(trial_match.group(1))
                    feature_data[trial_num] = features[feature_key]
            
            # Sort by trial number and stack
            sorted_trials = sorted(feature_data.keys())
            trial_features = []
            
            for trial in sorted_trials:
                trial_data = feature_data[trial]
                
                # Process based on dimensions
                if trial_data.ndim == 3:
                    # Shape: (channels, time_samples, freq_bands)
                    channels, time_samples, freq_bands = trial_data.shape
                    
                    # Reshape to (time_samples, channels * freq_bands)
                    reshaped = trial_data.transpose(1, 0, 2)  # (time, channels, freq_bands)
                    reshaped = reshaped.reshape(time_samples, channels * freq_bands)
                    trial_features.append(reshaped)
                    
                elif trial_data.ndim == 2:
                    # Already in correct format
                    trial_features.append(trial_data)
                else:
                    # 1D data - expand dimensions
                    trial_features.append(trial_data.reshape(-1, 1))
            
            if trial_features:
                # Stack all trials
                stacked_features = np.vstack(trial_features)
                extracted_features[feature_type] = stacked_features
                
                logger.debug(f"Extracted {feature_type}: {stacked_features.shape} "
                           f"from {len(trial_features)} trials")
        
        return extracted_features
    
    def create_emotion_labels(self, n_samples: int) -> np.ndarray:
        """
        Create emotion labels based on SEED-IV protocol
        
        SEED-IV emotion sequence per session:
        - 24 trials with specific emotion order
        - Each trial has multiple time samples
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples to label
            
        Returns:
        --------
        np.ndarray : Emotion labels
        """
        # SEED-IV emotion sequence (24 trials)
        # This is the standard SEED-IV emotion sequence
        emotion_sequence = [
            1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1,  # Trials 1-12
            1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1   # Trials 13-24
        ]
        
        # Calculate samples per trial
        samples_per_trial = n_samples // len(emotion_sequence)
        remainder = n_samples % len(emotion_sequence)
        
        labels = []
        for i, emotion in enumerate(emotion_sequence):
            # Add extra sample to first 'remainder' trials
            trial_samples = samples_per_trial + (1 if i < remainder else 0)
            labels.extend([emotion] * trial_samples)
        
        return np.array(labels[:n_samples])
    
    def load_subject_data(self, subject_id: int, session: int = None) -> Dict[str, np.ndarray]:
        """
        Load data for a specific subject (and optionally session)
        
        Parameters:
        -----------
        subject_id : int
            Subject ID (1-15)
        session : int, optional
            Session number (1-3). If None, loads all sessions
            
        Returns:
        --------
        Dict[str, np.ndarray] : Loaded data
        """
        logger.info(f"Loading data for Subject {subject_id}" + 
                   (f", Session {session}" if session else " (all sessions)"))
        
        subject_data = {
            'features': {},
            'labels': [],
            'metadata': []
        }
        
        # Determine sessions to load
        sessions_to_load = [session] if session else range(1, self.n_sessions + 1)
        
        for sess in sessions_to_load:
            session_dir = self.base_path / str(sess)
            
            # Find .mat file for this subject and session
            pattern = f"{subject_id}_*.mat"
            mat_files = list(session_dir.glob(pattern))
            
            if not mat_files:
                logger.warning(f"No .mat file found for Subject {subject_id}, Session {sess}")
                continue
            
            mat_file = mat_files[0]  # Take first match
            
            # Load and process the file
            mat_data = self.load_mat_file(str(mat_file))
            if mat_data is None:
                continue
            
            # Extract DE features
            de_features = self.extract_de_features(mat_data)
            
            # Create labels once per session (not per feature type)
            session_labels = None
            
            for feature_type, feature_data in de_features.items():
                if feature_type not in subject_data['features']:
                    subject_data['features'][feature_type] = []
                
                subject_data['features'][feature_type].append(feature_data)
                
                # Create labels only once per session
                if session_labels is None:
                    session_labels = self.create_emotion_labels(feature_data.shape[0])
                    subject_data['labels'].extend(session_labels)
                
                # Add metadata
                metadata = {
                    'subject_id': subject_id,
                    'session': sess,
                    'feature_type': feature_type,
                    'n_samples': feature_data.shape[0],
                    'n_features': feature_data.shape[1]
                }
                subject_data['metadata'].append(metadata)
        
        # Concatenate features across sessions
        for feature_type in subject_data['features']:
            if subject_data['features'][feature_type]:
                subject_data['features'][feature_type] = np.vstack(
                    subject_data['features'][feature_type]
                )
        
        subject_data['labels'] = np.array(subject_data['labels'])
        
        logger.info(f"Loaded Subject {subject_id}: "
                   f"{subject_data['labels'].shape[0]} samples, "
                   f"{len(subject_data['features'])} feature types")
        
        return subject_data
    
    def load_all_subjects(self, feature_type: str = 'de_LDS') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data for all subjects and combine
        
        Parameters:
        -----------
        feature_type : str
            Type of features to load
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray] : (features, labels, subjects)
        """
        logger.info(f"Loading all subjects with feature type: {feature_type}")
        
        all_features = []
        all_labels = []
        all_subjects = []
        
        for subject_id in range(1, self.n_subjects + 1):
            try:
                subject_data = self.load_subject_data(subject_id)
                
                if feature_type in subject_data['features']:
                    features = subject_data['features'][feature_type]
                    labels = subject_data['labels']
                    
                    all_features.append(features)
                    all_labels.append(labels)
                    # Create subject array for this subject's samples
                    subject_array = np.full(features.shape[0], subject_id)
                    all_subjects.append(subject_array)
                    
                    logger.info(f"Subject {subject_id}: {features.shape[0]} samples")
                else:
                    logger.warning(f"Feature type {feature_type} not found for Subject {subject_id}")
                    
            except Exception as e:
                logger.error(f"Failed to load Subject {subject_id}: {e}")
        
        if all_features:
            combined_features = np.vstack(all_features)
            combined_labels = np.hstack(all_labels)
            combined_subjects = np.hstack(all_subjects)
            
            logger.info(f"Combined dataset: {combined_features.shape[0]} samples, "
                       f"{combined_features.shape[1]} features")
            
            return combined_features, combined_labels, combined_subjects
        else:
            logger.error("No data loaded!")
            return np.array([]), np.array([]), np.array([])
    
    def preprocess_features(self, features: np.ndarray, method: str = "zscore", 
                          fit_scaler: bool = True) -> np.ndarray:
        """
        Preprocess features using specified normalization method
        
        Parameters:
        -----------
        features : np.ndarray
            Input features
        method : str
            Normalization method ("zscore", "minmax", "robust")
        fit_scaler : bool
            Whether to fit the scaler (True for training data)
            
        Returns:
        --------
        np.ndarray : Preprocessed features
        """
        if method not in self.scalers:
            if method == "zscore":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            self.scalers[method] = scaler
        
        scaler = self.scalers[method]
        
        if fit_scaler:
            preprocessed = scaler.fit_transform(features)
            logger.info(f"Fitted and transformed features using {method}")
        else:
            preprocessed = scaler.transform(features)
            logger.info(f"Transformed features using fitted {method} scaler")
        
        return preprocessed
    
    def save_processed_data(self, filename: str, features: np.ndarray, 
                          labels: np.ndarray, metadata: Dict = None):
        """
        Save processed data to cache
        
        Parameters:
        -----------
        filename : str
            Filename for saved data
        features : np.ndarray
            Processed features
        labels : np.ndarray
            Labels
        metadata : Dict, optional
            Additional metadata
        """
        save_path = self.cache_dir / f"{filename}.joblib"
        
        data_to_save = {
            'features': features,
            'labels': labels,
            'metadata': metadata or {},
            'scalers': self.scalers,
            'config': self.data_config.__dict__
        }
        
        joblib.dump(data_to_save, save_path)
        logger.info(f"Saved processed data to: {save_path}")
    
    def load_processed_data(self, filename: str) -> Dict:
        """
        Load processed data from cache
        
        Parameters:
        -----------
        filename : str
            Filename of saved data
            
        Returns:
        --------
        Dict : Loaded data
        """
        load_path = self.cache_dir / f"{filename}.joblib"
        
        if load_path.exists():
            data = joblib.load(load_path)
            self.scalers = data.get('scalers', {})
            logger.info(f"Loaded processed data from: {load_path}")
            return data
        else:
            logger.error(f"File not found: {load_path}")
            return None
    
    def get_dataset_statistics(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Get comprehensive dataset statistics
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        labels : np.ndarray
            Labels
            
        Returns:
        --------
        Dict : Dataset statistics
        """
        from collections import Counter
        
        stats = {
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'feature_range': {
                'min': float(features.min()),
                'max': float(features.max()),
                'mean': float(features.mean()),
                'std': float(features.std())
            },
            'class_distribution': dict(Counter(labels)),
            'class_balance': {},
            'missing_values': int(np.isnan(features).sum())
        }
        
        # Calculate class balance
        total_samples = len(labels)
        for emotion_id, count in stats['class_distribution'].items():
            emotion_name = self.emotions.get(emotion_id, f"Unknown_{emotion_id}")
            stats['class_balance'][emotion_name] = {
                'count': count,
                'percentage': (count / total_samples) * 100
            }
        
        return stats


def main():
    """
    Demonstration of SEED-IV data loading
    """
    print("🧠 SEED-IV Dataset Loader Demonstration")
    print("=" * 60)
    
    # Initialize loader
    loader = SeedIVLoader()
    
    # Scan dataset
    file_catalog = loader.scan_dataset()
    print(f"📁 Found files in {len(file_catalog)} sessions")
    
    # Load a single subject for testing
    print("\n📊 Loading Subject 1 data...")
    subject_data = loader.load_subject_data(subject_id=1)
    
    if subject_data['features']:
        feature_type = list(subject_data['features'].keys())[0]
        features = subject_data['features'][feature_type]
        labels = subject_data['labels']
        
        print(f"✅ Loaded: {features.shape} features, {labels.shape} labels")
        
        # Preprocess features
        preprocessed = loader.preprocess_features(features, method="zscore")
        print(f"🔧 Preprocessed: {preprocessed.shape}")
        
        # Get statistics
        stats = loader.get_dataset_statistics(features, labels)
        print(f"📈 Dataset statistics:")
        print(f"   Samples: {stats['n_samples']}")
        print(f"   Features: {stats['n_features']}")
        print(f"   Classes: {list(stats['class_distribution'].keys())}")
        print(f"   Class distribution: {stats['class_distribution']}")
        
        # Save processed data
        loader.save_processed_data("subject_1_demo", preprocessed, labels, stats)
        print("💾 Saved processed data")
    else:
        print("❌ No data loaded")


if __name__ == "__main__":
    main()
