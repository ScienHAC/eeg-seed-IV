"""
SEED-IV Advanced EEG Processing Pipeline
=====================================

Direct .mat file processing with proper preprocessing, heat map generation,
and advanced sequential methods for emotion recognition.

Based on SEED-IV dataset structure:
- 15 subjects, 3 sessions each, 24 trials per session
- 62 EEG channels, 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- 4 emotion categories: Neutral (0), Sad (1), Fear (2), Happy (3)

Features:
- Direct .mat file loading with scipy
- Advanced preprocessing pipeline
- Heat map visualization and flattening
- Sequential feature selection
- Joblib state persistence
- Multiple model architectures (CNN, RNN, Traditional ML)

Author: AI Assistant
Date: July 26, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core scientific libraries
import scipy.io
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SequentialFeatureSelector
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Machine Learning models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (optional)
try:
    import tensorflow as tf
    import keras
    from keras import layers
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization
    from keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using sklearn models only")

# Persistence and utilities
import joblib
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeedIVProcessor:
    """
    Advanced SEED-IV EEG data processor with comprehensive pipeline
    """
    
    def __init__(self, data_path=None, cache_dir="cache", random_state=42):
        """
        Initialize the SEED-IV processor
        
        Parameters:
        -----------
        data_path : str or Path
            Path to SEED-IV .mat files directory
        cache_dir : str
            Directory for caching processed data
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = Path(data_path) if data_path else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # Dataset configuration
        self.n_subjects = 15
        self.n_sessions = 3
        self.n_trials = 24
        self.n_channels = 62
        self.n_freq_bands = 5
        self.emotions = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        
        # SEED-IV official emotion labels for each trial
        self.session_labels = {
            1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
            3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
        }
        
        # Frequency bands (Hz)
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8), 
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 50)
        }
        
        # EEG channel positions (simplified 10-20 system)
        self.channel_positions = self._get_channel_positions()
        
        # Processing state
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.labels = None
        self.heat_maps = None
        self.models = {}
        
        # Create results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"SeedIVProcessor initialized with cache dir: {self.cache_dir}")
    
    def _get_channel_positions(self):
        """
        Get approximate 2D positions for 62 EEG channels for heat map visualization
        Returns mapping of channel index to (x, y) coordinates
        """
        # Simplified 62-channel layout (approximate positions)
        # This is a basic layout - for precise analysis, use actual electrode coordinates
        positions = {}
        
        # Create a rough 10x8 grid layout
        rows, cols = 10, 8
        for i in range(self.n_channels):
            row = i // cols
            col = i % cols
            # Normalize to 0-1 range
            x = col / (cols - 1)
            y = row / (rows - 1)
            positions[i] = (x, y)
        
        return positions
    
    def load_mat_data(self, session=1, subject=1, feature_type='de_LDS'):
        """
        Load EEG data directly from .mat files
        
        Parameters:
        -----------
        session : int
            Session number (1-3)
        subject : int  
            Subject number (1-15)
        feature_type : str
            Type of features ('de_LDS' or 'de_movingAve')
            
        Returns:
        --------
        dict : Dictionary containing loaded data for all trials
        """
        if not self.data_path:
            raise ValueError("Data path not specified. Please provide path to SEED-IV directory.")
        
        # Look for .mat files in the specified directory structure
        # Assuming structure: data_path/session/subject/filename.mat
        session_path = self.data_path / f"{session}" / f"{subject}"
        
        if not session_path.exists():
            logger.warning(f"Session path not found: {session_path}")
            return {}
        
        data = {}
        
        # Load each trial
        for trial in range(1, self.n_trials + 1):
            mat_file = session_path / f"{feature_type}{trial}.mat"
            
            if mat_file.exists():
                try:
                    mat_data = scipy.io.loadmat(str(mat_file))
                    
                    # Remove MATLAB metadata
                    clean_data = {k: v for k, v in mat_data.items() 
                                if not k.startswith('__')}
                    
                    # Get the main data array (usually the largest non-metadata item)
                    if clean_data:
                        main_key = max(clean_data.keys(), key=lambda k: clean_data[k].size)
                        trial_data = clean_data[main_key]
                        
                        # Store with emotion label
                        emotion_label = self.session_labels[session][trial - 1]
                        data[trial] = {
                            'data': trial_data,
                            'emotion': emotion_label,
                            'session': session,
                            'subject': subject,
                            'trial': trial
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to load {mat_file}: {e}")
            else:
                logger.warning(f"Mat file not found: {mat_file}")
        
        return data
    
    def load_all_data(self, max_subjects=15, feature_type='de_LDS'):
        """
        Load all SEED-IV data from .mat files
        
        Parameters:
        -----------
        max_subjects : int
            Maximum number of subjects to load
        feature_type : str
            Type of features to load
            
        Returns:
        --------
        dict : Complete dataset organized by session/subject/trial
        """
        cache_file = self.cache_dir / f"raw_data_{feature_type}_{max_subjects}.joblib"
        
        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return joblib.load(cache_file)
        
        logger.info(f"Loading SEED-IV data: {max_subjects} subjects, {feature_type} features")
        
        all_data = {}
        total_trials = 0
        
        for session in range(1, self.n_sessions + 1):
            all_data[session] = {}
            
            for subject in range(1, min(max_subjects + 1, self.n_subjects + 1)):
                logger.info(f"Loading Session {session}, Subject {subject}")
                
                subject_data = self.load_mat_data(session, subject, feature_type)
                
                if subject_data:
                    all_data[session][subject] = subject_data
                    total_trials += len(subject_data)
                else:
                    logger.warning(f"No data loaded for Session {session}, Subject {subject}")
        
        logger.info(f"Total trials loaded: {total_trials}")
        
        # Cache the loaded data
        joblib.dump(all_data, cache_file)
        logger.info(f"Data cached to {cache_file}")
        
        self.raw_data = all_data
        return all_data
    
    def preprocess_eeg_data(self, data_array, sampling_rate=200):
        """
        Advanced EEG preprocessing pipeline
        
        Parameters:
        -----------
        data_array : np.ndarray
            Raw EEG data (channels x time) or (channels x time x freq_bands)
        sampling_rate : int
            Sampling rate in Hz
            
        Returns:
        --------
        np.ndarray : Preprocessed EEG data
        """
        if data_array.ndim == 2:
            # Shape: (channels, time)
            preprocessed = data_array.copy()
            
            # 1. Remove baseline (mean centering)
            preprocessed = preprocessed - np.mean(preprocessed, axis=1, keepdims=True)
            
            # 2. Z-score normalization per channel
            preprocessed = zscore(preprocessed, axis=1)
            
            # 3. Outlier removal (clip to 3 standard deviations)
            preprocessed = np.clip(preprocessed, -3, 3)
            
        elif data_array.ndim == 3:
            # Shape: (channels, time, freq_bands) or (time, channels, freq_bands)
            if data_array.shape[1] == self.n_channels:
                # (time, channels, freq_bands) -> (channels, time, freq_bands)
                data_array = np.transpose(data_array, (1, 0, 2))
            
            preprocessed = data_array.copy()
            
            # Process each frequency band separately
            for freq_idx in range(preprocessed.shape[2]):
                freq_data = preprocessed[:, :, freq_idx]
                
                # Apply same preprocessing as 2D case
                freq_data = freq_data - np.mean(freq_data, axis=1, keepdims=True)
                freq_data = zscore(freq_data, axis=1)
                freq_data = np.clip(freq_data, -3, 3)
                
                preprocessed[:, :, freq_idx] = freq_data
        
        else:
            logger.warning(f"Unexpected data shape: {data_array.shape}")
            preprocessed = data_array
        
        # Handle NaN and infinite values
        preprocessed = np.nan_to_num(preprocessed, nan=0.0, posinf=0.0, neginf=0.0)
        
        return preprocessed
    
    def extract_advanced_features(self, preprocessed_data):
        """
        Extract comprehensive features from preprocessed EEG data
        
        Parameters:
        -----------
        preprocessed_data : np.ndarray
            Preprocessed EEG data
            
        Returns:
        --------
        np.ndarray : Feature vector
        """
        features = []
        
        if preprocessed_data.ndim == 2:
            # Shape: (channels, time)
            channels, time_points = preprocessed_data.shape
            
            for ch in range(channels):
                ch_data = preprocessed_data[ch, :]
                
                # Statistical features
                features.extend([
                    np.mean(ch_data),
                    np.std(ch_data),
                    np.var(ch_data),
                    np.median(ch_data),
                    np.percentile(ch_data, 25),
                    np.percentile(ch_data, 75),
                    scipy.stats.skew(ch_data),
                    scipy.stats.kurtosis(ch_data)
                ])
        
        elif preprocessed_data.ndim == 3:
            # Shape: (channels, time, freq_bands)
            channels, time_points, freq_bands = preprocessed_data.shape
            
            # Per-channel, per-frequency-band features
            for ch in range(channels):
                for freq in range(freq_bands):
                    ch_freq_data = preprocessed_data[ch, :, freq]
                    
                    # Statistical features
                    features.extend([
                        np.mean(ch_freq_data),
                        np.std(ch_freq_data),
                        np.var(ch_freq_data)
                    ])
            
            # Cross-frequency coupling features
            for ch in range(min(10, channels)):  # Limit to avoid explosion
                ch_data = preprocessed_data[ch, :, :]  # (time, freq_bands)
                
                # Correlation between frequency bands
                for i in range(freq_bands):
                    for j in range(i+1, freq_bands):
                        if np.std(ch_data[:, i]) > 1e-6 and np.std(ch_data[:, j]) > 1e-6:
                            corr = np.corrcoef(ch_data[:, i], ch_data[:, j])[0, 1]
                            features.append(corr if not np.isnan(corr) else 0)
                        else:
                            features.append(0)
            
            # Inter-channel connectivity (limited subset)
            for freq in range(freq_bands):
                freq_data = preprocessed_data[:, :, freq]  # (channels, time)
                
                # Calculate connectivity between first 10 channels
                n_conn_ch = min(10, channels)
                for i in range(n_conn_ch):
                    for j in range(i+1, n_conn_ch):
                        if np.std(freq_data[i, :]) > 1e-6 and np.std(freq_data[j, :]) > 1e-6:
                            corr = np.corrcoef(freq_data[i, :], freq_data[j, :])[0, 1]
                            features.append(corr if not np.isnan(corr) else 0)
                        else:
                            features.append(0)
        
        return np.array(features)
    
    def create_heat_maps(self, data, emotion_label, save_path=None):
        """
        Create EEG heat maps for visualization and feature extraction
        
        Parameters:
        -----------
        data : np.ndarray
            EEG data (channels x time x freq_bands)
        emotion_label : int
            Emotion category
        save_path : str or Path
            Path to save heat map images
            
        Returns:
        --------
        dict : Heat map data for each frequency band
        """
        if data.ndim != 3:
            logger.warning(f"Expected 3D data for heat maps, got {data.ndim}D")
            return {}
        
        channels, time_points, freq_bands = data.shape
        heat_maps = {}
        
        # Create heat map for each frequency band
        for freq_idx in range(freq_bands):
            freq_name = list(self.freq_bands.keys())[freq_idx]
            freq_data = data[:, :, freq_idx]  # (channels, time)
            
            # Average across time to get spatial pattern
            spatial_pattern = np.mean(freq_data, axis=1)  # (channels,)
            
            # Create 2D heat map (8x8 grid for 62 channels + padding)
            heat_map_2d = self._create_spatial_heatmap(spatial_pattern)
            
            heat_maps[freq_name] = {
                'spatial_pattern': spatial_pattern,
                'heat_map_2d': heat_map_2d,
                'flattened': heat_map_2d.flatten()
            }
            
            # Visualize and save if requested
            if save_path:
                self._plot_heatmap(heat_map_2d, freq_name, emotion_label, save_path)
        
        return heat_maps
    
    def _create_spatial_heatmap(self, channel_values):
        """
        Convert 62-channel values to 2D spatial heat map
        
        Parameters:
        -----------
        channel_values : np.ndarray
            Values for each EEG channel (62,)
            
        Returns:
        --------
        np.ndarray : 2D heat map (8x8)
        """
        # Create 8x8 grid (64 positions for 62 channels)
        heat_map = np.zeros((8, 8))
        
        # Map channels to grid positions
        for ch_idx, value in enumerate(channel_values):
            if ch_idx < 62:  # Ensure we don't exceed 62 channels
                row = ch_idx // 8
                col = ch_idx % 8
                if row < 8:  # Stay within bounds
                    heat_map[row, col] = value
        
        return heat_map
    
    def _plot_heatmap(self, heat_map_2d, freq_name, emotion_label, save_path):
        """
        Plot and save heat map visualization
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(heat_map_2d, 
                   cmap='RdYlBu_r', 
                   center=0,
                   cbar_kws={'label': 'Amplitude'},
                   square=True)
        
        emotion_name = self.emotions[emotion_label]
        plt.title(f'EEG Heat Map - {freq_name.title()} Band - {emotion_name}')
        plt.xlabel('Spatial Position (X)')
        plt.ylabel('Spatial Position (Y)')
        
        # Save plot
        save_file = Path(save_path) / f"heatmap_{freq_name}_{emotion_name}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_all_data(self, max_subjects=15, feature_type='de_LDS'):
        """
        Complete processing pipeline: load -> preprocess -> extract features
        
        Parameters:
        -----------
        max_subjects : int
            Maximum number of subjects to process
        feature_type : str
            Type of features to load
            
        Returns:
        --------
        tuple : (features_array, labels_array, metadata)
        """
        cache_file = self.cache_dir / f"processed_features_{feature_type}_{max_subjects}.joblib"
        
        # Try loading from cache
        if cache_file.exists():
            logger.info(f"Loading processed features from cache: {cache_file}")
            cached_data = joblib.load(cache_file)
            self.features = cached_data['features']
            self.labels = cached_data['labels']
            self.heat_maps = cached_data.get('heat_maps', {})
            return self.features, self.labels, cached_data['metadata']
        
        # Load raw data if not already loaded
        if self.raw_data is None:
            self.load_all_data(max_subjects, feature_type)
        
        logger.info("Processing all EEG data...")
        
        all_features = []
        all_labels = []
        all_heat_maps = {}
        metadata = {
            'n_trials': 0,
            'emotion_counts': {0: 0, 1: 0, 2: 0, 3: 0},
            'feature_type': feature_type,
            'max_subjects': max_subjects,
            'processing_date': datetime.now().isoformat()
        }
        
        # Process each trial
        for session in self.raw_data:
            for subject in self.raw_data[session]:
                for trial in self.raw_data[session][subject]:
                    trial_data = self.raw_data[session][subject][trial]
                    
                    # Preprocess EEG data
                    preprocessed = self.preprocess_eeg_data(trial_data['data'])
                    
                    # Extract advanced features
                    features = self.extract_advanced_features(preprocessed)
                    
                    # Create heat maps
                    if preprocessed.ndim == 3:
                        heat_maps = self.create_heat_maps(
                            preprocessed, 
                            trial_data['emotion'],
                            self.results_dir / "heat_maps"
                        )
                        
                        # Add flattened heat maps to features
                        for freq_name, hm_data in heat_maps.items():
                            features = np.concatenate([features, hm_data['flattened']])
                        
                        trial_key = f"s{session}_sub{subject}_t{trial}"
                        all_heat_maps[trial_key] = heat_maps
                    
                    all_features.append(features)
                    all_labels.append(trial_data['emotion'])
                    
                    metadata['n_trials'] += 1
                    metadata['emotion_counts'][trial_data['emotion']] += 1
        
        # Convert to numpy arrays
        self.features = np.array(all_features)
        self.labels = np.array(all_labels)
        self.heat_maps = all_heat_maps
        
        logger.info(f"Processing complete: {self.features.shape[0]} trials, {self.features.shape[1]} features")
        
        # Cache processed data
        cache_data = {
            'features': self.features,
            'labels': self.labels,
            'heat_maps': self.heat_maps,
            'metadata': metadata
        }
        
        joblib.dump(cache_data, cache_file)
        logger.info(f"Processed data cached to: {cache_file}")
        
        return self.features, self.labels, metadata
    
    def save_state(self, filename=None):
        """
        Save complete processor state using joblib
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"seed_iv_processor_state_{timestamp}.joblib"
        
        state = {
            'features': self.features,
            'labels': self.labels,
            'heat_maps': self.heat_maps,
            'models': self.models,
            'data_path': str(self.data_path) if self.data_path else None,
            'random_state': self.random_state,
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = self.cache_dir / filename
        joblib.dump(state, save_path)
        logger.info(f"Processor state saved to: {save_path}")
        
        return save_path
    
    def load_state(self, filename):
        """
        Load processor state from joblib file
        """
        load_path = Path(filename)
        if not load_path.exists():
            load_path = self.cache_dir / filename
        
        if load_path.exists():
            state = joblib.load(load_path)
            self.features = state.get('features')
            self.labels = state.get('labels')
            self.heat_maps = state.get('heat_maps', {})
            self.models = state.get('models', {})
            
            logger.info(f"Processor state loaded from: {load_path}")
            return True
        else:
            logger.error(f"State file not found: {load_path}")
            return False


def main():
    """
    Main function to demonstrate the processor
    """
    print("🧠 SEED-IV Advanced EEG Processing Pipeline")
    print("=" * 60)
    
    # You would need to provide the actual path to your SEED-IV .mat files
    # data_path = "C:/Users/piyus/Downloads/SEED_IV/SEED_IV/eeg_feature_smooth"
    data_path = None  # Set to your actual path
    
    # Initialize processor
    processor = SeedIVProcessor(data_path=data_path)
    
    if data_path is None:
        print("⚠️ Please set the data_path to your SEED-IV directory")
        print("Example: data_path = 'C:/Users/piyus/Downloads/SEED_IV/SEED_IV/eeg_feature_smooth'")
        return
    
    # Process data
    try:
        features, labels, metadata = processor.process_all_data(max_subjects=5)
        
        print(f"✅ Processing complete!")
        print(f"📊 Features shape: {features.shape}")
        print(f"🏷️ Labels shape: {labels.shape}")
        print(f"📈 Trials per emotion: {metadata['emotion_counts']}")
        
        # Save state
        state_file = processor.save_state()
        print(f"💾 State saved to: {state_file}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        print("Please check your data path and file structure")


if __name__ == "__main__":
    main()
