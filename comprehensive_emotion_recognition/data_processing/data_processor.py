"""
Data Processing Module for SEED-IV Dataset
==========================================

Comprehensive data processing pipeline including:
- .mat to CSV conversion
- Feature extraction and engineering
- Data preprocessing and normalization
- Dataset organization and validation

Author: AI Assistant
Date: July 26, 2025
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Local imports
import sys
sys.path.append('..')
from config import config, EMOTION_LABELS, FREQUENCY_BANDS

logger = logging.getLogger(__name__)

class SeedIVDataProcessor:
    """
    Comprehensive SEED-IV data processor with advanced preprocessing capabilities
    """
    
    def __init__(self, base_path: str = None, cache_dir: str = "cache"):
        """
        Initialize the data processor
        
        Parameters:
        -----------
        base_path : str
            Path to SEED-IV eeg_feature_smooth directory
        cache_dir : str
            Directory for caching processed data
        """
        self.base_path = Path(base_path) if base_path else Path(config.data.seed_iv_base_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.emotions = EMOTION_LABELS
        self.frequency_bands = FREQUENCY_BANDS
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.features = {}
        self.labels = {}
        
        # Preprocessing parameters
        self.scaler = None
        self.preprocessing_params = {}
        
        logger.info(f"SeedIVDataProcessor initialized with base path: {self.base_path}")
    
    def scan_dataset(self) -> Dict[str, any]:
        """
        Scan the SEED-IV dataset and return structure information
        
        Returns:
        --------
        dict : Dataset structure information
        """
        logger.info("Scanning SEED-IV dataset structure...")
        
        structure = {
            'subjects': [],
            'sessions': [],
            'files': [],
            'total_files': 0,
            'missing_files': []
        }
        
        # Scan all subject directories
        for subject_dir in sorted(self.base_path.iterdir()):
            if subject_dir.is_dir() and subject_dir.name.isdigit():
                subject_id = int(subject_dir.name)
                structure['subjects'].append(subject_id)
                
                # Scan session files for this subject
                for mat_file in subject_dir.glob("*.mat"):
                    file_info = {
                        'subject': subject_id,
                        'session': self._extract_session_from_filename(mat_file.name),
                        'file_path': str(mat_file),
                        'file_name': mat_file.name
                    }
                    structure['files'].append(file_info)
                    structure['total_files'] += 1
        
        # Identify unique sessions
        structure['sessions'] = sorted(list(set([f['session'] for f in structure['files']])))
        
        logger.info(f"Dataset scan complete: {len(structure['subjects'])} subjects, "
                   f"{len(structure['sessions'])} sessions, {structure['total_files']} files")
        
        return structure
    
    def _extract_session_from_filename(self, filename: str) -> int:
        """Extract session number from filename"""
        # Example: "1_20160518.mat" -> session 1
        return int(filename.split('_')[0])
    
    def load_mat_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load a single .mat file and extract features
        
        Parameters:
        -----------
        file_path : str
            Path to the .mat file
            
        Returns:
        --------
        dict : Dictionary containing extracted features
        """
        try:
            # Load .mat file
            mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
            
            # Remove MATLAB metadata
            features = {key: value for key, value in mat_data.items() 
                       if not key.startswith('__')}
            
            logger.debug(f"Loaded {file_path}: {list(features.keys())}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}
    
    def extract_features_from_mat(self, mat_data: Dict[str, np.ndarray], 
                                 target_features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract specific features from loaded .mat data
        
        Parameters:
        -----------
        mat_data : dict
            Loaded .mat file data
        target_features : list
            List of features to extract (e.g., ['de_LDS1', 'de_movingAve1'])
            
        Returns:
        --------
        dict : Extracted features
        """
        if target_features is None:
            # Default: extract all de_LDS and de_movingAve features
            target_features = [key for key in mat_data.keys() 
                             if key.startswith(('de_LDS', 'de_movingAve'))]
        
        extracted = {}
        for feature_name in target_features:
            if feature_name in mat_data:
                data = mat_data[feature_name]
                
                # Process the feature data
                if data.ndim == 3:
                    # Shape: (channels, time_samples, freq_bands)
                    channels, time_samples, freq_bands = data.shape
                    
                    # Reshape to (time_samples, channels * freq_bands)
                    reshaped = data.transpose(1, 0, 2)  # (time, channels, freq_bands)
                    reshaped = reshaped.reshape(time_samples, channels * freq_bands)
                    
                    extracted[feature_name] = reshaped
                    
                elif data.ndim == 2:
                    extracted[feature_name] = data
                else:
                    extracted[feature_name] = data.reshape(-1, 1)
                    
                logger.debug(f"Extracted {feature_name}: {extracted[feature_name].shape}")
        
        return extracted
    
    def load_complete_dataset(self, target_features: List[str] = None, 
                            use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load the complete SEED-IV dataset
        
        Parameters:
        -----------
        target_features : list
            Features to extract from each file
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        tuple : (features, labels, metadata)
        """
        cache_file = self.cache_dir / "complete_dataset.joblib"
        
        if use_cache and cache_file.exists():
            logger.info("Loading cached dataset...")
            cached_data = joblib.load(cache_file)
            return cached_data['features'], cached_data['labels'], cached_data['metadata']
        
        logger.info("Loading complete SEED-IV dataset...")
        
        if target_features is None:
            target_features = []
            for trial in range(1, 25):  # Trials 1-24
                target_features.extend([f'de_LDS{trial}', f'de_movingAve{trial}'])
        
        # Scan dataset structure
        structure = self.scan_dataset()
        
        all_features = []
        all_labels = []
        metadata = []
        
        for file_info in structure['files']:
            file_path = file_info['file_path']
            subject = file_info['subject']
            session = file_info['session']
            
            # Load .mat file
            mat_data = self.load_mat_file(file_path)
            if not mat_data:
                continue
            
            # Extract features
            features = self.extract_features_from_mat(mat_data, target_features)
            
            # Process each trial feature
            for feature_name, feature_data in features.items():
                # Extract trial number
                trial_num = int(feature_name.split('S')[-1] if 'LDS' in feature_name 
                               else feature_name.split('Ave')[-1])
                
                # Generate emotion label based on trial number (SEED-IV protocol)
                emotion_label = self._get_emotion_label_for_trial(trial_num)
                
                # Store data
                if feature_data.ndim == 2:
                    # Multiple time samples for this trial
                    for time_idx in range(feature_data.shape[0]):
                        all_features.append(feature_data[time_idx])
                        all_labels.append(emotion_label)
                        metadata.append({
                            'subject': subject,
                            'session': session,
                            'trial': trial_num,
                            'feature_type': feature_name.split(str(trial_num))[0],
                            'time_sample': time_idx,
                            'file_path': file_path
                        })
                else:
                    # Single sample
                    all_features.append(feature_data.flatten())
                    all_labels.append(emotion_label)
                    metadata.append({
                        'subject': subject,
                        'session': session,
                        'trial': trial_num,
                        'feature_type': feature_name.split(str(trial_num))[0],
                        'time_sample': 0,
                        'file_path': file_path
                    })
        
        # Convert to numpy arrays
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        logger.info(f"Dataset loaded: {features_array.shape[0]} samples, "
                   f"{features_array.shape[1]} features")
        
        # Cache the data
        if use_cache:
            cache_data = {
                'features': features_array,
                'labels': labels_array,
                'metadata': metadata,
                'target_features': target_features
            }
            joblib.dump(cache_data, cache_file)
            logger.info(f"Dataset cached to {cache_file}")
        
        return features_array, labels_array, metadata
    
    def _get_emotion_label_for_trial(self, trial_num: int) -> int:
        """
        Map trial number to emotion label based on SEED-IV protocol
        
        Parameters:
        -----------
        trial_num : int
            Trial number (1-24)
            
        Returns:
        --------
        int : Emotion label (0: Neutral, 1: Sad, 2: Fear, 3: Happy)
        """
        # SEED-IV emotion sequence (repeated pattern)
        # This is a simplified mapping - adjust based on actual SEED-IV protocol
        emotion_sequence = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,  # First 12 trials
                           0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]   # Last 12 trials
        
        return emotion_sequence[(trial_num - 1) % len(emotion_sequence)]
    
    def preprocess_features(self, features: np.ndarray, 
                          method: str = "zscore") -> np.ndarray:
        """
        Preprocess features using specified normalization method
        
        Parameters:
        -----------
        features : np.ndarray
            Raw features
        method : str
            Normalization method ("zscore", "minmax", "robust")
            
        Returns:
        --------
        np.ndarray : Normalized features
        """
        logger.info(f"Preprocessing features using {method} normalization...")
        
        if method == "zscore":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        normalized_features = self.scaler.fit_transform(features)
        
        # Store preprocessing parameters
        self.preprocessing_params = {
            'method': method,
            'input_shape': features.shape,
            'output_shape': normalized_features.shape
        }
        
        logger.info(f"Features preprocessed: {normalized_features.shape}")
        return normalized_features
    
    def extract_advanced_features(self, features: np.ndarray, 
                                 metadata: List[Dict]) -> np.ndarray:
        """
        Extract advanced features for Stage 2+
        
        Parameters:
        -----------
        features : np.ndarray
            Basic features
        metadata : list
            Sample metadata
            
        Returns:
        --------
        np.ndarray : Enhanced features
        """
        logger.info("Extracting advanced features...")
        
        enhanced_features = []
        
        for i, sample in enumerate(features):
            # Reshape to channel x frequency format for processing
            n_channels = config.data.n_channels
            n_freqs = config.data.n_frequency_bands
            
            if len(sample) == n_channels * n_freqs:
                # Reshape to (channels, frequencies)
                channel_freq = sample.reshape(n_channels, n_freqs)
                
                # Extract multiple feature types
                sample_features = []
                
                # 1. Original features
                sample_features.extend(sample)
                
                # 2. Statistical features
                sample_features.extend([
                    np.mean(channel_freq, axis=0),  # Mean across channels
                    np.std(channel_freq, axis=0),   # Std across channels
                    np.mean(channel_freq, axis=1),  # Mean across frequencies
                    np.std(channel_freq, axis=1)    # Std across frequencies
                ])
                
                # 3. Connectivity features (simplified)
                # Inter-channel correlations
                corr_matrix = np.corrcoef(channel_freq)
                # Use upper triangle (excluding diagonal)
                upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                sample_features.extend(upper_triangle[:100])  # Limit to first 100 connections
                
                # 4. Asymmetry features
                # Left-right hemisphere asymmetry (simplified)
                if n_channels >= 32:  # Ensure enough channels
                    left_channels = channel_freq[:n_channels//2]
                    right_channels = channel_freq[n_channels//2:]
                    asymmetry = np.mean(left_channels, axis=0) - np.mean(right_channels, axis=0)
                    sample_features.extend(asymmetry)
                
                # Flatten and store
                enhanced_features.append(np.concatenate([np.array(f).flatten() 
                                                       for f in sample_features]))
            else:
                # Fallback: use original features
                enhanced_features.append(sample)
        
        enhanced_array = np.array(enhanced_features)
        logger.info(f"Advanced features extracted: {enhanced_array.shape}")
        
        return enhanced_array
    
    def create_spatial_heatmaps(self, features: np.ndarray, 
                               save_examples: bool = True) -> np.ndarray:
        """
        Create spatial heatmaps from EEG features
        
        Parameters:
        -----------
        features : np.ndarray
            EEG features
        save_examples : bool
            Whether to save example heatmaps
            
        Returns:
        --------
        np.ndarray : Flattened heatmap features
        """
        logger.info("Creating spatial heatmaps...")
        
        # Simplified 8x8 spatial mapping for 62 channels
        spatial_maps = []
        
        for sample in features:
            n_channels = config.data.n_channels
            n_freqs = config.data.n_frequency_bands
            
            if len(sample) == n_channels * n_freqs:
                # Reshape to (channels, frequencies)
                channel_data = sample.reshape(n_channels, n_freqs)
                
                # Create spatial maps for each frequency band
                frequency_maps = []
                for freq_idx in range(n_freqs):
                    freq_data = channel_data[:, freq_idx]
                    
                    # Map 62 channels to 8x8 grid (simplified)
                    spatial_map = np.zeros((8, 8))
                    
                    # Fill the map with available channels
                    for ch_idx, value in enumerate(freq_data):
                        if ch_idx < 64:  # Ensure we don't exceed 8x8
                            row = ch_idx // 8
                            col = ch_idx % 8
                            spatial_map[row, col] = value
                    
                    frequency_maps.append(spatial_map)
                
                # Stack frequency maps and flatten
                stacked_maps = np.stack(frequency_maps, axis=-1)  # Shape: (8, 8, 5)
                flattened_map = stacked_maps.flatten()
                spatial_maps.append(flattened_map)
            else:
                # Fallback: use original features
                spatial_maps.append(sample)
        
        spatial_array = np.array(spatial_maps)
        
        # Save example heatmaps
        if save_examples and len(spatial_maps) > 0:
            self._save_example_heatmaps(spatial_array[:5])
        
        logger.info(f"Spatial heatmaps created: {spatial_array.shape}")
        return spatial_array
    
    def _save_example_heatmaps(self, heatmap_samples: np.ndarray):
        """Save example heatmaps for visualization"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Example EEG Spatial Heatmaps', fontsize=16)
            
            for i, sample in enumerate(heatmap_samples[:5]):
                if i >= 6:  # Limit to 6 subplots
                    break
                
                row = i // 3
                col = i % 3
                
                # Reshape back to spatial format for visualization
                spatial_data = sample.reshape(8, 8, -1)
                
                # Show first frequency band
                im = axes[row, col].imshow(spatial_data[:, :, 0], cmap='viridis')
                axes[row, col].set_title(f'Sample {i+1}')
                axes[row, col].set_xlabel('Spatial X')
                axes[row, col].set_ylabel('Spatial Y')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[row, col])
            
            # Remove empty subplots
            if len(heatmap_samples) < 6:
                for i in range(len(heatmap_samples), 6):
                    row = i // 3
                    col = i % 3
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            save_path = self.cache_dir / "example_heatmaps.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Example heatmaps saved to {save_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save example heatmaps: {e}")
    
    def save_processed_data(self, features: np.ndarray, labels: np.ndarray, 
                           metadata: List[Dict], stage_name: str):
        """
        Save processed data for a specific stage
        
        Parameters:
        -----------
        features : np.ndarray
            Processed features
        labels : np.ndarray
            Labels
        metadata : list
            Sample metadata
        stage_name : str
            Name of the processing stage
        """
        save_path = self.cache_dir / f"{stage_name}_processed_data.joblib"
        
        data_package = {
            'features': features,
            'labels': labels,
            'metadata': metadata,
            'preprocessing_params': self.preprocessing_params,
            'stage_name': stage_name,
            'n_samples': len(features),
            'n_features': features.shape[1] if features.ndim > 1 else 1,
            'n_classes': len(np.unique(labels))
        }
        
        joblib.dump(data_package, save_path)
        logger.info(f"Processed data saved for {stage_name}: {save_path}")
        
        return save_path
    
    def load_processed_data(self, stage_name: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load previously processed data
        
        Parameters:
        -----------
        stage_name : str
            Name of the processing stage
            
        Returns:
        --------
        tuple : (features, labels, metadata)
        """
        load_path = self.cache_dir / f"{stage_name}_processed_data.joblib"
        
        if load_path.exists():
            data_package = joblib.load(load_path)
            logger.info(f"Loaded processed data for {stage_name}: "
                       f"{data_package['n_samples']} samples, "
                       f"{data_package['n_features']} features")
            
            return data_package['features'], data_package['labels'], data_package['metadata']
        else:
            raise FileNotFoundError(f"No processed data found for {stage_name}")
    
    def generate_data_report(self, features: np.ndarray, labels: np.ndarray, 
                           metadata: List[Dict]) -> Dict:
        """
        Generate comprehensive data report
        
        Parameters:
        -----------
        features : np.ndarray
            Feature data
        labels : np.ndarray
            Labels
        metadata : list
            Sample metadata
            
        Returns:
        --------
        dict : Data report
        """
        logger.info("Generating data report...")
        
        report = {
            'dataset_info': {
                'n_samples': len(features),
                'n_features': features.shape[1] if features.ndim > 1 else 1,
                'n_classes': len(np.unique(labels)),
                'feature_shape': features.shape
            },
            'class_distribution': {},
            'subject_distribution': {},
            'session_distribution': {},
            'data_quality': {}
        }
        
        # Class distribution
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            emotion_name = EMOTION_LABELS.get(label, f"Unknown_{label}")
            report['class_distribution'][emotion_name] = {
                'count': int(count),
                'percentage': float(count / len(labels) * 100)
            }
        
        # Subject and session distribution
        if metadata:
            subjects = [m['subject'] for m in metadata]
            sessions = [m['session'] for m in metadata]
            
            unique_subjects, subject_counts = np.unique(subjects, return_counts=True)
            for subject, count in zip(unique_subjects, subject_counts):
                report['subject_distribution'][f'subject_{subject}'] = int(count)
            
            unique_sessions, session_counts = np.unique(sessions, return_counts=True)
            for session, count in zip(unique_sessions, session_counts):
                report['session_distribution'][f'session_{session}'] = int(count)
        
        # Data quality metrics
        report['data_quality'] = {
            'has_nan': bool(np.isnan(features).any()),
            'has_inf': bool(np.isinf(features).any()),
            'feature_mean': float(np.mean(features)),
            'feature_std': float(np.std(features)),
            'feature_min': float(np.min(features)),
            'feature_max': float(np.max(features))
        }
        
        logger.info("Data report generated successfully")
        return report


def main():
    """
    Demonstration of the data processing pipeline
    """
    print("🧠 SEED-IV Data Processing Pipeline")
    print("=" * 60)
    
    # Initialize processor
    processor = SeedIVDataProcessor()
    
    # Update data path if needed
    # processor.base_path = Path(r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth")
    
    # Scan dataset
    structure = processor.scan_dataset()
    print(f"📊 Dataset structure: {len(structure['subjects'])} subjects, "
          f"{len(structure['sessions'])} sessions, {structure['total_files']} files")
    
    # Load complete dataset (this will take some time)
    print("\n📁 Loading complete dataset...")
    features, labels, metadata = processor.load_complete_dataset()
    
    print(f"✅ Dataset loaded: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Generate report
    report = processor.generate_data_report(features, labels, metadata)
    print(f"\n📈 Data Report:")
    print(f"  Samples: {report['dataset_info']['n_samples']}")
    print(f"  Features: {report['dataset_info']['n_features']}")
    print(f"  Classes: {report['dataset_info']['n_classes']}")
    print(f"  Class distribution: {report['class_distribution']}")
    
    # Preprocess features
    print("\n🔄 Preprocessing features...")
    normalized_features = processor.preprocess_features(features, method="zscore")
    
    # Create advanced features
    print("\n🚀 Creating advanced features...")
    advanced_features = processor.extract_advanced_features(normalized_features, metadata)
    
    # Create spatial heatmaps
    print("\n🗺️  Creating spatial heatmaps...")
    heatmap_features = processor.create_spatial_heatmaps(normalized_features)
    
    # Save processed data
    print("\n💾 Saving processed data...")
    processor.save_processed_data(advanced_features, labels, metadata, "comprehensive")
    
    print("\n✅ Data processing pipeline completed successfully!")
    print(f"📁 Processed data saved in: {processor.cache_dir}")

if __name__ == "__main__":
    main()
