"""
Advanced Feature Engineering for SEED-IV
========================================

Comprehensive feature engineering pipeline including:
- Spatial features (connectivity, asymmetry)
- Temporal features (statistical moments, complexity)
- Frequency domain features (power spectral density, coherence)
- Advanced transformations and selection methods

Author: AI Assistant
Date: July 27, 2025
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Import config with absolute path handling
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
from config import config

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for EEG emotion recognition
    """
    
    def __init__(self, n_channels: int = 62, n_frequency_bands: int = 5):
        """
        Initialize feature engineer
        
        Parameters:
        -----------
        n_channels : int
            Number of EEG channels
        n_frequency_bands : int
            Number of frequency bands
        """
        self.n_channels = n_channels
        self.n_frequency_bands = n_frequency_bands
        self.feature_names = []
        
        # Channel positions for spatial analysis (simplified 10-20 system)
        self.channel_positions = self._get_channel_positions()
        
        logger.info(f"AdvancedFeatureEngineer initialized: {n_channels} channels, "
                   f"{n_frequency_bands} frequency bands")
    
    def _get_channel_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Get approximate 2D positions for EEG channels
        Simplified mapping for 62-channel system
        
        Returns:
        --------
        Dict[int, Tuple[float, float]] : Channel positions (x, y)
        """
        # Simplified grid layout for 62 channels (8x8 with 2 removed)
        positions = {}
        channels_per_row = 8
        
        for i in range(self.n_channels):
            row = i // channels_per_row
            col = i % channels_per_row
            
            # Convert to normalized coordinates
            x = (col - 3.5) / 3.5  # Center around 0
            y = (3.5 - row) / 3.5  # Flip Y axis
            
            positions[i] = (x, y)
        
        return positions
    
    def extract_spatial_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract spatial features from EEG data
        
        Parameters:
        -----------
        eeg_data : np.ndarray
            EEG data with shape (n_samples, n_channels * n_frequency_bands)
            
        Returns:
        --------
        np.ndarray : Spatial features
        """
        logger.info("Extracting spatial features...")
        
        # Reshape to (n_samples, n_channels, n_frequency_bands)
        n_samples = eeg_data.shape[0]
        reshaped_data = eeg_data.reshape(n_samples, self.n_channels, self.n_frequency_bands)
        
        spatial_features = []
        
        for sample_idx in range(n_samples):
            sample_features = []
            
            for freq_band in range(self.n_frequency_bands):
                channel_data = reshaped_data[sample_idx, :, freq_band]
                
                # 1. Inter-channel correlation
                corr_matrix = np.corrcoef(channel_data.reshape(1, -1), 
                                        channel_data.reshape(1, -1))[0, 1]
                if np.isnan(corr_matrix):
                    corr_matrix = 0.0
                sample_features.append(corr_matrix)
                
                # 2. Hemispheric asymmetry (simplified)
                left_channels = channel_data[:self.n_channels//2]
                right_channels = channel_data[self.n_channels//2:]
                
                left_power = np.mean(left_channels**2)
                right_power = np.mean(right_channels**2)
                
                # Asymmetry index
                asymmetry = (right_power - left_power) / (right_power + left_power + 1e-10)
                sample_features.append(asymmetry)
                
                # 3. Spatial complexity (approximate)
                # For 62 channels, use simpler spatial metrics
                spatial_mean = np.mean(channel_data)
                spatial_std = np.std(channel_data)
                spatial_range = np.max(channel_data) - np.min(channel_data)
                spatial_complexity = spatial_std / (spatial_mean + 1e-10)
                sample_features.append(spatial_complexity)
                
                # 4. Frontal-posterior ratio
                frontal_idx = list(range(self.n_channels//4))  # First quarter
                posterior_idx = list(range(3*self.n_channels//4, self.n_channels))  # Last quarter
                
                frontal_power = np.mean(channel_data[frontal_idx]**2)
                posterior_power = np.mean(channel_data[posterior_idx]**2)
                
                fp_ratio = frontal_power / (posterior_power + 1e-10)
                sample_features.append(fp_ratio)
            
            spatial_features.append(sample_features)
        
        spatial_features = np.array(spatial_features)
        
        # Generate feature names
        base_names = ['inter_corr', 'hemispheric_asym', 'spatial_complex', 'fp_ratio']
        spatial_feature_names = []
        for freq in range(self.n_frequency_bands):
            for name in base_names:
                spatial_feature_names.append(f'spatial_{name}_freq{freq+1}')
        
        self.feature_names.extend(spatial_feature_names)
        
        logger.info(f"Extracted {spatial_features.shape[1]} spatial features")
        return spatial_features
    
    def extract_temporal_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from EEG data
        
        Parameters:
        -----------
        eeg_data : np.ndarray
            EEG data with shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : Temporal features
        """
        logger.info("Extracting temporal features...")
        
        temporal_features = []
        
        for sample_idx in range(eeg_data.shape[0]):
            sample_data = eeg_data[sample_idx, :]
            sample_features = []
            
            # 1. Statistical moments
            sample_features.append(np.mean(sample_data))       # Mean
            sample_features.append(np.std(sample_data))        # Standard deviation
            sample_features.append(stats.skew(sample_data))    # Skewness
            sample_features.append(stats.kurtosis(sample_data)) # Kurtosis
            
            # 2. Range and percentiles
            sample_features.append(np.ptp(sample_data))        # Peak-to-peak
            sample_features.append(np.percentile(sample_data, 25))  # 25th percentile
            sample_features.append(np.percentile(sample_data, 75))  # 75th percentile
            sample_features.append(np.median(sample_data))     # Median
            
            # 3. Zero-crossing rate (approximate)
            zero_crossings = np.sum(np.diff(np.sign(sample_data - np.mean(sample_data))) != 0)
            sample_features.append(zero_crossings / len(sample_data))
            
            # 4. Hjorth parameters (simplified)
            # Activity (variance)
            activity = np.var(sample_data)
            sample_features.append(activity)
            
            # Mobility (approximate)
            if len(sample_data) > 1:
                diff_data = np.diff(sample_data)
                mobility = np.sqrt(np.var(diff_data) / (np.var(sample_data) + 1e-10))
            else:
                mobility = 0.0
            sample_features.append(mobility)
            
            # Complexity (approximate)
            if len(sample_data) > 2:
                diff2_data = np.diff(diff_data)
                complexity = np.sqrt(np.var(diff2_data) / (np.var(diff_data) + 1e-10)) / mobility
            else:
                complexity = 0.0
            sample_features.append(complexity)
            
            # 5. Energy and power
            energy = np.sum(sample_data**2)
            sample_features.append(energy)
            
            power = energy / len(sample_data)
            sample_features.append(power)
            
            temporal_features.append(sample_features)
        
        temporal_features = np.array(temporal_features)
        
        # Generate feature names
        temporal_feature_names = [
            'temp_mean', 'temp_std', 'temp_skewness', 'temp_kurtosis',
            'temp_ptp', 'temp_q25', 'temp_q75', 'temp_median',
            'temp_zcr', 'temp_activity', 'temp_mobility', 'temp_complexity',
            'temp_energy', 'temp_power'
        ]
        
        self.feature_names.extend(temporal_feature_names)
        
        logger.info(f"Extracted {temporal_features.shape[1]} temporal features")
        return temporal_features
    
    def extract_frequency_features(self, eeg_data: np.ndarray, sampling_rate: int = 200) -> np.ndarray:
        """
        Extract frequency domain features
        
        Parameters:
        -----------
        eeg_data : np.ndarray
            EEG data with shape (n_samples, n_features)
        sampling_rate : int
            Sampling rate in Hz
            
        Returns:
        --------
        np.ndarray : Frequency domain features
        """
        logger.info("Extracting frequency domain features...")
        
        frequency_features = []
        
        for sample_idx in range(eeg_data.shape[0]):
            sample_data = eeg_data[sample_idx, :]
            sample_features = []
            
            # 1. Power Spectral Density (PSD) using Welch's method
            try:
                frequencies, psd = signal.welch(sample_data, fs=sampling_rate, nperseg=min(256, len(sample_data)))
                
                # Band power in different frequency ranges
                delta_power = np.mean(psd[(frequencies >= 1) & (frequencies <= 4)])
                theta_power = np.mean(psd[(frequencies >= 4) & (frequencies <= 8)])
                alpha_power = np.mean(psd[(frequencies >= 8) & (frequencies <= 13)])
                beta_power = np.mean(psd[(frequencies >= 13) & (frequencies <= 30)])
                gamma_power = np.mean(psd[(frequencies >= 30) & (frequencies <= 50)])
                
                sample_features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])
                
                # 2. Spectral centroid
                spectral_centroid = np.sum(frequencies * psd) / (np.sum(psd) + 1e-10)
                sample_features.append(spectral_centroid)
                
                # 3. Spectral bandwidth
                spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-10))
                sample_features.append(spectral_bandwidth)
                
                # 4. Spectral rolloff (95% of energy)
                cumulative_psd = np.cumsum(psd)
                total_energy = cumulative_psd[-1]
                rolloff_threshold = 0.95 * total_energy
                rolloff_idx = np.where(cumulative_psd >= rolloff_threshold)[0]
                spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
                sample_features.append(spectral_rolloff)
                
            except Exception as e:
                logger.warning(f"Failed to compute frequency features for sample {sample_idx}: {e}")
                # Fill with zeros if computation fails
                sample_features.extend([0.0] * 8)
            
            frequency_features.append(sample_features)
        
        frequency_features = np.array(frequency_features)
        
        # Generate feature names
        freq_feature_names = [
            'freq_delta_power', 'freq_theta_power', 'freq_alpha_power', 
            'freq_beta_power', 'freq_gamma_power',
            'freq_spectral_centroid', 'freq_spectral_bandwidth', 'freq_spectral_rolloff'
        ]
        
        self.feature_names.extend(freq_feature_names)
        
        logger.info(f"Extracted {frequency_features.shape[1]} frequency domain features")
        return frequency_features
    
    def extract_connectivity_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract connectivity features between channels
        
        Parameters:
        -----------
        eeg_data : np.ndarray
            EEG data with shape (n_samples, n_channels * n_frequency_bands)
            
        Returns:
        --------
        np.ndarray : Connectivity features
        """
        logger.info("Extracting connectivity features...")
        
        # Reshape to (n_samples, n_channels, n_frequency_bands)
        n_samples = eeg_data.shape[0]
        reshaped_data = eeg_data.reshape(n_samples, self.n_channels, self.n_frequency_bands)
        
        connectivity_features = []
        
        for sample_idx in range(n_samples):
            sample_features = []
            
            for freq_band in range(self.n_frequency_bands):
                channel_data = reshaped_data[sample_idx, :, freq_band]
                
                # 1. Correlation matrix
                if len(channel_data) > 1:
                    corr_matrix = np.corrcoef(channel_data.reshape(1, -1))
                    if corr_matrix.size > 1:
                        # Extract upper triangle (excluding diagonal)
                        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                        
                        # Statistical measures of connectivity
                        mean_connectivity = np.mean(upper_tri)
                        std_connectivity = np.std(upper_tri)
                        max_connectivity = np.max(upper_tri)
                        min_connectivity = np.min(upper_tri)
                        
                        sample_features.extend([mean_connectivity, std_connectivity, 
                                              max_connectivity, min_connectivity])
                    else:
                        sample_features.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    sample_features.extend([0.0, 0.0, 0.0, 0.0])
                
                # 2. Network measures (simplified)
                try:
                    # Create adjacency matrix from correlation
                    adj_matrix = np.abs(np.corrcoef(channel_data.reshape(1, -1)))
                    if adj_matrix.size > 1:
                        # Threshold the matrix
                        threshold = np.percentile(adj_matrix, 75)
                        binary_adj = (adj_matrix > threshold).astype(int)
                        
                        # Create graph
                        G = nx.from_numpy_array(binary_adj)
                        
                        # Network measures
                        if G.number_of_edges() > 0:
                            clustering = nx.average_clustering(G)
                            density = nx.density(G)
                        else:
                            clustering = 0.0
                            density = 0.0
                        
                        sample_features.extend([clustering, density])
                    else:
                        sample_features.extend([0.0, 0.0])
                        
                except Exception as e:
                    logger.debug(f"Network analysis failed for sample {sample_idx}, freq {freq_band}: {e}")
                    sample_features.extend([0.0, 0.0])
            
            connectivity_features.append(sample_features)
        
        connectivity_features = np.array(connectivity_features)
        
        # Generate feature names
        base_names = ['conn_mean', 'conn_std', 'conn_max', 'conn_min', 'conn_clustering', 'conn_density']
        conn_feature_names = []
        for freq in range(self.n_frequency_bands):
            for name in base_names:
                conn_feature_names.append(f'{name}_freq{freq+1}')
        
        self.feature_names.extend(conn_feature_names)
        
        logger.info(f"Extracted {connectivity_features.shape[1]} connectivity features")
        return connectivity_features
    
    def extract_all_features(self, eeg_data: np.ndarray, 
                           feature_types: List[str] = None) -> np.ndarray:
        """
        Extract all specified feature types
        
        Parameters:
        -----------
        eeg_data : np.ndarray
            EEG data
        feature_types : List[str], optional
            Types of features to extract
            
        Returns:
        --------
        np.ndarray : Combined features
        """
        if feature_types is None:
            feature_types = ['spatial', 'temporal', 'frequency', 'connectivity']
        
        logger.info(f"Extracting features: {feature_types}")
        
        all_features = []
        self.feature_names = []  # Reset feature names
        
        if 'spatial' in feature_types:
            spatial_features = self.extract_spatial_features(eeg_data)
            all_features.append(spatial_features)
        
        if 'temporal' in feature_types:
            temporal_features = self.extract_temporal_features(eeg_data)
            all_features.append(temporal_features)
        
        if 'frequency' in feature_types:
            frequency_features = self.extract_frequency_features(eeg_data)
            all_features.append(frequency_features)
        
        if 'connectivity' in feature_types:
            connectivity_features = self.extract_connectivity_features(eeg_data)
            all_features.append(connectivity_features)
        
        if all_features:
            combined_features = np.hstack(all_features)
            logger.info(f"Combined features shape: {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No features extracted!")
            return np.array([])
    
    def select_features(self, features: np.ndarray, labels: np.ndarray, 
                       method: str = 'mutual_info', k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature selection using various methods
        
        Parameters:
        -----------
        features : np.ndarray
            Input features
        labels : np.ndarray
            Target labels
        method : str
            Selection method ('mutual_info', 'f_classif', 'rfe')
        k : int
            Number of features to select
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (selected_features, selected_indices)
        """
        logger.info(f"Selecting {k} features using {method}")
        
        k = min(k, features.shape[1])  # Ensure k doesn't exceed available features
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        selected_features = selector.fit_transform(features, labels)
        selected_indices = selector.get_support(indices=True)
        
        # Update feature names
        if self.feature_names:
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
            self.feature_names = selected_feature_names
        
        logger.info(f"Selected {selected_features.shape[1]} features from {features.shape[1]}")
        
        return selected_features, selected_indices
    
    def apply_dimensionality_reduction(self, features: np.ndarray, 
                                     method: str = 'pca', n_components: int = 50) -> np.ndarray:
        """
        Apply dimensionality reduction
        
        Parameters:
        -----------
        features : np.ndarray
            Input features
        method : str
            Reduction method ('pca', 'ica', 'tsne')
        n_components : int
            Number of components
            
        Returns:
        --------
        np.ndarray : Reduced features
        """
        logger.info(f"Applying {method} dimensionality reduction to {n_components} components")
        
        n_components = min(n_components, features.shape[1], features.shape[0])
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'ica':
            reducer = FastICA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=min(n_components, 3), random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        reduced_features = reducer.fit_transform(features)
        
        # Update feature names
        self.feature_names = [f'{method}_component_{i+1}' for i in range(n_components)]
        
        logger.info(f"Reduced features shape: {reduced_features.shape}")
        
        return reduced_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get current feature names
        
        Returns:
        --------
        List[str] : Feature names
        """
        return self.feature_names.copy()


def main():
    """
    Demonstration of advanced feature engineering
    """
    print("🧠 Advanced Feature Engineering Demonstration")
    print("=" * 60)
    
    # Create dummy EEG data
    n_samples = 100
    n_channels = 62
    n_frequency_bands = 5
    
    # Simulate EEG data
    eeg_data = np.random.randn(n_samples, n_channels * n_frequency_bands)
    labels = np.random.randint(0, 4, n_samples)
    
    print(f"📊 Input data: {eeg_data.shape}")
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer(n_channels, n_frequency_bands)
    
    # Extract all features
    all_features = feature_engineer.extract_all_features(eeg_data)
    print(f"🔧 Extracted features: {all_features.shape}")
    
    # Feature selection
    selected_features, indices = feature_engineer.select_features(
        all_features, labels, method='mutual_info', k=50
    )
    print(f"✅ Selected features: {selected_features.shape}")
    
    # Dimensionality reduction
    reduced_features = feature_engineer.apply_dimensionality_reduction(
        selected_features, method='pca', n_components=20
    )
    print(f"📉 Reduced features: {reduced_features.shape}")
    
    # Get feature names
    feature_names = feature_engineer.get_feature_names()
    print(f"📝 Feature names: {len(feature_names)} features")
    print(f"   First 5: {feature_names[:5]}")


if __name__ == "__main__":
    main()
