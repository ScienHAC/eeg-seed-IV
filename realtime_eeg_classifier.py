"""
Real-Time EEG Emotion Detection for Production Deployment
=========================================================

Lightweight script for real-time emotion detection using trained deep learning model.
Optimized for voice companion bot integration.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
import json
import pickle
from pathlib import Path
import time

class EEGAutoencoder(nn.Module):
    """Autoencoder for feature compression (same as training)"""
    def __init__(self, input_dim, encoding_dim):
        super(EEGAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class AdvancedCNNLSTM(nn.Module):
    """CNN-LSTM model (same as training)"""
    def __init__(self, input_dim, sequence_length=62, num_classes=4, dropout=0.3):
        super(AdvancedCNNLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(sequence_length, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        
        # LSTM
        self.lstm = nn.LSTM(512, 256, num_layers=3, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        
        # Attention
        self.attention = nn.MultiheadAttention(512, num_heads=8, dropout=dropout)
        
        # Classification layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class RealTimeEEGClassifier:
    """
    Production-ready real-time EEG emotion classifier
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.emotion_names = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        self.intensity_levels = {
            (0, 50): "Very Low",
            (50, 65): "Low", 
            (65, 80): "Medium",
            (80, 90): "High",
            (90, 100): "Very High"
        }
        
        # Model components
        self.scaler = None
        self.autoencoder = None
        self.model = None
        self.feature_mapping = None
        
        # Performance tracking
        self.prediction_times = []
        self.prediction_count = 0
        
    def load_model(self, model_path=None, autoencoder_path=None, scaler_path=None, mapping_path=None):
        """Load all model components"""
        print("🔄 Loading trained model components...")
        
        # Set default paths if not provided
        if model_path is None:
            model_path = self.model_dir / "best_eeg_model.pth"
        if autoencoder_path is None:
            autoencoder_path = self.model_dir / "eeg_autoencoder.pth"
        if scaler_path is None:
            scaler_path = self.model_dir / "eeg_scaler.pkl"
        if mapping_path is None:
            mapping_path = self.model_dir / "feature_mapping.json"
        
        try:
            # Load feature mapping
            with open(mapping_path, 'r') as f:
                self.feature_mapping = json.load(f)
            print(f"✅ Loaded feature mapping: {len(self.feature_mapping['original_features'])} → {len(self.feature_mapping['compressed_features'])} features")
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Loaded feature scaler")
            
            # Load autoencoder
            input_dim = len(self.feature_mapping['original_features'])
            encoding_dim = len(self.feature_mapping['compressed_features'])
            self.autoencoder = EEGAutoencoder(input_dim, encoding_dim).to(self.device)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            self.autoencoder.eval()
            print("✅ Loaded autoencoder")
            
            # Load main model
            self.model = AdvancedCNNLSTM(encoding_dim).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("✅ Loaded main classification model")
            
            print(f"🚀 Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def save_model(self, model_path=None, autoencoder_path=None, scaler_path=None, mapping_path=None):
        """Save all model components"""
        print("💾 Saving model components...")
        
        # Create model directory
        self.model_dir.mkdir(exist_ok=True)
        
        # Set default paths
        if model_path is None:
            model_path = self.model_dir / "best_eeg_model.pth"
        if autoencoder_path is None:
            autoencoder_path = self.model_dir / "eeg_autoencoder.pth"
        if scaler_path is None:
            scaler_path = self.model_dir / "eeg_scaler.pkl"
        if mapping_path is None:
            mapping_path = self.model_dir / "feature_mapping.json"
        
        try:
            # Save main model
            torch.save(self.model.state_dict(), model_path)
            print(f"✅ Saved main model: {model_path}")
            
            # Save autoencoder
            torch.save(self.autoencoder.state_dict(), autoencoder_path)
            print(f"✅ Saved autoencoder: {autoencoder_path}")
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Saved scaler: {scaler_path}")
            
            # Save feature mapping
            with open(mapping_path, 'w') as f:
                json.dump(self.feature_mapping, f, indent=2)
            print(f"✅ Saved feature mapping: {mapping_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def preprocess_raw_eeg(self, raw_eeg_data, sampling_rate=1000, window_size=4):
        """
        Preprocess raw EEG signals for emotion detection
        
        Args:
            raw_eeg_data: Dict with channel names as keys and signal arrays as values
                         e.g., {'Ch1': [signal_values], 'Ch2': [signal_values], ...}
            sampling_rate: EEG sampling rate in Hz
            window_size: Window size in seconds
        
        Returns:
            Dict with processed features matching training format
        """
        processed_features = {}
        
        # Process each channel
        for channel, signal in raw_eeg_data.items():
            if not channel.startswith('Ch'):
                continue
                
            # Convert to numpy array
            signal = np.array(signal)
            
            # Basic signal processing
            signal_clean = self._clean_signal(signal, sampling_rate)
            
            # Extract frequency band features (simulate the 5 frequency bands)
            freq_features = self._extract_frequency_features(signal_clean, sampling_rate)
            
            # Create features for each frequency band
            for freq_idx, freq_power in enumerate(freq_features):
                freq_name = f'Freq{freq_idx + 1}'
                
                # Statistical features (matching training data processing)
                processed_features[f'{channel}_{freq_name}_mean'] = np.mean(freq_power)
                processed_features[f'{channel}_{freq_name}_std'] = np.std(freq_power)
                processed_features[f'{channel}_{freq_name}_median'] = np.median(freq_power)
                processed_features[f'{channel}_{freq_name}_iqr'] = np.percentile(freq_power, 75) - np.percentile(freq_power, 25)
                processed_features[f'{channel}_{freq_name}_skewness'] = self._calculate_skewness(freq_power)
                processed_features[f'{channel}_{freq_name}_kurtosis'] = self._calculate_kurtosis(freq_power)
                processed_features[f'{channel}_{freq_name}_power'] = np.sum(freq_power ** 2)
                processed_features[f'{channel}_{freq_name}_rms'] = np.sqrt(np.mean(freq_power ** 2))
                processed_features[f'{channel}_{freq_name}_peak_freq'] = self._find_peak_frequency(freq_power)
                processed_features[f'{channel}_{freq_name}_bandwidth'] = self._calculate_bandwidth(freq_power)
        
        return processed_features
    
    def _clean_signal(self, signal, sampling_rate):
        """Clean raw EEG signal"""
        # Remove DC component
        signal = signal - np.mean(signal)
        
        # Simple bandpass filter (0.5-50 Hz simulation)
        # In real deployment, use proper signal processing library
        from scipy.signal import butter, filtfilt
        
        # Bandpass filter
        nyquist = sampling_rate / 2
        low_freq = 0.5 / nyquist
        high_freq = 50.0 / nyquist
        
        if high_freq >= 1.0:
            high_freq = 0.99
            
        b, a = butter(4, [low_freq, high_freq], btype='band')
        signal_filtered = filtfilt(b, a, signal)
        
        # Remove outliers (3-sigma rule)
        mean_val = np.mean(signal_filtered)
        std_val = np.std(signal_filtered)
        signal_clean = signal_filtered[np.abs(signal_filtered - mean_val) <= 3 * std_val]
        
        if len(signal_clean) == 0:
            signal_clean = signal_filtered
            
        return signal_clean
    
    def _extract_frequency_features(self, signal, sampling_rate):
        """Extract power in different frequency bands"""
        # Frequency bands: Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-50)
        freq_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        
        # FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        power_spectrum = np.abs(fft_vals) ** 2
        
        band_powers = []
        for low_freq, high_freq in freq_bands:
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = power_spectrum[freq_mask]
            
            if len(band_power) == 0:
                band_power = np.array([0])
                
            band_powers.append(band_power)
        
        return band_powers
    
    def _calculate_skewness(self, data):
        """Calculate skewness"""
        if len(data) < 3:
            return 0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis"""
        if len(data) < 4:
            return 0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _find_peak_frequency(self, data):
        """Find dominant frequency"""
        if len(data) < 2:
            return 0
        fft_vals = np.abs(np.fft.fft(data))
        return np.argmax(fft_vals[:len(fft_vals)//2])
    
    def _calculate_bandwidth(self, data):
        """Calculate spectral bandwidth"""
        if len(data) < 2:
            return 0
        fft_vals = np.abs(np.fft.fft(data))
        power_spectrum = fft_vals ** 2
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0
        
        freqs = np.arange(len(power_spectrum))
        centroid = np.sum(freqs * power_spectrum) / total_power
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power_spectrum) / total_power)
        return bandwidth
    
    def predict_emotion(self, eeg_features, return_processing_time=False):
        """
        Fast emotion prediction for real-time use
        
        Args:
            eeg_features: Either dict of preprocessed features or raw EEG signals
            return_processing_time: Whether to return processing time
        
        Returns:
            Dict with emotion prediction and confidence
        """
        start_time = time.time()
        
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        # Handle different input formats
        if isinstance(eeg_features, dict) and 'Ch1' in eeg_features and isinstance(eeg_features['Ch1'], (list, np.ndarray)):
            # Raw EEG signals - preprocess them
            eeg_features = self.preprocess_raw_eeg(eeg_features)
        
        # Ensure we have all required features
        feature_array = []
        for feat in self.feature_mapping['original_features']:
            if feat in eeg_features:
                feature_array.append(eeg_features[feat])
            else:
                feature_array.append(0)  # Default value for missing features
        
        # Convert to numpy array and scale
        features = np.array(feature_array).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Compress with autoencoder
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            features_compressed = self.autoencoder.encode(features_tensor)
        
        # Predict with main model
        with torch.no_grad():
            outputs = self.model(features_compressed)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        
        # Calculate confidence and intensity
        max_prob = float(torch.max(probabilities))
        intensity_score = max_prob * 100
        
        # Determine intensity level
        intensity = "Unknown"
        for (low, high), level in self.intensity_levels.items():
            if low <= intensity_score < high:
                intensity = level
                break
        
        processing_time = time.time() - start_time
        self.prediction_times.append(processing_time)
        self.prediction_count += 1
        
        result = {
            'emotion': self.emotion_names[int(prediction)],
            'emotion_code': int(prediction),
            'confidence': max_prob,
            'intensity': intensity,
            'intensity_score': intensity_score,
            'probabilities': {
                self.emotion_names[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        if return_processing_time:
            result['processing_time_ms'] = processing_time * 1000
            result['avg_processing_time_ms'] = np.mean(self.prediction_times) * 1000
        
        return result
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.prediction_times:
            return "No predictions made yet"
        
        avg_time = np.mean(self.prediction_times) * 1000
        min_time = np.min(self.prediction_times) * 1000
        max_time = np.max(self.prediction_times) * 1000
        std_time = np.std(self.prediction_times) * 1000
        
        return {
            'total_predictions': self.prediction_count,
            'avg_processing_time_ms': avg_time,
            'min_processing_time_ms': min_time,
            'max_processing_time_ms': max_time,
            'std_processing_time_ms': std_time,
            'predictions_per_second': 1000 / avg_time if avg_time > 0 else 0
        }

# Example usage for voice companion bot integration
def voice_bot_integration_example():
    """Example of how to integrate with voice companion bot"""
    
    # Initialize classifier
    classifier = RealTimeEEGClassifier()
    
    # Load trained model (you need to train first using advanced_deep_eeg_classifier.py)
    if not classifier.load_model():
        print("❌ Could not load model. Train the model first!")
        return
    
    print("🤖 Voice Companion Bot - EEG Emotion Detection Ready!")
    print("=" * 50)
    
    # Simulate real-time EEG data (replace with actual EEG input)
    def simulate_real_time_eeg():
        """Simulate real-time EEG data from electrodes"""
        channels = [f'Ch{i+1}' for i in range(62)]  # 62 EEG channels
        sampling_rate = 1000  # 1000 Hz
        window_samples = 4 * sampling_rate  # 4 second window
        
        # Generate simulated EEG signals (replace with real EEG interface)
        eeg_data = {}
        for channel in channels:
            # Simulate realistic EEG signal with different frequency components
            t = np.linspace(0, 4, window_samples)
            signal = (np.sin(2 * np.pi * 10 * t) +  # Alpha waves
                     0.5 * np.sin(2 * np.pi * 6 * t) +  # Theta waves  
                     0.3 * np.sin(2 * np.pi * 20 * t) +  # Beta waves
                     0.1 * np.random.randn(len(t)))  # Noise
            
            eeg_data[channel] = signal
        
        return eeg_data
    
    # Real-time emotion detection loop
    print("🧠 Starting real-time emotion detection...")
    print("Press Ctrl+C to stop")
    
    try:
        for i in range(10):  # Demo with 10 predictions
            # Get EEG data (in real deployment, this comes from EEG headset)
            eeg_data = simulate_real_time_eeg()
            
            # Predict emotion
            result = classifier.predict_emotion(eeg_data, return_processing_time=True)
            
            # Display results for voice bot
            emotion = result['emotion']
            confidence = result['confidence']
            intensity = result['intensity']
            processing_time = result['processing_time_ms']
            
            print(f"\\n🎯 Prediction {i+1}:")
            print(f"   Emotion: {emotion} ({confidence:.1%} confidence)")
            print(f"   Intensity: {intensity}")
            print(f"   Processing: {processing_time:.1f}ms")
            
            # Voice bot can use this information to:
            # 1. Adjust response tone based on emotion
            # 2. Provide emotional support if sad/fear detected
            # 3. Match energy level to user's emotional state
            # 4. Trigger specific responses for different emotions
            
            if emotion == 'Sad' and confidence > 0.7:
                print("   🤗 Voice Bot: Detected sadness - switching to supportive mode")
            elif emotion == 'Happy' and confidence > 0.7:
                print("   😊 Voice Bot: Detected happiness - matching positive energy")
            elif emotion == 'Fear' and confidence > 0.7:
                print("   😌 Voice Bot: Detected fear - switching to calming mode")
            
            time.sleep(1)  # Wait 1 second between predictions
            
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Stopped real-time detection")
    
    # Show performance statistics
    stats = classifier.get_performance_stats()
    print(f"\\n📊 Performance Statistics:")
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Average processing time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"   Predictions per second: {stats['predictions_per_second']:.1f}")

if __name__ == "__main__":
    voice_bot_integration_example()
