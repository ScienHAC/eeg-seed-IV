"""
Optimized EEG Emotion Classification System for SEED-IV Dataset
===============================================================

Simple, clean, and highly accurate emotion recognition system.
Automatically finds the best features and creates production-ready model.

Features:
- Loads all 2160 CSV files systematically
- Compares de_LDS vs de_movingAve features
- Finds optimal channels and frequency bands
- Creates clean final dataset
- Fast emotion detection with intensity

Emotion Labels: 0=Neutral, 1=Sad, 2=Fear, 3=Happy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class OptimizedEEGClassifier:
    """
    Complete optimized EEG emotion classification system
    """
    
    def __init__(self, data_dir="csv"):
        self.data_dir = Path(data_dir)
        
        # SEED-IV Labels from README
        self.session_labels = {
            1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
            3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
        }
        
        self.emotion_names = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        self.frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        self.scaler = StandardScaler()
        self.model = None
        self.best_features = None
        
    def load_all_data(self):
        """Load and organize all EEG data"""
        print("🔄 Loading complete SEED-IV dataset...")
        print("📊 Dataset structure: 3 sessions × 15 subjects × 48 trials = 2160 files")
        
        all_data = []
        file_count = 0
        
        # Load data systematically by session/subject
        for session in range(1, 4):  # Sessions 1, 2, 3
            for subject in range(1, 16):  # Subjects 1-15
                session_path = self.data_dir / str(session) / str(subject)
                
                if not session_path.exists():
                    continue
                    
                print(f"📁 Loading Session {session}, Subject {subject}...")
                
                # Load both de_LDS and de_movingAve files
                for trial in range(1, 25):  # Trials 1-24
                    lds_file = session_path / f"de_LDS{trial}.csv"
                    moving_file = session_path / f"de_movingAve{trial}.csv"
                    
                    emotion_label = self.session_labels[session][trial-1]
                    
                    # Load LDS features
                    if lds_file.exists():
                        try:
                            lds_data = pd.read_csv(lds_file)
                            lds_features = self._process_trial_data(
                                lds_data, session, subject, trial, emotion_label, 'LDS'
                            )
                            all_data.append(lds_features)
                            file_count += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {lds_file}: {e}")
                    
                    # Load Moving Average features  
                    if moving_file.exists():
                        try:
                            moving_data = pd.read_csv(moving_file)
                            moving_features = self._process_trial_data(
                                moving_data, session, subject, trial, emotion_label, 'MovingAve'
                            )
                            all_data.append(moving_features)
                            file_count += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {moving_file}: {e}")
        
        print(f"✅ Loaded {file_count} files successfully")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n📋 Final Dataset Summary:")
        print(f"   Total samples: {len(combined_df):,}")
        print(f"   Features per sample: {len([c for c in combined_df.columns if c.startswith('Ch')])}")
        print(f"   Sessions: {combined_df['session'].nunique()}")
        print(f"   Subjects: {combined_df['subject'].nunique()}")
        print(f"   Feature types: {combined_df['feature_type'].value_counts().to_dict()}")
        
        print(f"\n🎯 Emotion Distribution:")
        emotion_counts = combined_df['emotion'].value_counts().sort_index()
        for emotion_code, count in emotion_counts.items():
            print(f"   {emotion_code} ({self.emotion_names[emotion_code]}): {count:,} samples")
        
        return combined_df
    
    def _process_trial_data(self, data, session, subject, trial, emotion, feature_type):
        """Process individual trial data"""
        # Calculate statistical features for each channel-frequency combination
        features = {}
        
        # Add metadata
        features['session'] = session
        features['subject'] = subject  
        features['trial'] = trial
        features['emotion'] = emotion
        features['feature_type'] = feature_type
        
        # Extract features for each channel and frequency band
        for i, col in enumerate(data.columns):
            if col.startswith('Ch') and '_Freq' in col:
                channel_data = data[col].values
                
                # Calculate multiple statistical features
                features[f'{col}_mean'] = np.mean(channel_data)
                features[f'{col}_std'] = np.std(channel_data)
                features[f'{col}_max'] = np.max(channel_data)
                features[f'{col}_min'] = np.min(channel_data)
                features[f'{col}_median'] = np.median(channel_data)
                features[f'{col}_range'] = np.max(channel_data) - np.min(channel_data)
        
        return pd.DataFrame([features])
    
    def analyze_feature_types(self, data):
        """Compare de_LDS vs de_movingAve features"""
        print("\n🔍 Analyzing Feature Types (LDS vs MovingAve)...")
        
        feature_cols = [c for c in data.columns if c.startswith('Ch')]
        X = data[feature_cols].values
        y = data['emotion'].values
        
        # Separate by feature type
        lds_data = data[data['feature_type'] == 'LDS']
        moving_data = data[data['feature_type'] == 'MovingAve']
        
        # Test accuracy for each feature type
        results = {}
        
        for name, subset in [('LDS', lds_data), ('MovingAve', moving_data)]:
            if len(subset) > 100:  # Ensure sufficient data
                X_subset = subset[feature_cols].fillna(0).values
                y_subset = subset['emotion'].values
                
                # Quick Random Forest test
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subset, y_subset, test_size=0.3, random_state=42, stratify=y_subset
                )
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                accuracy = rf.score(X_test, y_test)
                
                results[name] = {
                    'accuracy': accuracy,
                    'samples': len(subset),
                    'features': len(feature_cols)
                }
                
                print(f"   {name:12s}: {accuracy:.3f} accuracy ({len(subset):,} samples)")
        
        # Determine best feature type
        best_type = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"\n🏆 Best feature type: {best_type} ({results[best_type]['accuracy']:.3f} accuracy)")
        
        return best_type, results
    
    def find_optimal_features(self, data, feature_type='MovingAve', top_k=50):
        """Find the best channels and frequency bands"""
        print(f"\n🧠 Finding optimal features using {feature_type} data...")
        
        # Filter by best feature type
        filtered_data = data[data['feature_type'] == feature_type].copy()
        feature_cols = [c for c in filtered_data.columns if c.startswith('Ch')]
        
        X = filtered_data[feature_cols].fillna(0).values
        y = filtered_data['emotion'].values
        
        # Feature selection using F-statistic
        selector = SelectKBest(score_func=f_classif, k=min(top_k, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features and their scores
        selected_mask = selector.get_support()
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
        feature_scores = selector.scores_
        
        print(f"✅ Selected {len(selected_features)} best features from {len(feature_cols)} total")
        
        # Analyze channel and frequency importance
        channel_importance = {}
        freq_importance = {f'Freq{i+1}': 0 for i in range(5)}
        
        for i, feature in enumerate(feature_cols):
            if selected_mask[i]:
                # Extract channel and frequency
                parts = feature.split('_')
                if len(parts) >= 2:
                    channel = parts[0]  # Ch1, Ch2, etc.
                    freq = parts[1]     # Freq1, Freq2, etc.
                    
                    channel_importance[channel] = channel_importance.get(channel, 0) + feature_scores[i]
                    if freq in freq_importance:
                        freq_importance[freq] += feature_scores[i]
        
        # Show top channels and frequencies
        print(f"\n🏆 Top 10 Most Important Channels:")
        sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (channel, score) in enumerate(sorted_channels[:10]):
            print(f"   {i+1:2d}. {channel:6s}: {score:8.1f}")
        
        print(f"\n🎵 Frequency Band Importance:")
        freq_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        sorted_freqs = sorted(freq_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (freq, score) in enumerate(sorted_freqs):
            freq_idx = int(freq.replace('Freq', '')) - 1
            band_name = freq_names[freq_idx] if freq_idx < 5 else freq
            print(f"   {i+1}. {band_name:8s} ({freq}): {score:8.1f}")
        
        self.best_features = selected_features
        return selected_features, filtered_data
    
    def train_optimized_model(self, data, selected_features):
        """Train the final optimized model"""
        print(f"\n🎯 Training optimized emotion classification model...")
        
        # Prepare data
        X = data[selected_features].fillna(0).values
        y = data['emotion'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Test multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        }
        
        best_model = None
        best_accuracy = 0
        best_name = ""
        
        print(f"🏋️ Testing different models...")
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            print(f"   {name:15s}: {accuracy:.4f} accuracy")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
        
        self.model = best_model
        print(f"\n🏆 Best model: {best_name} ({best_accuracy:.4f} accuracy)")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[self.emotion_names[i] for i in sorted(self.emotion_names.keys())]))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.emotion_names[i] for i in sorted(self.emotion_names.keys())],
                   yticklabels=[self.emotion_names[i] for i in sorted(self.emotion_names.keys())])
        plt.title('Confusion Matrix - Optimized EEG Emotion Classifier')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        plt.show()
        
        return self.model
    
    def create_final_dataset(self, data, selected_features, filename="final_eeg_dataset.csv"):
        """Create the final clean dataset for production use"""
        print(f"\n💾 Creating final production dataset...")
        
        # Select only the best features and metadata
        final_cols = ['session', 'subject', 'trial', 'emotion', 'feature_type'] + selected_features
        final_data = data[final_cols].copy()
        
        # Add emotion names for clarity
        final_data['emotion_name'] = final_data['emotion'].map(self.emotion_names)
        
        # Save dataset
        output_path = self.data_dir.parent / filename
        final_data.to_csv(output_path, index=False)
        
        print(f"✅ Final dataset saved: {output_path}")
        print(f"   Shape: {final_data.shape}")
        print(f"   Features: {len(selected_features)} optimized features")
        print(f"   Total samples: {len(final_data):,}")
        
        return output_path
    
    def predict_emotion(self, eeg_features):
        """Fast emotion prediction with confidence and intensity"""
        if self.model is None or self.best_features is None:
            raise ValueError("Model must be trained first!")
        
        # Ensure features match training format
        if isinstance(eeg_features, dict):
            # Convert dictionary to array in correct order
            feature_array = [eeg_features.get(feat, 0) for feat in self.best_features]
        else:
            feature_array = eeg_features
        
        # Reshape and scale
        features = np.array(feature_array).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Predict emotion and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Calculate intensity (confidence of prediction)
        max_prob = np.max(probabilities)
        intensity = "Low" if max_prob < 0.6 else "Medium" if max_prob < 0.8 else "High"
        
        result = {
            'emotion': self.emotion_names[prediction],
            'emotion_code': int(prediction),
            'confidence': float(max_prob),
            'intensity': intensity,
            'probabilities': {
                self.emotion_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result


def main():
    """Main execution pipeline"""
    print("🧠 Optimized EEG Emotion Classification System")
    print("=" * 60)
    
    # Initialize classifier
    classifier = OptimizedEEGClassifier(data_dir="csv")
    
    # Step 1: Load all data
    print("\n📋 Step 1: Loading Complete Dataset")
    all_data = classifier.load_all_data()
    
    # Step 2: Compare feature types
    print("\n🔍 Step 2: Analyzing Feature Types")
    best_type, comparison = classifier.analyze_feature_types(all_data)
    
    # Step 3: Find optimal features
    print("\n🧠 Step 3: Finding Optimal Features")
    selected_features, filtered_data = classifier.find_optimal_features(
        all_data, feature_type=best_type, top_k=75
    )
    
    # Step 4: Train model
    print("\n🎯 Step 4: Training Optimized Model")
    model = classifier.train_optimized_model(filtered_data, selected_features)
    
    # Step 5: Create final dataset
    print("\n💾 Step 5: Creating Final Dataset")
    final_dataset_path = classifier.create_final_dataset(filtered_data, selected_features)
    
    print("\n✅ Optimized EEG Emotion Classification System Ready!")
    print("🚀 Use classifier.predict_emotion(features) for real-time predictions")
    print(f"📁 Final dataset: {final_dataset_path}")
    
    # Example prediction
    print("\n🔮 Example prediction:")
    try:
        # Use first sample as example
        sample_features = filtered_data[selected_features].iloc[0].to_dict()
        result = classifier.predict_emotion(sample_features)
        print(f"   Predicted: {result['emotion']} ({result['confidence']:.3f} confidence, {result['intensity']} intensity)")
    except Exception as e:
        print(f"   Example prediction failed: {e}")
    
    return classifier

if __name__ == "__main__":
    classifier = main()
