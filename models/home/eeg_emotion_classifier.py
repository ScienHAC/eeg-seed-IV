"""
EEG Emotion Classification System for SEED-IV Dataset
=====================================================

A comprehensive system for emotion recognition using EEG signals with:
- Intelligent feature selection using sklearn
- Advanced deep learning models (CNN-LSTM hybrid)
- Clean dataset generation
- Real-world application focus

Emotion Labels:
0 = Neutral, 1 = Sad, 2 = Fear, 3 = Happy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class EEGDataProcessor:
    """
    Handles EEG data loading, preprocessing, and feature selection
    """
    
    def __init__(self, data_dir="csv"):
        """
        Initialize the EEG data processor
        
        Args:
            data_dir (str): Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.emotion_labels = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        self.eeg_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.n_channels = 62
        
        # SEED-IV session labels (predefined emotion sequences)
        self.session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
        self.session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
        self.session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
        
        self.all_data = []
        self.all_labels = []
        self.feature_names = []
        
    def load_and_process_data(self):
        """
        Load all CSV files and create a comprehensive dataset
        """
        print("🔄 Loading EEG data from CSV files...")
        
        # Walk through directory structure
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = Path(root) / file
                    
                    # Extract session, subject, and trial info from path
                    path_parts = file_path.parts
                    subject_id = None
                    session_id = None
                    
                    # Parse directory structure: csv/subject/session/file.csv
                    if len(path_parts) >= 4:
                        subject_id = path_parts[-3]
                        session_id = path_parts[-2]
                    
                    print(f"📁 Processing: {file_path}")
                    
                    try:
                        # Load CSV data
                        data = pd.read_csv(file_path)
                        
                        # Skip if data is empty or too small
                        if len(data) < 10:
                            print(f"⚠️  Skipping {file} - insufficient data")
                            continue
                        
                        # Rename columns to standard format if needed
                        data = self._standardize_column_names(data)
                        
                        # Get emotion label for this trial
                        emotion_label = self._get_emotion_label(file, session_id)
                        
                        if emotion_label is not None:
                            # Add metadata columns
                            data['subject'] = subject_id
                            data['session'] = session_id
                            data['trial'] = file.replace('.csv', '')
                            data['emotion'] = emotion_label
                            
                            self.all_data.append(data)
                            print(f"✅ Loaded: {len(data)} samples, emotion: {self.emotion_labels[emotion_label]}")
                        
                    except Exception as e:
                        print(f"❌ Error loading {file}: {e}")
        
        if not self.all_data:
            raise ValueError("No valid data files found!")
        
        print(f"\n📊 Total files loaded: {len(self.all_data)}")
        return self._combine_data()
    
    def _standardize_column_names(self, data):
        """
        Standardize column names to Ch<X>_<band> format
        """
        # If columns are already in correct format, return as is
        if any('Ch' in col and '_' in col for col in data.columns):
            return data
        
        # Generate standard column names
        new_columns = []
        for ch in range(1, self.n_channels + 1):
            for band in self.eeg_bands:
                new_columns.append(f'Ch{ch}_{band}')
        
        # Only rename if we have the expected number of columns
        if len(data.columns) == len(new_columns):
            data.columns = new_columns
        
        return data
    
    def _get_emotion_label(self, filename, session_id):
        """
        Get emotion label based on filename and session
        """
        try:
            # Extract trial number from filename
            # Assuming format like 'de_movingAve1.csv' -> trial 1
            trial_num = int(''.join(filter(str.isdigit, filename.split('.')[0])))
            
            # Adjust for 0-based indexing
            trial_idx = trial_num - 1
            
            # Get label based on session
            if session_id == '1' and trial_idx < len(self.session1_label):
                return self.session1_label[trial_idx]
            elif session_id == '2' and trial_idx < len(self.session2_label):
                return self.session2_label[trial_idx]
            elif session_id == '3' and trial_idx < len(self.session3_label):
                return self.session3_label[trial_idx]
            else:
                # Default mapping if session info not available
                return trial_idx % 4  # Cycle through 0,1,2,3
                
        except:
            return 0  # Default to neutral if parsing fails
    
    def _combine_data(self):
        """
        Combine all data into a single DataFrame
        """
        print("🔗 Combining all data...")
        combined_df = pd.concat(self.all_data, ignore_index=True)
        
        # Separate features from metadata
        feature_columns = [col for col in combined_df.columns 
                          if col.startswith('Ch') and '_' in col]
        metadata_columns = ['subject', 'session', 'trial', 'emotion']
        
        self.feature_names = feature_columns
        
        print(f"📋 Dataset Summary:")
        print(f"   Total samples: {len(combined_df):,}")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Emotion distribution:")
        
        emotion_counts = combined_df['emotion'].value_counts().sort_index()
        for emotion_code, count in emotion_counts.items():
            emotion_name = self.emotion_labels.get(emotion_code, 'Unknown')
            print(f"     {emotion_code} ({emotion_name}): {count:,} samples")
        
        return combined_df
    
    def perform_feature_selection(self, data, method='f_classif', k=100):
        """
        Perform intelligent feature selection
        
        Args:
            data (DataFrame): Combined dataset
            method (str): 'f_classif' or 'mutual_info_classif'
            k (int): Number of top features to select
        
        Returns:
            tuple: (selected_data, feature_scores, selected_features)
        """
        print(f"\n🧠 Performing feature selection using {method}...")
        
        # Prepare features and labels
        feature_columns = [col for col in data.columns 
                          if col.startswith('Ch') and '_' in col]
        X = data[feature_columns].values
        y = data['emotion'].values
        
        # Handle missing values
        X = np.nan_to_num(X)
        
        # Select method
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features and scores
        selected_mask = selector.get_support()
        selected_features = [feature_columns[i] for i in range(len(feature_columns)) 
                           if selected_mask[i]]
        feature_scores = selector.scores_
        
        print(f"✅ Selected {len(selected_features)} most informative features")
        
        # Create new dataframe with selected features
        selected_data = data.copy()
        # Keep only selected feature columns + metadata
        columns_to_keep = selected_features + ['subject', 'session', 'trial', 'emotion']
        selected_data = selected_data[columns_to_keep]
        
        # Show top features
        print(f"\n🏆 Top 10 most important features:")
        feature_importance = list(zip(feature_columns, feature_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(feature_importance[:10]):
            channel = feature.split('_')[0]
            band = feature.split('_')[1]
            print(f"   {i+1:2d}. {feature:15s} (Score: {score:8.2f}) - {channel} {band} band")
        
        return selected_data, feature_scores, selected_features
    
    def save_clean_dataset(self, data, filename="clean_eeg_dataset.csv"):
        """
        Save the cleaned and feature-selected dataset
        """
        output_path = self.data_dir.parent / filename
        data.to_csv(output_path, index=False)
        print(f"\n💾 Clean dataset saved to: {output_path}")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        return output_path


class AdvancedEEGModel(nn.Module):
    """
    Advanced PyTorch model combining CNN and LSTM for EEG emotion classification
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        super(AdvancedEEGModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(256, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Reshape for 1D conv: (batch, channels, features)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, channels) -> (batch, channels, features)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EEGEmotionClassifier:
    """
    Complete EEG emotion classification system
    """
    
    def __init__(self, data_dir="csv"):
        self.data_processor = EEGDataProcessor(data_dir)
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
    def prepare_data(self, feature_selection_method='f_classif', k_features=100):
        """
        Complete data preparation pipeline
        """
        # Load raw data
        raw_data = self.data_processor.load_and_process_data()
        
        # Perform feature selection
        clean_data, feature_scores, selected_features = self.data_processor.perform_feature_selection(
            raw_data, method=feature_selection_method, k=k_features
        )
        
        # Save clean dataset
        clean_file = self.data_processor.save_clean_dataset(clean_data)
        
        return clean_data, selected_features
    
    def train_model(self, data, selected_features, test_size=0.2, epochs=100, batch_size=32):
        """
        Train the advanced EEG emotion classification model
        """
        print(f"\n🎯 Training advanced EEG emotion classification model...")
        
        # Prepare features and labels
        X = data[selected_features].values
        y = data['emotion'].values
        
        # Handle missing values and scale features
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = AdvancedEEGModel(input_size=input_size).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        print(f"🏋️ Training model for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = accuracy_score(y_test, test_predicted.cpu().numpy())
        
        print(f"\n🎉 Training completed!")
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        
        # Plot training history
        self._plot_training_history(train_losses, train_accuracies)
        
        # Detailed evaluation
        self._evaluate_model(X_test_tensor, y_test)
        
        return self.model
    
    def _plot_training_history(self, losses, accuracies):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(accuracies)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _evaluate_model(self, X_test, y_test):
        """Detailed model evaluation"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
            
        # Handle both tensor and numpy array inputs
        if torch.is_tensor(y_test):
            y_true = y_test.cpu().numpy()
        else:
            y_true = y_test
        y_pred = predicted.cpu().numpy()
        
        # Classification report
        print("\n📊 Detailed Classification Report:")
        emotion_names = ['Neutral', 'Sad', 'Fear', 'Happy']
        print(classification_report(y_true, y_pred, target_names=emotion_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_names, yticklabels=emotion_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def predict_emotion(self, eeg_features):
        """
        Predict emotion from EEG features
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        # Preprocess features
        if isinstance(eeg_features, list):
            eeg_features = np.array(eeg_features)
        
        if len(eeg_features.shape) == 1:
            eeg_features = eeg_features.reshape(1, -1)
        
        eeg_features = np.nan_to_num(eeg_features)
        eeg_features = self.scaler.transform(eeg_features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(eeg_features).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Format results
        emotion_labels = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        
        results = []
        for i, (pred_class, prob) in enumerate(zip(predicted.cpu(), probabilities.cpu())):
            result = {
                'predicted_emotion': emotion_labels[pred_class.item()],
                'emotion_code': pred_class.item(),
                'confidence': float(torch.max(prob)),
                'emotion_probabilities': {
                    emotion_labels[j]: float(prob[j]) for j in range(len(prob))
                }
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results


# Main execution
if __name__ == "__main__":
    print("🧠 EEG Emotion Classification System")
    print("=" * 60)
    
    # Initialize classifier
    classifier = EEGEmotionClassifier(data_dir="csv")
    
    # Prepare data with feature selection
    print("\n📋 Step 1: Data Preparation and Feature Selection")
    clean_data, selected_features = classifier.prepare_data(
        feature_selection_method='f_classif',  # or 'mutual_info_classif'
        k_features=150  # Select top 150 features
    )
    
    # Train model
    print("\n🎯 Step 2: Model Training")
    model = classifier.train_model(
        data=clean_data,
        selected_features=selected_features,
        epochs=50,
        batch_size=64
    )
    
    print("\n✅ EEG Emotion Classification System Ready!")
    print("💡 You can now use classifier.predict_emotion(eeg_features) for predictions")
