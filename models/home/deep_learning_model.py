"""
Deep Learning Emotion Recognition Model
=======================================

Advanced deep learning model for SEED-IV emotion recognition using TensorFlow/Keras.
This model implements CNN-LSTM architecture optimized for EEG signal processing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import (
    Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    Flatten, Input, concatenate, Attention, GlobalAveragePooling1D
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DeepEmotionRecognizer:
    """
    Deep Learning model for emotion recognition from EEG signals.
    
    This class implements a sophisticated neural network architecture
    combining CNN and LSTM layers for temporal-spatial feature extraction.
    """
    
    def __init__(self, n_channels=62, n_frequencies=5, n_classes=4):
        """
        Initialize the deep learning model.
        
        Args:
            n_channels (int): Number of EEG channels
            n_frequencies (int): Number of frequency bands
            n_classes (int): Number of emotion classes
        """
        self.n_channels = n_channels
        self.n_frequencies = n_frequencies
        self.n_classes = n_classes
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_cnn_lstm_model(self, input_shape):
        """
        Build CNN-LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tf.keras.Model: Compiled model
        """
        model = Sequential([
            # Convolutional layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # LSTM layers for temporal dependencies
            LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(50, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers for classification
            Dense(100, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')
        ])
        
        return model
    
    def build_attention_model(self, input_shape):
        """
        Build model with attention mechanism.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tf.keras.Model: Compiled model
        """
        inputs = Input(shape=input_shape)
        
        # CNN feature extraction
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.25)(x)
        
        x = Conv1D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.25)(x)
        
        # LSTM with attention
        lstm_out = LSTM(100, return_sequences=True, dropout=0.2)(x)
        
        # Global average pooling acts as attention mechanism
        attention = GlobalAveragePooling1D()(lstm_out)
        
        # Final classification layers
        x = Dense(100, activation='relu')(attention)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def prepare_data(self, X, sequence_length=None):
        """
        Prepare EEG data for deep learning model.
        
        Args:
            X (np.array): Input features
            sequence_length (int): Length of sequences for LSTM
            
        Returns:
            np.array: Reshaped data
        """
        if sequence_length is None:
            # Reshape for Conv1D: (samples, time_steps, features)
            # Assuming features are arranged as (channel1_freq1, channel1_freq2, ...)
            n_samples = X.shape[0]
            n_features = X.shape[1]
            
            # Reshape to (samples, channels, frequencies)
            X_reshaped = X.reshape(n_samples, self.n_channels, self.n_frequencies)
            
        else:
            # Create sequences for LSTM
            X_reshaped = self._create_sequences(X, sequence_length)
        
        return X_reshaped
    
    def _create_sequences(self, X, sequence_length):
        """Create sequences for LSTM processing."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # For demonstration, we'll create overlapping sequences
        sequences = []
        for i in range(0, n_samples - sequence_length + 1, sequence_length // 2):
            seq = X[i:i + sequence_length]
            if len(seq) == sequence_length:
                sequences.append(seq)
        
        return np.array(sequences)
    
    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the deep learning model.
        
        Args:
            X (np.array): Training features
            y (np.array): Training labels
            validation_split (float): Validation data proportion
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        print("Preparing data for deep learning...")
        
        # Prepare data
        X_prepared = self.prepare_data(X)
        y_categorical = to_categorical(y, num_classes=self.n_classes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_prepared, y_categorical, test_size=validation_split, random_state=42
        )
        
        # Scale data
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_val_scaled = self.scaler.transform(X_val_flat)
        
        X_train_final = X_train_scaled.reshape(original_shape)
        X_val_final = X_val_scaled.reshape(X_val.shape)
        
        print(f"Training data shape: {X_train_final.shape}")
        print(f"Validation data shape: {X_val_final.shape}")
        
        # Build model
        input_shape = X_train_final.shape[1:]
        self.model = self.build_attention_model(input_shape)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7)
        ]
        
        # Train model
        print("Starting training...")
        history = self.model.fit(
            X_train_final, y_train,
            validation_data=(X_val_final, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        # Prepare test data
        X_test_prepared = self.prepare_data(X_test)
        y_test_categorical = to_categorical(y_test, num_classes=self.n_classes)
        
        # Scale test data
        X_test_flat = X_test_prepared.reshape(X_test_prepared.shape[0], -1)
        X_test_scaled = self.scaler.transform(X_test_flat)
        X_test_final = X_test_scaled.reshape(X_test_prepared.shape)
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test_final, y_test_categorical, verbose=0)
        
        # Predictions
        y_pred_prob = self.model.predict(X_test_final)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_prob
        }
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Training history from model.fit()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training & validation accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_emotion(self, eeg_features):
        """
        Predict emotion from EEG features.
        
        Args:
            eeg_features (np.array): EEG features
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        # Prepare data
        if len(eeg_features.shape) == 1:
            eeg_features = eeg_features.reshape(1, -1)
        
        X_prepared = self.prepare_data(eeg_features)
        
        # Scale data
        X_flat = X_prepared.reshape(X_prepared.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        X_final = X_scaled.reshape(X_prepared.shape)
        
        # Predict
        predictions = self.model.predict(X_final)
        predicted_class = np.argmax(predictions, axis=1)
        
        emotion_labels = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        
        results = []
        for i, (pred_class, pred_prob) in enumerate(zip(predicted_class, predictions)):
            result = {
                'predicted_emotion': emotion_labels[pred_class],
                'emotion_code': pred_class,
                'confidence': float(np.max(pred_prob)),
                'emotion_probabilities': {
                    emotion_labels[j]: float(pred_prob[j]) for j in range(len(pred_prob))
                }
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results


# Demo/test section
if __name__ == "__main__":
    print("Deep Learning Emotion Recognition Model")
    print("=" * 50)
    
    # Create an instance of the model
    print("Creating DeepEmotionRecognizer instance...")
    recognizer = DeepEmotionRecognizer(n_channels=62, n_frequencies=5, n_classes=4)
    print(f"✓ Model created with:")
    print(f"  - Channels: {recognizer.n_channels}")
    print(f"  - Frequency bands: {recognizer.n_frequencies}")
    print(f"  - Emotion classes: {recognizer.n_classes}")
    
    # Generate some dummy data for demonstration
    print("\nGenerating demo data...")
    n_samples = 1000
    n_features = recognizer.n_channels * recognizer.n_frequencies  # 62 * 5 = 310
    
    # Create synthetic EEG-like data
    np.random.seed(42)
    X_demo = np.random.randn(n_samples, n_features) * 0.1
    y_demo = np.random.randint(0, recognizer.n_classes, n_samples)
    
    print(f"✓ Generated demo data:")
    print(f"  - Features shape: {X_demo.shape}")
    print(f"  - Labels shape: {y_demo.shape}")
    print(f"  - Emotion distribution: {np.bincount(y_demo)}")
    
    # Test data preparation
    print("\nTesting data preparation...")
    X_prepared = recognizer.prepare_data(X_demo)
    print(f"✓ Prepared data shape: {X_prepared.shape}")
    
    # Test model building (without training)
    print("\nTesting model architecture...")
    input_shape = X_prepared.shape[1:]
    test_model = recognizer.build_attention_model(input_shape)
    print("✓ Model architecture created successfully!")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Total parameters: {test_model.count_params():,}")
    
    # Show model summary
    print("\nModel Summary:")
    print("-" * 30)
    test_model.summary()
    
    print("\n" + "=" * 50)
    print("Deep Learning Model Test Complete!")
    print("To train the model, use:")
    print("recognizer.train_model(X_demo, y_demo, epochs=10)")
