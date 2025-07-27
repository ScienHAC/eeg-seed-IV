"""
EEG Emotion Recognition - Optimized Ensemble Approach
====================================================

Based on successful eye-tracking methodology but adapted for EEG data:
- Advanced feature engineering for EEG signals
- Proven ensemble: Random Forest + Extra Trees + XGBoost + LightGBM
- Multi-step feature selection optimized for 1080 samples
- Goal: Beat 50% baseline significantly (target 65-75%)

Differences from eye-tracking approach:
1. EEG-specific preprocessing (band-pass filtering, artifact removal)
2. Spectral features (power in different frequency bands)
3. Cross-channel connectivity features
4. Temporal dynamics specific to EEG signals
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Advanced ensemble models (proven successful)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb

# Signal processing for EEG
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt, welch

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time

class EEGEmotionEnsemble:
    """
    Optimized EEG Emotion Recognition using proven ensemble methods
    Adapted from successful eye-tracking approach for EEG brain signals
    """
    
    def __init__(self):
        self.emotion_labels = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        self.X_features = None
        self.y_labels = None
        self.feature_names = []
        self.scaler = None
        self.feature_selector = None
        self.ensemble_model = None
        
    def load_successful_sfs_data(self):
        """Load data using your successful SFS preprocessing"""
        print("🔄 Loading data with successful SFS preprocessing...")
        
        try:
            # Import your successful SFS module
            import importlib.util
            
            sfs_path = Path("models/sequential_feature_selection/clean_eeg_classifier.py").resolve()
            if not sfs_path.exists():
                print("❌ SFS module not found")
                return False
                
            spec = importlib.util.spec_from_file_location("clean_eeg_classifier", sfs_path)
            clean_eeg_classifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(clean_eeg_classifier)
            
            # Load data using your successful method
            X, y, metadata = clean_eeg_classifier.load_clean_seed_iv_data("csv", "de_LDS", max_subjects=15)
            
            if len(X) == 0:
                print("❌ Could not load SEED-IV data")
                return False
            
            # Store raw data for advanced feature extraction
            self.X_raw = X
            self.y_labels = y
            
            print(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} original features")
            print(f"🎯 Label distribution: {Counter(y)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def advanced_eeg_feature_extraction(self):
        """
        Advanced EEG feature extraction - much more comprehensive than basic stats
        Based on successful eye-tracking approach but adapted for EEG
        """
        print("\n⚡ Advanced EEG feature extraction...")
        
        all_features = []
        feature_names = []
        
        for sample_idx in range(len(self.X_raw)):
            sample_features = []
            
            # Reshape data: assume 310 features = 62 channels × 5 frequency bands
            data = self.X_raw[sample_idx].reshape(62, 5) if len(self.X_raw[sample_idx]) == 310 else self.X_raw[sample_idx]
            
            # 1. ENHANCED STATISTICAL FEATURES (per frequency band)
            stats_features = self._extract_enhanced_statistical_features(data)
            sample_features.extend(stats_features)
            
            # 2. SPECTRAL POWER FEATURES (EEG-specific)
            spectral_features = self._extract_spectral_power_features(data)
            sample_features.extend(spectral_features)
            
            # 3. CROSS-CHANNEL CONNECTIVITY (EEG brain connectivity)
            connectivity_features = self._extract_connectivity_features(data)
            sample_features.extend(connectivity_features)
            
            # 4. TEMPORAL DYNAMICS (adapted for EEG)
            temporal_features = self._extract_temporal_dynamics(data)
            sample_features.extend(temporal_features)
            
            # 5. FREQUENCY BAND RATIOS (EEG emotion indicators)
            ratio_features = self._extract_frequency_ratios(data)
            sample_features.extend(ratio_features)
            
            all_features.append(sample_features)
        
        self.X_features = np.array(all_features)
        
        # Handle NaN values
        self.X_features = np.nan_to_num(self.X_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✅ Advanced feature extraction complete: {self.X_features.shape}")
        print(f"📊 Feature enhancement: {len(self.X_raw[0])} → {self.X_features.shape[1]} features")
        
    def _extract_enhanced_statistical_features(self, data):
        """Enhanced statistical features for EEG (much better than basic mean/std)"""
        features = []
        
        # For each frequency band (columns)
        for band_idx in range(data.shape[1] if len(data.shape) > 1 else 1):
            if len(data.shape) > 1:
                band_data = data[:, band_idx]  # All channels for this frequency band
            else:
                band_data = data
                
            # Advanced statistical measures
            features.extend([
                np.mean(band_data),
                np.std(band_data),
                np.median(band_data),
                np.percentile(band_data, 25),
                np.percentile(band_data, 75),
                np.max(band_data) - np.min(band_data),  # Range
                skew(band_data),                        # Asymmetry
                kurtosis(band_data),                    # Tail heaviness
                np.sqrt(np.mean(band_data**2)),         # RMS
                np.mean(np.abs(band_data))              # Mean absolute value
            ])
        
        return features
    
    def _extract_spectral_power_features(self, data):
        """EEG-specific spectral power features"""
        features = []
        
        # If we have multiple channels, analyze key channels
        n_channels = min(10, data.shape[0] if len(data.shape) > 1 else 1)
        
        for ch_idx in range(n_channels):
            if len(data.shape) > 1:
                channel_data = data[ch_idx, :]
            else:
                channel_data = data
                
            # Spectral power in different bands
            if len(channel_data) > 4:
                # Relative power in frequency bands
                total_power = np.sum(channel_data**2)
                if total_power > 0:
                    features.extend([
                        np.sum(channel_data[:2]**2) / total_power,  # Delta-like
                        np.sum(channel_data[1:3]**2) / total_power, # Theta-like  
                        np.sum(channel_data[2:4]**2) / total_power, # Alpha-like
                        np.sum(channel_data[3:]**2) / total_power,  # Beta-like
                    ])
                else:
                    features.extend([0, 0, 0, 0])
                    
                # Peak frequency
                peak_idx = np.argmax(np.abs(channel_data))
                features.append(peak_idx / len(channel_data))
            else:
                features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def _extract_connectivity_features(self, data):
        """Cross-channel connectivity features (EEG brain connectivity)"""
        features = []
            
        if len(data.shape) > 1 and data.shape[0] > 1:
            # Cross-correlation between channels
            n_channels = min(8, data.shape[0])  # Limit to prevent explosion
            
            correlations = []
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    if np.std(data[i, :]) > 1e-6 and np.std(data[j, :]) > 1e-6:
                        corr = np.corrcoef(data[i, :], data[j, :])[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0)
            
            if correlations:
                features.extend([
                    np.mean(correlations),
                    np.std(correlations),
                    np.max(correlations),
                    np.min(correlations),
                    np.sum(np.array(correlations) > 0.7) / len(correlations)  # Strong connectivity
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
                
            # Global connectivity measures
            if data.shape[0] > 2:
                # Average correlation strength
                corr_matrix = np.corrcoef(data)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0)
                upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                
                features.extend([
                    np.mean(np.abs(upper_triangle)),
                    np.std(upper_triangle),
                    np.sum(np.abs(upper_triangle) > 0.5) / len(upper_triangle)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0] * 8)  # Default values
            
        return features
    
    def _extract_temporal_dynamics(self, data):
        """Temporal dynamics adapted for EEG"""
        features = []
        
        # For key channels/features
        n_features = min(5, data.shape[0] if len(data.shape) > 1 else 1)
        
        for feat_idx in range(n_features):
            if len(data.shape) > 1:
                signal_data = data[feat_idx, :]
            else:
                signal_data = data
                
            if len(signal_data) > 2 and np.std(signal_data) > 1e-6:
                # Temporal complexity
                diff1 = np.diff(signal_data)
                diff2 = np.diff(diff1) if len(diff1) > 1 else [0]
                
                features.extend([
                    np.std(diff1) / np.std(signal_data),  # Mobility
                    np.std(diff2) / np.std(diff1) if np.std(diff1) > 1e-6 else 0,  # Complexity
                    np.mean(np.abs(diff1)),  # Average change
                    len(signal_data[signal_data > np.median(signal_data)]) / len(signal_data)  # Above median ratio
                ])
            else:
                features.extend([0, 0, 0, 0])
                
        return features
    
    def _extract_frequency_ratios(self, data):
        """Frequency band ratios (important for EEG emotion recognition)"""
        features = []
        
        if len(data.shape) > 1 and data.shape[1] >= 5:
            # Assume columns are frequency bands
            band_powers = np.mean(data**2, axis=0)  # Power in each band
            
            # Important EEG ratios for emotion
            total_power = np.sum(band_powers)
            if total_power > 0:
                features.extend([
                    band_powers[0] / band_powers[1] if band_powers[1] > 0 else 0,  # Delta/Theta
                    band_powers[2] / band_powers[3] if band_powers[3] > 0 else 0,  # Alpha/Beta  
                    band_powers[4] / total_power,  # Gamma ratio
                    (band_powers[0] + band_powers[1]) / total_power,  # Low freq ratio
                    (band_powers[3] + band_powers[4]) / total_power,  # High freq ratio
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
                
            # Cross-band relationships
            features.extend([
                np.std(band_powers) / np.mean(band_powers) if np.mean(band_powers) > 0 else 0,
                np.max(band_powers) / np.min(band_powers) if np.min(band_powers) > 0 else 0
            ])
        else:
            features.extend([0] * 7)
            
        return features
    
    def optimized_feature_selection(self):
        """
        Optimized feature selection - adapted from successful eye-tracking approach
        4-step process: variance → correlation → statistical → RFE
        """
        print("\n🎯 Optimized 4-step feature selection...")
        
        # Step 1: Remove very low variance features
        print("Step 1: Variance filtering...")
        var_threshold = 0.0001  # More conservative for EEG
        var_selector = VarianceThreshold(threshold=var_threshold)
        X_var = var_selector.fit_transform(self.X_features)
        print(f"   {self.X_features.shape[1]} → {X_var.shape[1]} features")
        
        # Step 2: Remove highly correlated features  
        print("Step 2: Correlation filtering...")
        corr_matrix = np.corrcoef(X_var.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0)
        
        # Find highly correlated pairs
        high_corr_pairs = np.where(np.abs(corr_matrix) > 0.95)
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i != j and i not in features_to_remove:
                features_to_remove.add(j)
        
        features_to_keep = [i for i in range(X_var.shape[1]) if i not in features_to_remove]
        X_decorr = X_var[:, features_to_keep]
        print(f"   {X_var.shape[1]} → {X_decorr.shape[1]} features")
        
        # Step 3: Statistical feature selection (F-test + Mutual Information)
        print("Step 3: Hybrid statistical selection...")
        k_stats = min(50, X_decorr.shape[1]//2, len(self.y_labels)//2)
        
        # Combine F-test and mutual information
        f_scores = f_classif(X_decorr, self.y_labels)[0]
        f_scores = np.nan_to_num(f_scores, nan=0)
        
        try:
            mi_scores = mutual_info_classif(X_decorr, self.y_labels, random_state=42)
            mi_scores = np.nan_to_num(mi_scores, nan=0)
            combined_scores = 0.7 * f_scores + 0.3 * mi_scores
        except:
            combined_scores = f_scores
        
        top_indices = np.argsort(combined_scores)[-k_stats:]
        X_stats = X_decorr[:, top_indices]
        print(f"   {X_decorr.shape[1]} → {X_stats.shape[1]} features")
        
        # Step 4: RFE with Random Forest (proven effective)
        print("Step 4: RFE selection...")
        n_final = min(25, X_stats.shape[1]//2)  # More features than original
        
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced'
        )
        
        rfe_selector = RFE(rf_selector, n_features_to_select=n_final, step=1)
        X_final = rfe_selector.fit_transform(X_stats, self.y_labels)
        print(f"   {X_stats.shape[1]} → {X_final.shape[1]} features")
        
        print(f"✅ Feature selection complete: {self.X_features.shape[1]} → {X_final.shape[1]}")
        
        # Store selector for later use
        self.feature_selector = {
            'variance': var_selector,
            'correlation_indices': features_to_keep,
            'stats_indices': top_indices,
            'rfe': rfe_selector
        }
        
        return X_final
    
    def create_optimized_ensemble(self, X_train, y_train):
        """
        Create optimized ensemble using proven successful models:
        Random Forest + Extra Trees + XGBoost + LightGBM
        """
        print("\n🤖 Creating optimized ensemble...")
        
        # Model 1: Random Forest (proven successful)
        rf_model = RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        
        # Model 2: Extra Trees (extremely randomized trees)
        et_model = ExtraTreesClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        
        # Model 3: XGBoost (gradient boosting)
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Model 4: LightGBM (optimized gradient boosting)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            random_state=42,
            verbose=-1
        )
        
        # Evaluate individual models
        models = {
            'Random Forest': rf_model,
            'Extra Trees': et_model,
            'XGBoost': xgb_model,
            'LightGBM': lgb_model
        }
        
        model_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("📊 Individual model evaluation:")
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                model_scores[name] = {
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                    'model': model
                }
                print(f"   {name}: {scores.mean():.3f} ± {scores.std():.3f}")
            except Exception as e:
                print(f"   {name}: Failed - {e}")
        
        # Create ensemble from all successful models
        successful_models = [(name.lower().replace(' ', '_'), results['model']) 
                           for name, results in model_scores.items()]
        
        if len(successful_models) >= 2:
            ensemble = VotingClassifier(
                estimators=successful_models,
                voting='soft'
            )
            print(f"✅ Created ensemble with {len(successful_models)} models")
        else:
            # Fallback to best individual model
            ensemble = rf_model
            print("⚠️ Using Random Forest fallback")
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        self.ensemble_model = ensemble
        
        return ensemble, model_scores
    
    def run_optimized_pipeline(self):
        """
        Run the complete optimized EEG emotion recognition pipeline
        Goal: Significantly beat 50% baseline
        """
        print("🚀 OPTIMIZED EEG EMOTION RECOGNITION PIPELINE")
        print("=" * 60)
        print("🎯 Goal: Beat 50% baseline significantly (target 65%+)")
        print("🧠 Method: Advanced ensemble adapted from successful eye-tracking approach")
        print("⚡ Features: Enhanced EEG-specific feature extraction")
        
        start_time = time.time()
        
        # Step 1: Load data using successful SFS preprocessing
        if not self.load_successful_sfs_data():
            return None
        
        # Step 2: Advanced EEG feature extraction  
        self.advanced_eeg_feature_extraction()
        
        # Step 3: Optimized feature selection (4-step process)
        X_selected = self.optimized_feature_selection()
        
        # Step 4: Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, self.y_labels,
            test_size=0.25,
            random_state=42,
            stratify=self.y_labels
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples") 
        print(f"   Features: {X_selected.shape[1]}")
        
        # Step 5: Robust scaling
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 6: Create and train optimized ensemble
        ensemble, model_scores = self.create_optimized_ensemble(X_train_scaled, y_train)
        
        # Step 7: Final evaluation
        y_pred = ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation on full training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        total_time = time.time() - start_time
        
        # Results summary
        print(f"\n" + "=" * 60)
        print("🏆 OPTIMIZED ENSEMBLE RESULTS")
        print("=" * 60)
        print(f"📈 Test Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📈 Test F1-Score:       {f1:.4f}")
        print(f"📈 CV Accuracy:         {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"📊 Features Used:       {X_selected.shape[1]}")
        print(f"⏱️ Total Time:          {total_time:.2f} seconds")
        
        # Performance assessment
        baseline = 0.50
        improvement = (accuracy - baseline) / baseline * 100
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print(f"   Baseline (SFS):      50.0%")
        print(f"   Optimized Ensemble:  {accuracy*100:.1f}%")
        print(f"   Improvement:         {improvement:+.1f}%")
        
        if accuracy > 0.70:
            print("🟢 EXCELLENT: >70% - Outstanding for EEG!")
        elif accuracy > 0.65:
            print("🟢 VERY GOOD: 65-70% - Excellent for EEG!")
        elif accuracy > 0.60:
            print("🟡 GOOD: 60-65% - Solid improvement!")
        elif accuracy > 0.55:
            print("🟡 BETTER: 55-60% - Good improvement!")
        else:
            print("🟠 MARGINAL: <55% - Still needs work")
        
        # Detailed results
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, 
                                  target_names=list(self.emotion_labels.values())))
        
        # Visualization
        self._create_results_visualization(y_test, y_pred, accuracy, model_scores)
        
        # Save results
        results = {
            'method': 'Optimized EEG Ensemble',
            'test_accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'improvement_over_baseline': improvement,
            'feature_count': X_selected.shape[1],
            'total_time': total_time,
            'model_scores': model_scores
        }
        
        pd.DataFrame([results]).to_csv('optimized_eeg_ensemble_results.csv', index=False)
        
        return results
    
    def _create_results_visualization(self, y_test, y_pred, accuracy, model_scores):
        """Create comprehensive results visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'🧠 Optimized EEG Ensemble - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=list(self.emotion_labels.values()),
                   yticklabels=list(self.emotion_labels.values()))
        ax1.set_title('🔄 Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Per-class performance
        report = classification_report(y_test, y_pred, 
                                     target_names=list(self.emotion_labels.values()),
                                     output_dict=True)
        
        emotions = list(self.emotion_labels.values())
        precision = [report[e]['precision'] for e in emotions]
        recall = [report[e]['recall'] for e in emotions]
        f1_scores = [report[e]['f1-score'] for e in emotions]
        
        x = np.arange(len(emotions))
        width = 0.25
        
        ax2.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax2.bar(x, recall, width, label='Recall', alpha=0.8)
        ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax2.set_xlabel('Emotions')
        ax2.set_ylabel('Score')
        ax2.set_title('🎭 Per-Class Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(emotions)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Model comparison
        if model_scores:
            models = list(model_scores.keys())
            cv_means = [model_scores[m]['cv_mean'] for m in models]
            cv_stds = [model_scores[m]['cv_std'] for m in models]
            
            ax3.bar(range(len(models)), cv_means, yerr=cv_stds, 
                   capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
            ax3.axhline(y=0.50, color='red', linestyle='--', label='SFS Baseline (50%)')
            ax3.set_xlabel('Models')
            ax3.set_ylabel('CV Accuracy')
            ax3.set_title('🤖 Individual Model Performance')
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels([m.replace(' ', '\n') for m in models])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Overall performance summary
        improvement = (accuracy - 0.50) / 0.50 * 100
        
        ax4.text(0.5, 0.7, 'Optimized\nEnsemble', ha='center', va='center',
                transform=ax4.transAxes, fontsize=16, fontweight='bold')
        
        ax4.text(0.5, 0.5, f'{accuracy:.3f}', ha='center', va='center',
                transform=ax4.transAxes, fontsize=32, fontweight='bold',
                color='green' if accuracy > 0.65 else 'orange' if accuracy > 0.55 else 'red')
        
        ax4.text(0.5, 0.3, f'{improvement:+.1f}% vs SFS', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14,
                color='green' if improvement > 20 else 'orange' if improvement > 10 else 'red')
        
        status = "SUCCESS!" if accuracy > 0.60 else "IMPROVED" if accuracy > 0.55 else "NEEDS WORK"
        ax4.text(0.5, 0.1, status, ha='center', va='center',
                transform=ax4.transAxes, fontsize=12, fontweight='bold',
                color='green' if accuracy > 0.60 else 'orange' if accuracy > 0.55 else 'red')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('optimized_eeg_ensemble_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_optimized_eeg_ensemble():
    """Main function to run the optimized EEG ensemble"""
    classifier = EEGEmotionEnsemble()
    
    print("\n🧠 OPTIMIZED EEG EMOTION RECOGNITION")
    print("=" * 50)
    print("✨ Based on successful eye-tracking methodology")
    print("🎯 Adapted specifically for EEG brain signals")
    print("🚀 Goal: Significantly beat 50% baseline")
    
    results = classifier.run_optimized_pipeline()
    
    if results:
        print(f"\n🎉 PIPELINE COMPLETED!")
        print(f"🏆 FINAL ACCURACY: {results['test_accuracy']:.3f} ({results['test_accuracy']*100:.1f}%)")
        print(f"📈 IMPROVEMENT: {results['improvement_over_baseline']:+.1f}% over baseline")
        
        if results['test_accuracy'] > 0.65:
            print("🎯 OUTSTANDING: Excellent performance for EEG emotion recognition!")
        elif results['test_accuracy'] > 0.60:
            print("🎯 VERY GOOD: Solid improvement over baseline!")
        elif results['test_accuracy'] > 0.55:
            print("🎯 GOOD: Meaningful improvement achieved!")
        else:
            print("🎯 MARGINAL: Some improvement, but more work needed")
            
    return results

if __name__ == "__main__":
    results = run_optimized_eeg_ensemble()
