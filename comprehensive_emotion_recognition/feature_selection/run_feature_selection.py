"""
Comprehensive Feature Selection for EEG Emotion Recognition
===========================================================

This script runs feature selection on ALL 310 DE features from the SEED-IV dataset.

PAUSE/RESUME FUNCTIONALITY:
- Press Ctrl+C at any time to gracefully stop the experiment
- Results are automatically saved to checkpoint.joblib
- Run the script again to resume from where you left off
- Final results are saved as both .joblib (for loading) and .json (human-readable)

FEATURES TESTED:
- Uses ALL 310 original DE features (62 channels × 5 frequency bands)  
- Tests ~10,000 samples from multiple subjects for better accuracy
- Evaluates feature counts: 5, 10, 15, 20, ..., 100
- Tests 6 different feature selection methods

OUTPUT:
- feature_selection_results/best_features.joblib (for loading in other scripts)
- feature_selection_results/feature_selection_results.json (human-readable)
- feature_selection_results/checkpoint.joblib (for pause/resume, auto-deleted when complete)

Usage: python run_feature_selection.py

Author: AI Assistant  
Date: July 29, 2025
"""

import sys
import numpy as np
from pathlib import Path

# Add the parent directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from data_processing.seed_iv_loader import SeedIVLoader
from data_processing.feature_engineering import AdvancedFeatureEngineer
from feature_selection.feature_selector import AdvancedFeatureSelector
from config import DataConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run feature selection on enhanced EEG features."""
    
    print("🧠 EEG Feature Selection Pipeline")
    print("=" * 50)
    
    try:
        # 1. Load MORE data for better accuracy (increase samples)
        logger.info("📊 Loading EEG data from .mat files with MORE samples...")
        data_config = DataConfig()
        
        loader = SeedIVLoader(data_config)
        
        # MANUALLY override the loader to use MORE subjects for ~10,000 samples
        logger.info("🔄 Overriding loader to use MORE subjects for better accuracy...")
        
        # Load individual subjects and combine manually
        all_features = []
        all_labels = []
        all_subjects = []
        
        # Use more subjects (up to 15 if available)
        subjects_to_use = list(range(1, 16))  # Subjects 1-15
        max_samples_per_subject = 1000  # Increase samples per subject
        
        for subject_id in subjects_to_use:
            try:
                subject_data = loader.load_subject_data(subject_id)
                if 'de_LDS' in subject_data['features']:
                    features = subject_data['features']['de_LDS']
                    labels = subject_data['labels']
                    
                    # Take more samples per subject
                    n_samples = min(max_samples_per_subject, features.shape[0])
                    if n_samples > 0:
                        features_subset = features[:n_samples]
                        labels_subset = labels[:n_samples]
                        
                        all_features.append(features_subset)
                        all_labels.extend(labels_subset)
                        all_subjects.extend([subject_id] * n_samples)
                        
                        logger.info(f"Subject {subject_id}: Added {n_samples} samples")
            except Exception as e:
                logger.warning(f"Failed to load Subject {subject_id}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No data loaded!")
        
        # Combine all data
        X_raw = np.vstack(all_features)
        y = np.array(all_labels)
        subject_ids = np.array(all_subjects)
        
        logger.info(f"📊 TOTAL LOADED: {X_raw.shape[0]} samples with {X_raw.shape[1]} features")
        logger.info(f"🎯 Using ALL {X_raw.shape[1]} DE features (should be 310)")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # 2. USE RAW 310 FEATURES DIRECTLY (not enhanced 72)
        logger.info("🎯 Using RAW 310 DE features directly for feature selection...")
        logger.info("📝 This tests ALL 310 original features, not enhanced subset")
        
        # Use raw features directly (all 310)
        train_size = int(0.8 * len(X_raw))
        X_train = X_raw[:train_size]
        y_train = y[:train_size]
        X_test = X_raw[train_size:]
        y_test = y[train_size:]
        
        logger.info(f"Training data: {X_train.shape} (using ALL {X_train.shape[1]} features)")
        
        # Apply standard preprocessing
        logger.info("🔄 Applying StandardScaler preprocessing...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 3. Run feature selection on ALL 310 RAW features
        logger.info("🚀 Running feature selection on ALL 310 raw DE features...")
        logger.info("📝 NOTE: Using Random Forest (100 estimators) for accuracy evaluation")
        logger.info("📝 Feature selection picks highest cross-validation accuracy method")
        logger.info("🎯 Goal: Find best subset from ALL 310 original DE features")
        logger.info("⏱️  Note: You can press Ctrl+C to stop early and use current best result")
        
        selector = AdvancedFeatureSelector(
            output_dir="feature_selection_results",
            random_state=42
        )
        
        # Test with BETTER k_range as you requested: 5, 10, 15, 20, 25, 30, ... up to 100
        k_range = list(range(5, 101, 5))  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        logger.info(f"Testing feature counts: {k_range}")
        logger.info("💡 TIP: Press Ctrl+C anytime to stop and use current best result!")
        
        # Test different methods and feature counts on the RAW 310 features
        try:
            selection_results = selector.select_best_features(
                X_train_scaled,  # Use the raw 310 features (scaled)
                y_train,
                methods=['random_forest_importance', 'mutual_info', 'f_classif', 'rfe_rf', 'lasso', 'extra_trees'],
                k_range=k_range,  # Test 5, 10, 15, ... 100
                cv_folds=5,
                resume_from_checkpoint=True  # Enable pause/resume functionality
            )
        except KeyboardInterrupt:
            logger.info("🛑 INTERRUPTED BY USER!")
            logger.info("Using best result found so far...")
            if hasattr(selector, 'best_method') and selector.best_method:
                # Use current best result
                best_features = selector.selection_methods[selector.best_method](X_train_scaled, y_train, selector.best_k)
                selection_results = {
                    'best_method': selector.best_method,
                    'best_k': selector.best_k,
                    'best_score': selector.best_score,
                    'selected_features': best_features,
                    'all_results': [],
                    'feature_names': [f'feature_{i}' for i in best_features]
                }
                logger.info(f"🏆 INTERRUPTED RESULT: {selector.best_method} with {selector.best_k} features → {selector.best_score:.4f}")
            else:
                logger.error("No results available yet. Please run longer next time.")
                return None
        
        # 4. Display results
        print("\n🏆 FEATURE SELECTION RESULTS")
        print("=" * 50)
        print(f"Best Method: {selection_results['best_method']}")
        print(f"Optimal Features: {selection_results['best_k']}")
        print(f"Cross-Validation Score: {selection_results['best_score']:.4f}")
        print(f"Improvement Potential: {(selection_results['best_score'] - 0.912) * 100:.2f}% over current 91.2%")
        
        # 5. Show selected feature indices
        selected_features = selection_results['selected_features']
        print(f"\nSelected Feature Indices: {selected_features}")
        print(f"Feature reduction: {X_train_scaled.shape[1]} → {len(selected_features)} features")
        print(f"Percentage reduction: {(1 - len(selected_features)/X_train_scaled.shape[1])*100:.1f}%")
        
        # 6. Create visualization
        logger.info("📊 Creating visualization...")
        selector.visualize_results(selection_results['all_results'])
        
        # 6.5. Validate selected features against full raw features
        logger.info("🧪 Validating selected features vs full 310 raw features...")
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        
        # Test with all 310 raw features (baseline)
        rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        cv_full = cross_val_score(rf_full, X_train_scaled, y_train, 
                                 cv=StratifiedKFold(5, shuffle=True, random_state=42), 
                                 scoring='accuracy')
        
        # Test with selected features
        X_selected = X_train_scaled[:, selected_features]
        rf_selected = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        cv_selected = cross_val_score(rf_selected, X_selected, y_train,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                     scoring='accuracy')
        
        print(f"\n🔬 VALIDATION RESULTS:")
        print(f"Full Raw Features ({X_train_scaled.shape[1]}): {cv_full.mean():.4f} ± {cv_full.std():.4f}")
        print(f"Selected Features ({len(selected_features)}): {cv_selected.mean():.4f} ± {cv_selected.std():.4f}")
        print(f"Performance Change: {(cv_selected.mean() - cv_full.mean())*100:+.2f}%")
        
        if cv_selected.mean() > cv_full.mean():
            print("✅ Feature selection IMPROVED performance!")
        else:
            print("⚠️  Feature selection reduced performance (but faster training)")
        
        # 7. Save for next stage
        logger.info("💾 Saving results for model training...")
        
        # The selector has already saved the selected features with joblib
        # This makes them ready for the next prompt for model training
        
        print("\n✅ Feature selection completed successfully!")
        print("📁 Results saved in: feature_selection_results/")
        print("🚀 Ready for next step: Use selected features in model training")
        
        return selection_results
        
    except Exception as e:
        logger.error(f"❌ Feature selection failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
