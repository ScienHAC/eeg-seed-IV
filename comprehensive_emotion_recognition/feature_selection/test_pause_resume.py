#!/usr/bin/env python3
"""
Quick Test for Feature Selection Pause/Resume
=============================================

This script runs a quick test of the feature selection system with 
pause/resume functionality to verify everything works correctly.

It uses a smaller parameter set for faster testing.
"""

import sys
import numpy as np
from pathlib import Path

# Add the parent directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from data_processing.seed_iv_loader import SeedIVLoader
from feature_selection.feature_selector import AdvancedFeatureSelector
from config import DataConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Quick test of feature selection with pause/resume."""
    
    print("🧪 Quick Feature Selection Test")
    print("=" * 40)
    print("Press Ctrl+C to test pause/resume functionality!")
    
    try:
        # Load a smaller dataset for quick testing
        logger.info("📊 Loading test data...")
        data_config = DataConfig()
        loader = SeedIVLoader(data_config)
        
        # Use just 2 subjects with fewer samples for quick testing
        all_features = []
        all_labels = []
        
        for subject_id in [1, 2]:  # Just 2 subjects
            try:
                subject_data = loader.load_subject_data(subject_id)
                if 'de_LDS' in subject_data['features']:
                    features = subject_data['features']['de_LDS']
                    labels = subject_data['labels']
                    
                    # Take only first 200 samples per subject for speed
                    n_samples = min(200, features.shape[0])
                    features_subset = features[:n_samples]
                    labels_subset = labels[:n_samples]
                    
                    all_features.append(features_subset)
                    all_labels.extend(labels_subset)
                    
                    logger.info(f"Subject {subject_id}: Added {n_samples} samples")
            except Exception as e:
                logger.warning(f"Failed to load Subject {subject_id}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No data loaded!")
        
        # Combine data
        X_raw = np.vstack(all_features)
        y = np.array(all_labels)
        
        logger.info(f"📊 Test data: {X_raw.shape[0]} samples with {X_raw.shape[1]} features")
        
        # Scale the features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        # Run feature selection with small parameter set for quick testing
        logger.info("🚀 Starting feature selection test...")
        logger.info("💡 TIP: Press Ctrl+C to test pause/resume!")
        
        selector = AdvancedFeatureSelector(
            output_dir="test_feature_selection",
            random_state=42
        )
        
        # Use smaller parameters for quick testing
        test_methods = ['random_forest_importance', 'mutual_info']  # Only 2 methods
        test_k_range = [5, 10, 15]  # Only 3 feature counts
        
        logger.info(f"Testing methods: {test_methods}")
        logger.info(f"Testing feature counts: {test_k_range}")
        logger.info(f"Total combinations: {len(test_methods) * len(test_k_range)} (should complete quickly)")
        
        try:
            results = selector.select_best_features(
                X_scaled,
                y,
                methods=test_methods,
                k_range=test_k_range,
                cv_folds=3,  # Fewer folds for speed
                resume_from_checkpoint=True
            )
            
            if results:
                logger.info(f"\n🏆 TEST COMPLETED!")
                logger.info(f"Best method: {results['best_method']}")
                logger.info(f"Best k: {results['best_k']}")
                logger.info(f"Best score: {results['best_score']:.4f}")
                logger.info(f"Selected features: {len(results['selected_features'])} features")
                
                # Check if JSON file was created
                json_file = Path("test_feature_selection/feature_selection_results.json")
                if json_file.exists():
                    logger.info(f"✅ JSON results file created: {json_file}")
                else:
                    logger.warning("❌ JSON results file not found")
                    
            else:
                logger.error("No results returned!")
                
        except KeyboardInterrupt:
            logger.info("\n🛑 INTERRUPTED! Testing resume functionality...")
            logger.info("💡 Run this script again to test resume from checkpoint!")
            
            # Check if checkpoint was created
            checkpoint_file = Path("test_feature_selection/checkpoint.joblib")
            if checkpoint_file.exists():
                logger.info(f"✅ Checkpoint saved: {checkpoint_file}")
            else:
                logger.warning("❌ No checkpoint file found")
                
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
