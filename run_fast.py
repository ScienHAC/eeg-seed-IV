"""
Quick configuration for faster testing - Stage 1 only
"""

from comprehensive_emotion_recognition.config import ComprehensiveConfig
import logging

def create_fast_config():
    """Create a faster configuration for testing"""
    config = ComprehensiveConfig()
    
    # Speed up Stage 1
    config.stages_to_run = [1]  # Only run Stage 1
    config.stage1.use_grid_search = False  # Already disabled
    config.stage1.svm_C = 1.0  # Use fixed parameters
    config.stage1.svm_kernel = 'linear'  # Linear is faster than RBF
    
    return config

def main():
    """Run with fast configuration"""
    from comprehensive_emotion_recognition.main import ComprehensiveEmotionRecognition
    
    config = create_fast_config()
    system = ComprehensiveEmotionRecognition(config=config)
    
    print("🚀 Running with FAST configuration:")
    print("   - Stage 1 only")
    print("   - Linear SVM (faster)")
    print("   - No grid search")
    print("   - Expected time: 5-10 minutes")
    
    results = system.run_comprehensive_analysis()
    
    print("\n✅ Fast run completed!")
    return results

if __name__ == "__main__":
    main()
