"""
Main validation runner for comprehensive model testing
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import joblib

# Add the comprehensive_emotion_recognition to path
sys.path.append(str(Path(__file__).parent.parent))

from model_validation.config import ValidationConfig
from model_validation.model_loader import ModelLoader
from model_validation.data_loader import UnseenDataLoader
from model_validation.validation_engine import ValidationEngine
from model_validation.report_generator import ValidationReportGenerator

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config.validation_output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """
    Main validation pipeline
    """
    print("🧠 EEG Model Validation System")
    print("=" * 50)
    
    # Initialize configuration
    config = ValidationConfig()
    logger = setup_logging(config)
    
    logger.info("Starting model validation process...")
    
    try:
        # Initialize components
        model_loader = ModelLoader(config)
        data_loader = UnseenDataLoader(config)
        validation_engine = ValidationEngine(config)
        report_generator = ValidationReportGenerator(config)
        
        # Step 1: Load available models
        logger.info("Step 1: Discovering trained models...")
        models = model_loader.load_all_models()
        
        if not models:
            logger.error("❌ No trained models found!")
            print("\n❌ No trained models found in the specified directory.")
            print(f"   Expected location: {config.model_dir}")
            print("   Make sure you have trained models saved as .joblib files.")
            return
        
        logger.info(f"✅ Found {len(models)} trained models")
        for model_name in models.keys():
            print(f"   📋 {model_name}")
        
        # Step 2: Load unseen test data
        logger.info("Step 2: Loading unseen test data...")
        test_data = data_loader.load_unseen_test_data()
        
        if test_data is None:
            logger.error("❌ Failed to load test data!")
            print("\n❌ Failed to load unseen test data.")
            print(f"   Check data directory: {config.data_dir}")
            return
        
        X_test, y_test = test_data
        logger.info(f"✅ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"   📊 Test data shape: {X_test.shape}")
        
        # Step 3: Validate each model
        logger.info("Step 3: Running validation on all models...")
        print("\n🔍 Validating models on unseen data...")
        
        validation_results = []
        
        for i, (model_name, model_data) in enumerate(models.items(), 1):
            print(f"\n   [{i}/{len(models)}] Testing {model_name}...")
            
            try:
                # Run validation
                result = validation_engine.validate_single_model(
                    model_name, model_data['model'], X_test, y_test
                )
                
                validation_results.append(result)
                
                # Display quick results
                if 'error' not in result:
                    accuracy = result['test_accuracy']
                    print(f"       ✅ Test Accuracy: {accuracy:.1%}")
                    
                    if 'overfitting_status' in result:
                        status = result['overfitting_status']
                        emoji = "✅" if status == "Generalizable" else "⚠️" if status == "Moderate" else "❌"
                        print(f"       {emoji} Generalization: {status}")
                else:
                    print(f"       ❌ Error: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error validating {model_name}: {str(e)}")
                validation_results.append({
                    'model_name': model_name,
                    'error': str(e)
                })
        
        # Step 4: Generate summary statistics
        logger.info("Step 4: Generating summary statistics...")
        summary_stats = validation_engine.generate_summary_statistics(validation_results)
        
        # Display summary
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 30)
        
        if summary_stats and 'model_performance' in summary_stats:
            perf = summary_stats['model_performance']
            print(f"Models Tested: {summary_stats.get('n_models_tested', 0)}")
            print(f"Average Accuracy: {perf.get('mean_accuracy', 0):.1%} ± {perf.get('std_accuracy', 0):.1%}")
            
            best_model = summary_stats.get('best_model')
            if best_model:
                print(f"Best Model: {best_model['name']} ({best_model['accuracy']:.1%})")
            
            overfitting_analysis = summary_stats.get('overfitting_analysis', {})
            overfitting_rate = overfitting_analysis.get('overfitting_rate', 0)
            print(f"Overfitting Rate: {overfitting_rate:.1%}")
        
        # Step 5: Generate comprehensive report
        logger.info("Step 5: Generating validation report...")
        print("\n📝 Generating comprehensive report...")
        
        # Prepare metadata
        metadata = {
            'test_subjects': config.test_subjects,
            'n_samples': X_test.shape[0],
            'feature_shape': X_test.shape,
            'feature_type': 'de_LDS',
            'random_state': config.random_state,
            'models_tested': list(models.keys())
        }
        
        # Generate report
        report_path = report_generator.generate_full_report(
            validation_results, summary_stats, metadata
        )
        
        print(f"✅ Report generated: {Path(report_path).name}")
        
        # Step 6: Summary and recommendations
        print("\n🎯 KEY FINDINGS")
        print("-" * 20)
        
        successful_validations = [r for r in validation_results if 'error' not in r]
        if successful_validations:
            # Best performing model
            best_result = max(successful_validations, key=lambda x: x['test_accuracy'])
            print(f"🏆 Best Model: {best_result['model_name']} ({best_result['test_accuracy']:.1%})")
            
            # Generalization assessment
            generalizable_models = [r for r in successful_validations 
                                  if r.get('overfitting_status') == 'Generalizable']
            
            if generalizable_models:
                print(f"✅ Generalizable Models: {len(generalizable_models)}/{len(successful_validations)}")
            else:
                print("⚠️ Potential overfitting detected in some models")
            
            # Performance range
            accuracies = [r['test_accuracy'] for r in successful_validations]
            print(f"📊 Performance Range: {min(accuracies):.1%} - {max(accuracies):.1%}")
        
        print(f"\n📁 All results saved to: {config.validation_output_dir}")
        print(f"📝 Full report: {Path(report_path).name}")
        
        logger.info("✅ Model validation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        print(f"\n❌ Validation failed: {str(e)}")
        raise

def validate_specific_model(model_path: str, test_subject: int = None):
    """
    Validate a specific model file
    
    Parameters:
    -----------
    model_path : str
        Path to the .joblib model file
    test_subject : int, optional
        Specific subject to use for testing
    """
    config = ValidationConfig()
    if test_subject:
        config.test_subjects = [test_subject]
    
    logger = setup_logging(config)
    
    try:
        # Load specific model
        model = joblib.load(model_path)
        model_name = Path(model_path).stem
        
        # Initialize components
        data_loader = UnseenDataLoader(config)
        validation_engine = ValidationEngine(config)
        
        # Load test data
        test_data = data_loader.load_unseen_test_data()
        if test_data is None:
            print("❌ Failed to load test data")
            return
        
        X_test, y_test = test_data
        print(f"✅ Loaded test data: {X_test.shape[0]} samples")
        
        # Validate model
        result = validation_engine.validate_single_model(
            model_name, model, X_test, y_test
        )
        
        # Display results
        if 'error' not in result:
            print(f"\n🎯 {model_name} Validation Results:")
            print(f"   Test Accuracy: {result['test_accuracy']:.1%}")
            print(f"   F1-Score: {result['test_f1']:.1%}")
            print(f"   Status: {result.get('overfitting_status', 'Unknown')}")
        else:
            print(f"❌ Validation failed: {result['error']}")
            
    except Exception as e:
        logger.error(f"Specific model validation failed: {str(e)}")
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Validate specific model
        model_path = sys.argv[1]
        test_subject = int(sys.argv[2]) if len(sys.argv) > 2 else None
        validate_specific_model(model_path, test_subject)
    else:
        # Run full validation suite
        main()
