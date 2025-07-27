"""
SEED-IV Comprehensive Emotion Recognition System - Main Orchestrator

This module orchestrates the entire 6-stage emotion recognition pipeline,
from traditional machine learning baseline to state-of-the-art deep learning models.

6-Stage Progressive Approach:
1. Traditional Baseline (SVM): 70-75% accuracy
2. Enhanced Features (Random Forest): 75-80% accuracy
3. Advanced ML (XGBoost/LightGBM): 80-85% accuracy
4. Deep Learning Foundation (CNN/LSTM): 85-88% accuracy
5. Advanced Deep Learning (Attention/Transformer): 88-92% accuracy
6. State-of-Art Models (Vision Transformer): 92-96% accuracy

Author: GitHub Copilot
Date: 2024
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_emotion_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from config import ComprehensiveConfig, STAGE_PROGRESSION, print_config_summary
    from data_processing.seed_iv_loader import SeedIVLoader
    from models.stage1_traditional import TraditionalBaseline
    from models.stage2_enhanced import EnhancedFeaturesModel
    # Placeholders for future stages
    # from models.stage3_advanced import AdvancedMLModel
    # from models.stage4_deeplearning import DeepLearningModel
    # from models.stage5_attention import AttentionModel
    # from models.stage6_transformer import TransformerModel
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all required modules are in the correct directories")
    sys.exit(1)


class ComprehensiveEmotionRecognition:
    """
    Main orchestrator for the comprehensive emotion recognition system
    """
    
    def __init__(self, config: Optional[ComprehensiveConfig] = None):
        """
        Initialize the comprehensive emotion recognition system
        
        Parameters:
        -----------
        config : ComprehensiveConfig
            Configuration object for the entire pipeline
        """
        self.config = config or ComprehensiveConfig()
        self.stage_results = {}
        self.experiment_metadata = {
            'start_time': datetime.now().isoformat(),
            'python_version': sys.version,
            'config': self.config.__dict__
        }
        
        # Create output directory
        Path(self.config.data.csv_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.data.csv_output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Comprehensive Emotion Recognition System Initialized")
        logger.info(f"Output directory: {self.config.data.csv_output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
    def validate_data_access(self) -> bool:
        """
        Validate that we can access the SEED-IV dataset
        
        Returns:
        --------
        bool
            True if data is accessible, False otherwise
        """
        logger.info("Validating data access...")
        
        try:
            # Check if data path exists
            data_path = Path(self.config.data.seed_iv_base_path)
            if not data_path.exists():
                logger.error(f"Data path does not exist: {data_path}")
                return False
            
            # Try to load a small sample
            loader = SeedIVLoader(self.config.data)
            available_subjects = loader.scan_dataset()
            
            if not available_subjects:
                logger.error("No subjects found in dataset")
                return False
            
            logger.info(f"Data validation successful. Found {len(available_subjects)} subjects")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def save_checkpoint(self, stage_num: int, model, stage_result: Dict[str, Any]) -> None:
        """
        Save checkpoint for a completed stage
        
        Parameters:
        -----------
        stage_num : int
            Stage number
        model : object
            Trained model object
        stage_result : Dict[str, Any]
            Stage execution results
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"stage_{stage_num}_checkpoint.joblib"
            checkpoint_data = {
                'stage_num': stage_num,
                'model': model,
                'result': stage_result,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__
            }
            joblib.dump(checkpoint_data, checkpoint_file)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for stage {stage_num}: {e}")
    
    def load_checkpoint(self, stage_num: int) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a stage if it exists
        
        Parameters:
        -----------
        stage_num : int
            Stage number to load
            
        Returns:
        --------
        Optional[Dict[str, Any]]
            Checkpoint data if exists, None otherwise
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"stage_{stage_num}_checkpoint.joblib"
            if checkpoint_file.exists():
                checkpoint_data = joblib.load(checkpoint_file)
                logger.info(f"Checkpoint loaded: {checkpoint_file}")
                return checkpoint_data
            return None
        except Exception as e:
            logger.warning(f"Failed to load checkpoint for stage {stage_num}: {e}")
            return None
    
    def has_checkpoint(self, stage_num: int) -> bool:
        """
        Check if checkpoint exists for a stage
        
        Parameters:
        -----------
        stage_num : int
            Stage number to check
            
        Returns:
        --------
        bool
            True if checkpoint exists, False otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"stage_{stage_num}_checkpoint.joblib"
        return checkpoint_file.exists()
    
    def run_stage(self, stage_num: int, force_run: bool = False) -> Dict[str, Any]:
        """
        Run a specific stage of the emotion recognition pipeline
        
        Parameters:
        -----------
        stage_num : int
            Stage number to run (1-6)
        force_run : bool
            Force running even if results already exist
            
        Returns:
        --------
        Dict[str, Any]
            Results dictionary containing metrics and model info
        """
        stage_info = STAGE_PROGRESSION.get(stage_num)
        if not stage_info:
            error_msg = f"Invalid stage number: {stage_num}"
            logger.error(error_msg)
            return {'error': error_msg}
        
        # Check for existing checkpoint unless force_run is True
        if not force_run and self.has_checkpoint(stage_num):
            logger.info(f"Loading existing checkpoint for Stage {stage_num}")
            checkpoint_data = self.load_checkpoint(stage_num)
            if checkpoint_data:
                logger.info(f"Stage {stage_num} resumed from checkpoint")
                return checkpoint_data['result']
        
        logger.info(f"Running Stage {stage_num}: {stage_info['name']}")
        logger.info(f"Target Accuracy: {stage_info['target_accuracy']:.1%}")
        
        start_time = time.time()
        
        try:
            model = None
            if stage_num == 1:
                model = TraditionalBaseline(self.config.stage1)
                result = model.run_complete_pipeline(
                    data_config=self.config.data,
                    save_results=True
                )
                
            elif stage_num == 2:
                model = EnhancedFeaturesModel(self.config.stage2)
                result = model.run_complete_pipeline(
                    data_config=self.config.data,
                    save_results=True
                )
                
            elif stage_num == 3:
                # Placeholder for Stage 3: Advanced ML (XGBoost/LightGBM)
                logger.warning("WARNING: Stage 3 (Advanced ML) not implemented yet")
                result = {
                    'error': 'Stage 3 implementation pending',
                    'placeholder': True,
                    'expected_accuracy': 0.825,
                    'methods': ['XGBoost', 'LightGBM', 'CatBoost']
                }
                
            elif stage_num == 4:
                # Placeholder for Stage 4: Deep Learning Foundation
                logger.warning("WARNING: Stage 4 (Deep Learning) not implemented yet")
                result = {
                    'error': 'Stage 4 implementation pending',
                    'placeholder': True,
                    'expected_accuracy': 0.865,
                    'methods': ['CNN', 'LSTM', 'CNN-LSTM']
                }
                
            elif stage_num == 5:
                # Placeholder for Stage 5: Advanced Deep Learning
                logger.warning("WARNING: Stage 5 (Attention Models) not implemented yet")
                result = {
                    'error': 'Stage 5 implementation pending',
                    'placeholder': True,
                    'expected_accuracy': 0.90,
                    'methods': ['Multi-head Attention', 'Temporal Attention']
                }
                
            elif stage_num == 6:
                # Placeholder for Stage 6: State-of-Art
                logger.warning("WARNING: Stage 6 (Vision Transformer) not implemented yet")
                result = {
                    'error': 'Stage 6 implementation pending',
                    'placeholder': True,
                    'expected_accuracy': 0.94,
                    'methods': ['Vision Transformer', 'EEG Transformer']
                }
            
            else:
                error_msg = f"Stage {stage_num} not implemented"
                logger.error(error_msg)
                result = {'error': error_msg}
            
            execution_time = time.time() - start_time
            
            # Add metadata to result
            result.update({
                'stage_num': stage_num,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'target_accuracy': stage_info['target_accuracy']
            })
            
            # Check if target was achieved (for successfully completed stages)
            if 'error' not in result and 'accuracy' in result:
                target_achieved = result['accuracy'] >= stage_info['target_accuracy']
                result['target_achieved'] = target_achieved
                
                status = "ACHIEVED" if target_achieved else "NOT ACHIEVED"
                logger.info(f"Target accuracy {status}: {result['accuracy']:.1%} vs {stage_info['target_accuracy']:.1%}")
            
            # Save checkpoint for successful stages (only stages 1-2 for now)
            if stage_num <= 2 and 'error' not in result and model is not None:
                self.save_checkpoint(stage_num, model, result)
            
            logger.info(f"Stage {stage_num} completed in {execution_time:.1f} seconds")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Stage {stage_num} failed: {str(e)}"
            logger.error(error_msg)
            logger.exception("Detailed error:")
            
            return {
                'error': error_msg,
                'stage_num': stage_num,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_stages(self, stages: Optional[List[int]] = None, force_run: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Run all stages of the emotion recognition pipeline
        
        Parameters:
        -----------
        stages : List[int], optional
            Specific stages to run. If None, runs all stages 1-6
        force_run : bool
            Force running even if results already exist
            
        Returns:
        --------
        Dict[int, Dict[str, Any]]
            Results for all stages
        """
        if stages is None:
            stages = list(range(1, 7))  # All stages 1-6
        
        logger.info(f"Starting comprehensive pipeline for stages: {stages}")
        
        # Validate data access first
        if not self.validate_data_access():
            logger.error("ERROR: Cannot proceed - data validation failed")
            return {}
        
        total_start_time = time.time()
        
        # Run each stage
        for stage_num in stages:
            logger.info(f"\n{'='*60}")
            stage_result = self.run_stage(stage_num, force_run=force_run)
            self.stage_results[stage_num] = stage_result
            
            # Save intermediate results
            self.save_stage_result(stage_num, stage_result)
        
        total_time = time.time() - total_start_time
        self.experiment_metadata.update({
            'end_time': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'stages_completed': len([s for s in self.stage_results.values() if 'error' not in s]),
            'stages_failed': len([s for s in self.stage_results.values() if 'error' in s])
        })
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        logger.info(f"Pipeline completed in {total_time:.1f} seconds")
        
        return self.stage_results
    
    def clear_checkpoints(self, stages: Optional[List[int]] = None) -> None:
        """
        Clear checkpoints for specified stages
        
        Parameters:
        -----------
        stages : List[int], optional
            Stages to clear checkpoints for. If None, clears all checkpoints
        """
        if stages is None:
            stages = list(range(1, 7))
        
        for stage_num in stages:
            checkpoint_file = self.checkpoint_dir / f"stage_{stage_num}_checkpoint.joblib"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Cleared checkpoint for stage {stage_num}")
    
    def show_checkpoint_status(self) -> Dict[int, bool]:
        """
        Show which stages have checkpoints available
        
        Returns:
        --------
        Dict[int, bool]
            Dictionary mapping stage numbers to checkpoint availability
        """
        status = {}
        for stage_num in range(1, 7):
            status[stage_num] = self.has_checkpoint(stage_num)
        
        logger.info("Checkpoint Status:")
        for stage_num, has_cp in status.items():
            status_str = "[Available]" if has_cp else "[Not found]"
            logger.info(f"  Stage {stage_num}: {status_str}")
        
        return status
    
    def save_stage_result(self, stage_num: int, result: Dict[str, Any]):
        """
        Save individual stage result
        
        Parameters:
        -----------
        stage_num : int
            Stage number
        result : Dict[str, Any]
            Stage result dictionary
        """
        result_file = Path(self.config.data.csv_output_dir) / f"stage_{stage_num}_result.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean result for JSON serialization
        clean_result = {}
        for key, value in result.items():
            try:
                clean_result[key] = convert_numpy(value)
            except (TypeError, ValueError):
                clean_result[key] = str(value)
        
        with open(result_file, 'w') as f:
            json.dump(clean_result, f, indent=2, default=str)
        
        logger.info(f"Stage {stage_num} result saved to: {result_file}")
    
    def _get_total_subjects(self) -> int:
        """
        Get total number of subjects from any completed stage
        
        Returns:
        --------
        int
            Number of subjects
        """
        for stage_result in self.stage_results.values():
            if 'subjects' in stage_result and stage_result['subjects'] is not None:
                if isinstance(stage_result['subjects'], (list, np.ndarray)):
                    return len(np.unique(stage_result['subjects']))
        return 0
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive report of all stages
        """
        logger.info("Generating comprehensive report...")
        
        # Create report text
        report_lines = [
            "="*80,
            "COMPREHENSIVE EMOTION RECOGNITION REPORT",
            "="*80,
            f"Dataset: SEED-IV",
            f"Total subjects: {self._get_total_subjects()}",
            f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "STAGE PROGRESSION SUMMARY:",
            "-"*50
        ]
        
        # Add results for each completed stage
        for stage_num in sorted(self.stage_results.keys()):
            result = self.stage_results[stage_num]
            report_lines.extend([
                f"Stage {stage_num}: {result.get('model_type', 'Unknown')}",
                f"  - Accuracy: {result.get('accuracy', 0)*100:.2f}%",
                f"  - F1-Score: {result.get('f1_score', 0)*100:.2f}%",
                f"  - Processing time: {result.get('processing_time', 0):.1f}s",
                ""
            ])
        
        # Save report
        report_path = Path(self.config.data.csv_output_dir) / "comprehensive_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to: {report_path}")
        
        return report_lines


def main():
    """
    Main function to run the comprehensive emotion recognition system
    """
    print("🧠 SEED-IV Comprehensive Emotion Recognition System")
    print("=" * 60)
    print("Progressive 6-stage approach: 70% → 96% accuracy")
    print()
    
    # Print configuration summary
    print_config_summary()
    print()
    
    # Initialize system
    system = ComprehensiveEmotionRecognition()
    
    # Show checkpoint status
    print("CHECKPOINT STATUS:")
    print("-" * 30)
    checkpoint_status = system.show_checkpoint_status()
    available_checkpoints = [stage for stage, has_cp in checkpoint_status.items() if has_cp]
    
    if available_checkpoints:
        print(f"Found checkpoints for stages: {available_checkpoints}")
        print("  → These stages will be resumed from checkpoints")
        print("  → Use force_run=True to rerun from scratch")
    else:
        print("  → No checkpoints found, running all stages from scratch")
    print()
    
    # Run all stages (currently only 1-2 are implemented)
    print("Starting comprehensive experiment...")
    print("Note: Stages 3-6 are placeholders for next iteration")
    print("Expected completion time: ~15-20 minutes for stages 1-2")
    print()
    
    # Run currently implemented stages
    results = system.run_all_stages(stages=[1, 2], force_run=False)
    
    if results:
        print("\nExperiment completed!")
        print(f"Results saved to: {system.config.data.csv_output_dir}")
        print(f"Checkpoints saved to: {system.checkpoint_dir}")
        print("\nCheck the comprehensive report for detailed analysis")
        print("\nRESUME INFO:")
        print("  -> If interrupted, run again to resume from checkpoints")
        print("  -> To start fresh, delete checkpoint files or use force_run=True")
    else:
        print("\nExperiment failed!")
        print("Check the logs for error details")


if __name__ == "__main__":
    main()
