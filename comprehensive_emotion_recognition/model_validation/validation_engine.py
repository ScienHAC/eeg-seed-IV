"""
Core validation engine for testing model generalizability
"""

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        precision_recall_fscore_support
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ValidationEngine:
    """
    Core engine for validating trained emotion recognition models
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.figures = {}
        
        # Set plotting style
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            try:
                sns.set_palette("husl")
            except:
                pass  # Continue without seaborn styling
        
    def validate_single_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                            model_name: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Validate a single trained model on unseen data
        
        Parameters:
        -----------
        model : sklearn model
            Trained model to validate
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        model_name : str
            Name identifier for the model
        metadata : Dict, optional
            Model metadata (training accuracy, etc.)
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive validation results
        """
        logger.info(f"Validating model: {model_name}")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    logger.warning(f"Could not get prediction probabilities for {model_name}")
            
            # Calculate metrics
            if SKLEARN_AVAILABLE:
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Per-class metrics
                per_class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
            else:
                # Manual calculation if sklearn not available
                accuracy = np.mean(y_test == y_pred)
                cm = self._manual_confusion_matrix(y_test, y_pred)
                per_class_report = self._manual_classification_report(y_test, y_pred)
                precision = recall = f1 = accuracy  # Simplified
            
            # Compile results
            results = {
                'model_name': model_name,
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'per_class_metrics': per_class_report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'true_labels': y_test,
                'prediction_probabilities': y_pred_proba,
                'n_test_samples': len(y_test),
                'metadata': metadata or {}
            }
            
            # Calculate training vs test performance gap
            if metadata and 'accuracy' in metadata:
                training_accuracy = metadata['accuracy']
                accuracy_gap = training_accuracy - accuracy
                results['training_test_gap'] = accuracy_gap
                results['training_accuracy'] = training_accuracy
                
                # Determine if model is overfitted
                if accuracy_gap > 0.1:  # 10% drop threshold
                    results['overfitting_status'] = 'Likely Overfitted'
                    results['overfitting_severity'] = 'High' if accuracy_gap > 0.2 else 'Moderate'
                elif accuracy_gap > 0.05:  # 5% drop threshold
                    results['overfitting_status'] = 'Possibly Overfitted'
                    results['overfitting_severity'] = 'Low'
                else:
                    results['overfitting_status'] = 'Generalizable'
                    results['overfitting_severity'] = 'None'
            
            logger.info(f"  Test Accuracy: {accuracy:.1%}")
            logger.info(f"  Test F1-Score: {f1:.1%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'metadata': metadata or {}
            }
    
    def _manual_confusion_matrix(self, y_true, y_pred):
        """Manual confusion matrix calculation"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for true_class in classes:
            for pred_class in classes:
                cm[true_class, pred_class] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return cm
    
    def _manual_classification_report(self, y_true, y_pred):
        """Manual classification report calculation"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        report = {}
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = np.sum(y_true == cls)
            
            report[str(cls)] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }
        
        return report
    
    def analyze_class_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze per-class performance and identify problematic classes
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Validation results from validate_single_model
            
        Returns:
        --------
        Dict[str, Any]
            Class-wise analysis
        """
        if 'per_class_metrics' not in results:
            return {'error': 'No per-class metrics available'}
        
        per_class = results['per_class_metrics']
        emotion_names = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        
        analysis = {
            'class_performance': {},
            'problematic_classes': [],
            'best_performing_classes': [],
            'class_balance_issues': []
        }
        
        # Analyze each class
        for class_id in [0, 1, 2, 3]:
            class_key = str(class_id)
            if class_key in per_class:
                class_metrics = per_class[class_key]
                emotion_name = emotion_names.get(class_id, f'Class_{class_id}')
                
                analysis['class_performance'][emotion_name] = {
                    'precision': class_metrics['precision'],
                    'recall': class_metrics['recall'],
                    'f1_score': class_metrics['f1-score'],
                    'support': class_metrics['support']
                }
                
                # Identify problematic classes (F1 < 0.8)
                if class_metrics['f1-score'] < 0.8:
                    analysis['problematic_classes'].append({
                        'emotion': emotion_name,
                        'f1_score': class_metrics['f1-score'],
                        'issue': 'Low F1-score'
                    })
                
                # Identify best performing classes (F1 > 0.95)
                if class_metrics['f1-score'] > 0.95:
                    analysis['best_performing_classes'].append({
                        'emotion': emotion_name,
                        'f1_score': class_metrics['f1-score']
                    })
                
                # Check for class imbalance (support much lower/higher than average)
                avg_support = np.mean([per_class[str(i)]['support'] for i in range(4) if str(i) in per_class])
                if class_metrics['support'] < 0.7 * avg_support:
                    analysis['class_balance_issues'].append({
                        'emotion': emotion_name,
                        'support': class_metrics['support'],
                        'issue': 'Underrepresented'
                    })
                elif class_metrics['support'] > 1.3 * avg_support:
                    analysis['class_balance_issues'].append({
                        'emotion': emotion_name,
                        'support': class_metrics['support'],
                        'issue': 'Overrepresented'
                    })
        
        return analysis
    
    def create_visualizations(self, results: Dict[str, Any], save_dir: str) -> Dict[str, str]:
        """
        Create visualization plots for validation results
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Validation results
        save_dir : str
            Directory to save plots
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping plot names to file paths
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        try:
            # 1. Confusion Matrix
            if 'confusion_matrix' in results:
                plt.figure(figsize=self.config.figure_size)
                cm = results['confusion_matrix']
                emotion_names = ['Neutral', 'Sad', 'Fear', 'Happy']
                
                if SEABORN_AVAILABLE:
                    try:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=emotion_names, yticklabels=emotion_names)
                    except:
                        # Fallback without seaborn
                        plt.imshow(cm, interpolation='nearest', cmap='Blues')
                        plt.colorbar()
                else:
                    # Fallback without seaborn
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.colorbar()
                    
                    # Add text annotations manually
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
                    
                    plt.xticks(range(len(emotion_names)), emotion_names)
                    plt.yticks(range(len(emotion_names)), emotion_names)
                    for i in range(len(emotion_names)):
                        for j in range(len(emotion_names)):
                            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
                    
                plt.title(f'Confusion Matrix - {results["model_name"]}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                cm_file = save_path / f"{results['model_name']}_confusion_matrix.png"
                plt.savefig(cm_file, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plot_files['confusion_matrix'] = str(cm_file)
            
            # 2. Per-class Performance
            if 'per_class_metrics' in results:
                per_class = results['per_class_metrics']
                emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
                metrics = ['precision', 'recall', 'f1-score']
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                for i, metric in enumerate(metrics):
                    values = [per_class[str(j)][metric] for j in range(4) if str(j) in per_class]
                    if SEABORN_AVAILABLE:
                        try:
                            colors = sns.color_palette("husl", len(values))
                        except:
                            colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
                    else:
                        colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
                        
                    axes[i].bar(emotions[:len(values)], values, color=colors)
                    axes[i].set_title(f'{metric.capitalize()}')
                    axes[i].set_ylim(0, 1)
                    axes[i].set_ylabel('Score')
                    
                    # Add value labels on bars
                    for j, v in enumerate(values):
                        axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
                plt.suptitle(f'Per-Class Performance - {results["model_name"]}')
                plt.tight_layout()
                
                perf_file = save_path / f"{results['model_name']}_per_class_performance.png"
                plt.savefig(perf_file, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plot_files['per_class_performance'] = str(perf_file)
            
            # 3. Training vs Test Accuracy (if available)
            if 'training_accuracy' in results and 'test_accuracy' in results:
                plt.figure(figsize=(8, 6))
                
                accuracies = [results['training_accuracy'], results['test_accuracy']]
                labels = ['Training', 'Test']
                colors = ['lightblue', 'lightcoral']
                
                bars = plt.bar(labels, accuracies, color=colors)
                plt.title(f'Training vs Test Accuracy - {results["model_name"]}')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1)
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.01, 
                            f'{acc:.1%}', ha='center', va='bottom')
                
                # Add gap annotation
                if 'training_test_gap' in results:
                    gap = results['training_test_gap']
                    plt.annotate(f'Gap: {gap:.1%}', 
                               xy=(0.5, max(accuracies) - gap/2),
                               xytext=(0.5, max(accuracies) + 0.05),
                               ha='center', va='bottom',
                               arrowprops=dict(arrowstyle='<->', color='red'))
                
                acc_file = save_path / f"{results['model_name']}_accuracy_comparison.png"
                plt.savefig(acc_file, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plot_files['accuracy_comparison'] = str(acc_file)
            
            logger.info(f"Created {len(plot_files)} visualization plots in {save_dir}")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
        
        return plot_files
    
    def generate_summary_statistics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics across all validated models
        
        Parameters:
        -----------
        all_results : List[Dict[str, Any]]
            List of validation results from all models
            
        Returns:
        --------
        Dict[str, Any]
            Summary statistics
        """
        if not all_results:
            return {'error': 'No results to summarize'}
        
        summary = {
            'n_models_tested': len(all_results),
            'model_performance': {},
            'overfitting_analysis': {},
            'best_model': None,
            'worst_model': None
        }
        
        # Extract performance metrics
        accuracies = []
        f1_scores = []
        model_names = []
        
        for result in all_results:
            if 'error' not in result:
                accuracies.append(result['test_accuracy'])
                f1_scores.append(result['test_f1'])
                model_names.append(result['model_name'])
        
        if accuracies:
            summary['model_performance'] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores)
            }
            
            # Best and worst models
            best_idx = np.argmax(accuracies)
            worst_idx = np.argmin(accuracies)
            
            summary['best_model'] = {
                'name': model_names[best_idx],
                'accuracy': accuracies[best_idx],
                'f1_score': f1_scores[best_idx]
            }
            
            summary['worst_model'] = {
                'name': model_names[worst_idx],
                'accuracy': accuracies[worst_idx],
                'f1_score': f1_scores[worst_idx]
            }
        
        # Overfitting analysis
        overfitted_models = [r for r in all_results if r.get('overfitting_status') == 'Likely Overfitted']
        generalizable_models = [r for r in all_results if r.get('overfitting_status') == 'Generalizable']
        
        summary['overfitting_analysis'] = {
            'n_overfitted': len(overfitted_models),
            'n_generalizable': len(generalizable_models),
            'overfitting_rate': len(overfitted_models) / len(all_results) if all_results else 0
        }
        
        return summary
