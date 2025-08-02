"""
Generate comprehensive validation reports in Markdown format
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ValidationReportGenerator:
    """
    Generates comprehensive validation reports
    """
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.validation_output_dir)
        
    def generate_full_report(self, validation_results: List[Dict[str, Any]], 
                           summary_stats: Dict[str, Any],
                           metadata: Dict = None) -> str:
        """
        Generate complete validation report in Markdown format
        
        Parameters:
        -----------
        validation_results : List[Dict[str, Any]]
            Results from all validated models
        summary_stats : Dict[str, Any]
            Summary statistics
        metadata : Dict, optional
            Additional metadata
            
        Returns:
        --------
        str
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"model_validation_report_{timestamp}.md"
        
        # Generate report content
        report_content = self._create_report_content(validation_results, summary_stats, metadata)
        
        # Write to file
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Validation report generated: {report_file}")
        return str(report_file)
    
    def _create_report_content(self, results: List[Dict[str, Any]], 
                             summary: Dict[str, Any], metadata: Dict = None) -> str:
        """Create the full report content"""
        
        # Header
        content = [
            "# 🧠 EEG Emotion Recognition Model Validation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Dataset:** SEED-IV EEG Emotion Recognition  ",
            f"**Validation Type:** Unseen Subject Testing  ",
            "",
            "---",
            "",
            "## 📋 Executive Summary",
            "",
        ]
        
        # Executive summary
        if summary and 'model_performance' in summary:
            perf = summary['model_performance']
            content.extend([
                f"**Models Tested:** {summary.get('n_models_tested', 0)}  ",
                f"**Average Test Accuracy:** {perf.get('mean_accuracy', 0):.1%} ± {perf.get('std_accuracy', 0):.1%}  ",
                f"**Best Model:** {summary.get('best_model', {}).get('name', 'N/A')} ({summary.get('best_model', {}).get('accuracy', 0):.1%})  ",
                f"**Overfitting Rate:** {summary.get('overfitting_analysis', {}).get('overfitting_rate', 0):.1%}  ",
                "",
            ])
        
        # Key findings
        content.extend([
            "### 🎯 Key Findings",
            "",
            self._generate_key_findings(results, summary),
            "",
            "---",
            "",
            "## 📊 Model Performance Analysis",
            "",
        ])
        
        # Individual model results
        for i, result in enumerate(results, 1):
            if 'error' not in result:
                content.extend(self._format_model_results(result, i))
        
        # Comparative analysis
        content.extend([
            "---",
            "",
            "## 📈 Comparative Analysis",
            "",
            self._generate_comparative_analysis(results, summary),
            "",
        ])
        
        # Overfitting analysis
        content.extend([
            "## 🎯 Overfitting Assessment",
            "",
            self._generate_overfitting_analysis(results),
            "",
        ])
        
        # Class-wise performance
        content.extend([
            "## 🎭 Per-Class Performance Analysis",
            "",
            self._generate_class_analysis(results),
            "",
        ])
        
        # Recommendations
        content.extend([
            "## 💡 Recommendations & Conclusions",
            "",
            self._generate_recommendations(results, summary),
            "",
        ])
        
        # Technical details
        if metadata:
            content.extend([
                "---",
                "",
                "## 🔧 Technical Details",
                "",
                self._format_technical_details(metadata),
                "",
            ])
        
        # Footer
        content.extend([
            "---",
            "",
            "## 📁 Generated Files",
            "",
            "This validation generated the following files:",
            f"- `{self.output_dir.name}/model_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md` - This report",
            "- Confusion matrix plots for each model",
            "- Per-class performance visualizations",
            "- Training vs Test accuracy comparisons",
            "",
            "---",
            "",
            f"*Report generated by EEG Model Validation System v{getattr(self.config, 'version', '1.0.0')}*"
        ])
        
        return '\n'.join(content)
    
    def _generate_key_findings(self, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
        """Generate key findings section"""
        findings = []
        
        if not results:
            return "❌ **No models successfully validated**"
        
        # Overall performance
        if summary and 'model_performance' in summary:
            perf = summary['model_performance']
            avg_acc = perf.get('mean_accuracy', 0)
            
            if avg_acc > 0.95:
                findings.append("✅ **Excellent Generalization**: Models maintain >95% accuracy on unseen data")
            elif avg_acc > 0.85:
                findings.append("✅ **Good Generalization**: Models show >85% accuracy on unseen data")
            elif avg_acc > 0.70:
                findings.append("⚠️ **Moderate Generalization**: Models show 70-85% accuracy on unseen data")
            else:
                findings.append("❌ **Poor Generalization**: Models show <70% accuracy on unseen data")
        
        # Overfitting analysis
        overfitting_analysis = summary.get('overfitting_analysis', {})
        overfitting_rate = overfitting_analysis.get('overfitting_rate', 0)
        
        if overfitting_rate == 0:
            findings.append("✅ **No Overfitting Detected**: All models generalize well")
        elif overfitting_rate < 0.3:
            findings.append("⚠️ **Minor Overfitting**: Some models show signs of overfitting")
        else:
            findings.append("❌ **Significant Overfitting**: Multiple models show poor generalization")
        
        # Best model performance
        best_model = summary.get('best_model')
        if best_model:
            findings.append(f"🏆 **Best Model**: {best_model['name']} achieved {best_model['accuracy']:.1%} test accuracy")
        
        return '\n'.join(f"- {finding}" for finding in findings)
    
    def _format_model_results(self, result: Dict[str, Any], model_num: int) -> List[str]:
        """Format individual model results"""
        content = [
            f"### {model_num}. {result['model_name']}",
            "",
            f"**Test Performance:**",
            f"- Accuracy: {result['test_accuracy']:.1%}",
            f"- F1-Score: {result['test_f1']:.1%}",
            f"- Precision: {result['test_precision']:.1%}",
            f"- Recall: {result['test_recall']:.1%}",
            "",
        ]
        
        # Training vs Test comparison
        if 'training_accuracy' in result:
            gap = result.get('training_test_gap', 0)
            status = result.get('overfitting_status', 'Unknown')
            
            content.extend([
                f"**Generalization Analysis:**",
                f"- Training Accuracy: {result['training_accuracy']:.1%}",
                f"- Test Accuracy: {result['test_accuracy']:.1%}",
                f"- Performance Gap: {gap:.1%}",
                f"- Status: **{status}**",
                "",
            ])
        
        # Per-class performance
        if 'per_class_metrics' in result:
            content.extend([
                f"**Per-Class Performance:**",
                "",
                "| Emotion | Precision | Recall | F1-Score | Support |",
                "|---------|-----------|--------|----------|---------|",
            ])
            
            emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
            per_class = result['per_class_metrics']
            
            for i, emotion in enumerate(emotions):
                if str(i) in per_class:
                    metrics = per_class[str(i)]
                    content.append(
                        f"| {emotion} | {metrics['precision']:.3f} | "
                        f"{metrics['recall']:.3f} | {metrics['f1-score']:.3f} | "
                        f"{metrics['support']} |"
                    )
            
            content.append("")
        
        return content
    
    def _generate_comparative_analysis(self, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
        """Generate comparative analysis section"""
        if not results or len(results) < 2:
            return "Only one model tested - no comparison available."
        
        analysis = []
        
        # Performance comparison
        accuracies = [r['test_accuracy'] for r in results if 'error' not in r]
        if accuracies:
            best_acc = max(accuracies)
            worst_acc = min(accuracies)
            
            analysis.extend([
                f"**Performance Range:** {worst_acc:.1%} - {best_acc:.1%}",
                f"**Performance Spread:** {(best_acc - worst_acc):.1%}",
                "",
            ])
        
        # Model ranking
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x['test_accuracy'], reverse=True)
            
            analysis.extend([
                "**Model Ranking by Test Accuracy:**",
                "",
            ])
            
            for i, result in enumerate(sorted_results, 1):
                status_emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                analysis.append(f"{i}. {status_emoji} {result['model_name']}: {result['test_accuracy']:.1%}")
            
            analysis.append("")
        
        return '\n'.join(analysis)
    
    def _generate_overfitting_analysis(self, results: List[Dict[str, Any]]) -> str:
        """Generate overfitting analysis section"""
        analysis = []
        
        models_with_gap = [r for r in results if 'training_test_gap' in r]
        
        if not models_with_gap:
            return "❌ **No training data available for overfitting analysis**"
        
        analysis.extend([
            "**Training vs Test Performance:**",
            "",
            "| Model | Training Acc | Test Acc | Gap | Status |",
            "|-------|-------------|----------|-----|---------|",
        ])
        
        for result in models_with_gap:
            gap = result['training_test_gap']
            status = result.get('overfitting_status', 'Unknown')
            
            # Status emoji
            if gap < 0.05:
                emoji = "✅"
            elif gap < 0.1:
                emoji = "⚠️"
            else:
                emoji = "❌"
            
            analysis.append(
                f"| {result['model_name']} | {result['training_accuracy']:.1%} | "
                f"{result['test_accuracy']:.1%} | {gap:.1%} | {emoji} {status} |"
            )
        
        analysis.extend(["", "**Interpretation:**"])
        
        # Interpretation
        high_gap_models = [r for r in models_with_gap if r['training_test_gap'] > 0.1]
        if high_gap_models:
            analysis.append(f"- ❌ **{len(high_gap_models)} model(s) show significant overfitting** (>10% gap)")
        
        moderate_gap_models = [r for r in models_with_gap if 0.05 < r['training_test_gap'] <= 0.1]
        if moderate_gap_models:
            analysis.append(f"- ⚠️ **{len(moderate_gap_models)} model(s) show moderate overfitting** (5-10% gap)")
        
        good_models = [r for r in models_with_gap if r['training_test_gap'] <= 0.05]
        if good_models:
            analysis.append(f"- ✅ **{len(good_models)} model(s) generalize well** (<5% gap)")
        
        return '\n'.join(analysis)
    
    def _generate_class_analysis(self, results: List[Dict[str, Any]]) -> str:
        """Generate per-class analysis section"""
        if not results:
            return "No results available for class analysis."
        
        analysis = ["**Emotion Class Performance Summary:**", ""]
        
        # Aggregate class performance across models
        emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
        class_performances = {emotion: [] for emotion in emotions}
        
        for result in results:
            if 'per_class_metrics' in result:
                per_class = result['per_class_metrics']
                for i, emotion in enumerate(emotions):
                    if str(i) in per_class:
                        f1_score = per_class[str(i)]['f1-score']
                        class_performances[emotion].append(f1_score)
        
        # Create summary table
        analysis.extend([
            "| Emotion | Avg F1 | Min F1 | Max F1 | Models Tested |",
            "|---------|--------|--------|--------|---------------|",
        ])
        
        for emotion in emotions:
            scores = class_performances[emotion]
            if scores:
                avg_f1 = np.mean(scores)
                min_f1 = np.min(scores)
                max_f1 = np.max(scores)
                
                # Performance indicator
                if avg_f1 > 0.95:
                    indicator = "🟢"
                elif avg_f1 > 0.85:
                    indicator = "🟡"
                else:
                    indicator = "🔴"
                
                analysis.append(
                    f"| {indicator} {emotion} | {avg_f1:.3f} | {min_f1:.3f} | "
                    f"{max_f1:.3f} | {len(scores)} |"
                )
        
        analysis.append("")
        
        # Identify problematic classes
        problematic = []
        excellent = []
        
        for emotion in emotions:
            scores = class_performances[emotion]
            if scores:
                avg_f1 = np.mean(scores)
                if avg_f1 < 0.85:
                    problematic.append(f"{emotion} ({avg_f1:.1%})")
                elif avg_f1 > 0.95:
                    excellent.append(f"{emotion} ({avg_f1:.1%})")
        
        if problematic:
            analysis.append(f"⚠️ **Classes needing attention:** {', '.join(problematic)}")
        
        if excellent:
            analysis.append(f"✅ **Excellent performing classes:** {', '.join(excellent)}")
        
        return '\n'.join(analysis)
    
    def _generate_recommendations(self, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        recommendations = []
        
        if not results:
            return "❌ **No results available for recommendations**"
        
        # Based on overfitting analysis
        overfitted_models = [r for r in results if r.get('overfitting_status') == 'Likely Overfitted']
        if overfitted_models:
            recommendations.extend([
                "### 🎯 Overfitting Mitigation",
                "- **Increase regularization** in overfitted models",
                "- **Add more diverse training data** from different subjects/sessions",
                "- **Implement cross-validation** during training",
                "- **Reduce model complexity** or apply dropout",
                "",
            ])
        
        # Based on performance
        if summary and 'model_performance' in summary:
            avg_acc = summary['model_performance'].get('mean_accuracy', 0)
            
            if avg_acc < 0.85:
                recommendations.extend([
                    "### 📈 Performance Improvement",
                    "- **Feature engineering**: Add more discriminative features",
                    "- **Data quality**: Review preprocessing pipeline", 
                    "- **Model selection**: Try different algorithms",
                    "- **Ensemble methods**: Combine multiple models",
                    "",
                ])
        
        # Based on class performance
        class_issues = self._identify_class_issues(results)
        if class_issues:
            recommendations.extend([
                "### 🎭 Class-Specific Issues",
                *[f"- **{emotion}**: {issue}" for emotion, issue in class_issues],
                "",
            ])
        
        # General recommendations
        recommendations.extend([
            "### 🔬 Research Recommendations",
            "- **Validate on larger test set** with more subjects",
            "- **Cross-session validation** to test temporal stability",
            "- **Gender-balanced testing** to ensure fairness",
            "- **Real-time performance testing** for practical deployment",
            "",
            "### ✅ Model Selection Guidance",
        ])
        
        # Best model recommendation
        best_model = summary.get('best_model')
        if best_model:
            recommendations.append(f"- **Recommended model**: {best_model['name']} (Test Accuracy: {best_model['accuracy']:.1%})")
        
        generalizable_models = [r for r in results if r.get('overfitting_status') == 'Generalizable']
        if generalizable_models:
            best_generalizable = max(generalizable_models, key=lambda x: x['test_accuracy'])
            recommendations.append(f"- **Most robust model**: {best_generalizable['model_name']} (Generalizable with {best_generalizable['test_accuracy']:.1%} accuracy)")
        
        return '\n'.join(recommendations)
    
    def _identify_class_issues(self, results: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Identify issues with specific emotion classes"""
        emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
        issues = []
        
        # Aggregate performance across all models
        for i, emotion in enumerate(emotions):
            f1_scores = []
            for result in results:
                if 'per_class_metrics' in result and str(i) in result['per_class_metrics']:
                    f1_scores.append(result['per_class_metrics'][str(i)]['f1-score'])
            
            if f1_scores:
                avg_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                
                if avg_f1 < 0.8:
                    issues.append((emotion, f"Low average F1-score ({avg_f1:.1%})"))
                elif std_f1 > 0.1:
                    issues.append((emotion, f"Inconsistent performance across models (std: {std_f1:.1%})"))
        
        return issues
    
    def _format_technical_details(self, metadata: Dict) -> str:
        """Format technical details section"""
        details = [
            "**Dataset Information:**",
            f"- Test Subjects: {metadata.get('test_subjects', 'Unknown')}",
            f"- Total Samples: {metadata.get('n_samples', 'Unknown')}",
            f"- Feature Dimensions: {metadata.get('feature_shape', 'Unknown')}",
            "",
            "**Validation Configuration:**",
            f"- Feature Type: {metadata.get('feature_type', 'de_LDS')}",
            f"- Random Seed: {metadata.get('random_state', 42)}",
            f"- Cross-validation: {metadata.get('cv_folds', 5)} folds",
            "",
            "**Model Details:**",
        ]
        
        # Add model-specific details if available
        if 'models_tested' in metadata:
            for model_info in metadata['models_tested']:
                details.append(f"- {model_info}")
        
        return '\n'.join(details)
