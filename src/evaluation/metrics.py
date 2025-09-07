"""Evaluation metrics and statistical tests for model comparison."""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'specificity': calculate_specificity(y_true, y_pred),
        'false_positive_rate': calculate_false_positive_rate(y_true, y_pred),
        'matthews_corrcoef': calculate_mcc(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (True Negative Rate)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0


def calculate_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate false positive rate."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return 0.0


def calculate_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Matthews Correlation Coefficient."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0.0
    return 0.0


def statistical_significance_test(
    scores_model1: List[float],
    scores_model2: List[float],
    test_type: str = "paired_ttest"
) -> Dict[str, Any]:
    """
    Perform statistical significance test between two models.
    
    Args:
        scores_model1: Performance scores from model 1
        scores_model2: Performance scores from model 2
        test_type: Type of test ('paired_ttest', 'wilcoxon')
        
    Returns:
        Test results dictionary
    """
    if test_type == "paired_ttest":
        statistic, p_value = stats.ttest_rel(scores_model1, scores_model2)
        test_name = "Paired t-test"
    elif test_type == "wilcoxon":
        statistic, p_value = stats.wilcoxon(scores_model1, scores_model2)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(scores_model1) - 1) * np.var(scores_model1) + 
                         (len(scores_model2) - 1) * np.var(scores_model2)) / 
                        (len(scores_model1) + len(scores_model2) - 2))
    cohens_d = (np.mean(scores_model1) - np.mean(scores_model2)) / pooled_std
    
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': interpret_effect_size(cohens_d)
    }


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def cross_validation_comparison(
    model_results: Dict[str, List[float]],
    metric_name: str = "f1_score"
) -> Dict[str, Any]:
    """
    Compare multiple models using cross-validation results.
    
    Args:
        model_results: Dictionary with model names as keys and CV scores as values
        metric_name: Name of the metric being compared
        
    Returns:
        Comparison results
    """
    comparison = {
        'metric': metric_name,
        'model_stats': {},
        'rankings': {},
        'significance_tests': {}
    }
    
    # Calculate statistics for each model
    for model_name, scores in model_results.items():
        comparison['model_stats'][model_name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        }
    
    # Rank models by mean performance
    mean_scores = {name: stats['mean'] for name, stats in comparison['model_stats'].items()}
    sorted_models = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    comparison['rankings'] = {name: rank + 1 for rank, (name, score) in enumerate(sorted_models)}
    
    # Perform pairwise significance tests
    model_names = list(model_results.keys())
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            test_key = f"{model1}_vs_{model2}"
            comparison['significance_tests'][test_key] = statistical_significance_test(
                model_results[model1], model_results[model2]
            )
    
    return comparison


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yticklabels(['Normal', 'Fraud'])
    
    return fig


def plot_model_comparison(
    comparison_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot model comparison results.
    
    Args:
        comparison_results: Results from cross_validation_comparison
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    model_stats = comparison_results['model_stats']
    models = list(model_stats.keys())
    means = [model_stats[model]['mean'] for model in models]
    stds = [model_stats[model]['std'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot with error bars
    bars = ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
    ax1.set_title(f"{comparison_results['metric'].replace('_', ' ').title()} Comparison")
    ax1.set_ylabel('Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    # Box plot
    box_data = [list(model_stats[model].values()) for model in models]
    ax2.boxplot(box_data, labels=models)
    ax2.set_title('Score Distribution')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.results = {}
        self.comparisons = {}
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Evaluate a single model and store results.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Evaluation metrics
        """
        metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        return metrics
    
    def compare_models(
        self,
        cv_results: Dict[str, List[float]],
        metric: str = "f1_score"
    ) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation results.
        
        Args:
            cv_results: Cross-validation results for each model
            metric: Metric to use for comparison
            
        Returns:
            Comparison results
        """
        comparison = cross_validation_comparison(cv_results, metric)
        self.comparisons[metric] = comparison
        return comparison
    
    def generate_report(self, model_name: str) -> str:
        """
        Generate a detailed evaluation report for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted report string
        """
        if model_name not in self.results:
            return f"No results found for model: {model_name}"
        
        result = self.results[model_name]
        metrics = result['metrics']
        
        report = f"\n{'='*50}\n"
        report += f"EVALUATION REPORT: {model_name.upper()}\n"
        report += f"{'='*50}\n\n"
        
        report += "CLASSIFICATION METRICS:\n"
        report += f"  Accuracy:      {metrics['accuracy']:.4f}\n"
        report += f"  Precision:     {metrics['precision']:.4f}\n"
        report += f"  Recall:        {metrics['recall']:.4f}\n"
        report += f"  F1-Score:      {metrics['f1_score']:.4f}\n"
        report += f"  Specificity:   {metrics['specificity']:.4f}\n"
        report += f"  FP Rate:       {metrics['false_positive_rate']:.4f}\n"
        report += f"  MCC:           {metrics['matthews_corrcoef']:.4f}\n"
        
        if 'roc_auc' in metrics:
            report += f"  ROC AUC:       {metrics['roc_auc']:.4f}\n"
        
        report += f"\n{'='*50}\n"
        
        return report


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing Evaluation Metrics...")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    y_pred = y_true.copy()
    # Add some noise
    noise_indices = np.random.choice(len(y_pred), 50, replace=False)
    y_pred[noise_indices] = 1 - y_pred[noise_indices]
    y_pred_proba = np.random.random(len(y_true))
    
    # Test comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
    print("Comprehensive metrics:", metrics)
    
    # Test model evaluator
    evaluator = ModelEvaluator()
    evaluator.evaluate_model("test_model", y_true, y_pred, y_pred_proba)
    report = evaluator.generate_report("test_model")
    print(report)
    
    print("Evaluation testing completed successfully!")
