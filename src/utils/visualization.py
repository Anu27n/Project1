"""Visualization utilities for the quantum fraud detection project."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_data_distribution(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot data distribution for fraud detection dataset.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: Names of features
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_features = min(X.shape[1], 12)  # Limit to 12 features
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_features):
        ax = axes[i]
        
        # Separate normal and fraud cases
        normal_data = X[y == 0, i]
        fraud_data = X[y == 1, i]
        
        # Plot histograms
        ax.hist(normal_data, bins=30, alpha=0.7, label='Normal', density=True)
        ax.hist(fraud_data, bins=30, alpha=0.7, label='Fraud', density=True)
        
        feature_name = feature_names[i] if feature_names else f'Feature {i+1}'
        ax.set_title(feature_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions: Normal vs Fraud', fontsize=16)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot correlation matrix of features.
    
    Args:
        X: Feature matrix
        feature_names: Names of features
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    correlation_matrix = np.corrcoef(X.T)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Set labels
    if feature_names:
        labels = feature_names[:X.shape[1]]
    else:
        labels = [f'F{i+1}' for i in range(X.shape[1])]
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
    
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_model_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot comparison of model performances across multiple metrics.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        metrics: List of metrics to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    models = list(results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get metric values for all models
        values = [results[model].get(metric, 0) for model in models]
        
        # Create bar plot
        bars = ax.bar(models, values, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    return fig


def plot_roc_curves(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_data: Dictionary with model names and (y_true, y_pred_proba) tuples
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (y_true, y_pred_proba) in models_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_precision_recall_curves(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models.
    
    Args:
        models_data: Dictionary with model names and (y_true, y_pred_proba) tuples
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (y_true, y_pred_proba) in models_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        ax.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
    
    # Plot baseline
    baseline = np.mean(list(models_data.values())[0][0])
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (AP = {baseline:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_learning_curves(
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: List[int] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot learning curves showing training and validation performance.
    
    Args:
        train_scores: Training scores over epochs/iterations
        val_scores: Validation scores over epochs/iterations
        train_sizes: Training set sizes (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if train_sizes is None:
        train_sizes = list(range(1, len(train_scores) + 1))
    
    ax.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
    ax.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
    
    ax.set_xlabel('Training Set Size' if len(train_sizes) != len(train_scores) else 'Epoch/Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def create_interactive_dashboard(
    results: Dict[str, Dict[str, float]],
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = None
) -> go.Figure:
    """
    Create an interactive dashboard using Plotly.
    
    Args:
        results: Model performance results
        models_data: Model prediction data for ROC curves
        
    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Comparison', 'Metric Distribution', 'ROC Curves', 'Performance Radar'),
        specs=[[{"type": "bar"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "scatterpolar"}]]
    )
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # 1. Model comparison bar chart
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric, showlegend=True),
            row=1, col=1
        )
    
    # 2. Metric distribution box plot
    for metric in metrics:
        values = [results[model].get(metric, 0) for model in models]
        fig.add_trace(
            go.Box(y=values, name=metric, showlegend=False),
            row=1, col=2
        )
    
    # 3. ROC curves (if data provided)
    if models_data:
        from sklearn.metrics import roc_curve
        for model_name, (y_true, y_pred_proba) in models_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} ROC', showlegend=False),
                row=2, col=1
            )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                      line=dict(dash='dash'), showlegend=False),
            row=2, col=1
        )
    
    # 4. Performance radar chart
    for model in models:
        values = [results[model].get(metric, 0) for metric in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model,
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Quantum Fraud Detection - Model Performance Dashboard",
        title_x=0.5
    )
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing Visualization Functions...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.random((1000, 10))
    y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    
    # Test data distribution plot
    fig1 = plot_data_distribution(X, y)
    print("✓ Data distribution plot created")
    
    # Test correlation matrix
    fig2 = plot_correlation_matrix(X)
    print("✓ Correlation matrix plot created")
    
    # Test model comparison
    sample_results = {
        'Random Forest': {'accuracy': 0.92, 'precision': 0.85, 'recall': 0.78, 'f1_score': 0.81},
        'XGBoost': {'accuracy': 0.94, 'precision': 0.87, 'recall': 0.82, 'f1_score': 0.84},
        'Quantum Model': {'accuracy': 0.96, 'precision': 0.91, 'recall': 0.88, 'f1_score': 0.89}
    }
    
    fig3 = plot_model_performance_comparison(sample_results)
    print("✓ Model performance comparison plot created")
    
    # Close figures to save memory
    plt.close('all')
    
    print("Visualization testing completed successfully!")
