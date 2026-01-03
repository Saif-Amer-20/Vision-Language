"""
Visualization utilities for VQA Error Analysis.

Generates publication-quality plots for error analysis results using
matplotlib and seaborn.

Outputs:
- error_types.png: Bar chart of error type breakdown
- question_type_accuracy.png: Accuracy by question type
- confusion_heatmap.png: Common prediction confusions (optional)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings


def plot_error_analysis(
    analysis: Union[Dict[str, Any], 'ErrorAnalysisResult'],
    output_dir: Union[str, Path],
    figsize: tuple = (10, 6),
    dpi: int = 150,
    style: str = 'whitegrid',
) -> Dict[str, Path]:
    """
    Generate all error analysis visualizations.
    
    Args:
        analysis: ErrorAnalysisResult or its dict representation
        output_dir: Directory to save plots
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved images
        style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark')
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        warnings.warn(
            "matplotlib and seaborn required for visualizations. "
            "Install with: pip install matplotlib seaborn"
        )
        return {}
    
    # Convert to dict if ErrorAnalysisResult
    if hasattr(analysis, 'to_dict'):
        analysis = analysis.to_dict()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style(style)
    
    saved_plots = {}
    
    # 1. Error Types Bar Chart
    error_path = _plot_error_types(analysis, output_dir, figsize, dpi)
    if error_path:
        saved_plots['error_types'] = error_path
    
    # 2. Question Type Accuracy
    qtype_path = _plot_question_type_accuracy(analysis, output_dir, figsize, dpi)
    if qtype_path:
        saved_plots['question_type_accuracy'] = qtype_path
    
    # 3. Answer Length Accuracy
    length_path = _plot_answer_length_accuracy(analysis, output_dir, figsize, dpi)
    if length_path:
        saved_plots['answer_length_accuracy'] = length_path
    
    # Reset style
    plt.style.use('default')
    
    return saved_plots


def _plot_error_types(
    analysis: Dict[str, Any],
    output_dir: Path,
    figsize: tuple,
    dpi: int,
) -> Optional[Path]:
    """Generate error types bar chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    error_data = analysis.get('error_types', {})
    counts = error_data.get('counts', {})
    
    if not counts:
        return None
    
    # Prepare data
    labels = list(counts.keys())
    values = list(counts.values())
    
    # Sort by value
    sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = sns.color_palette("husl", len(labels))
    
    # Create bars
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )
    
    # Labels and title
    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Type Distribution', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'error_types.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _plot_question_type_accuracy(
    analysis: Dict[str, Any],
    output_dir: Path,
    figsize: tuple,
    dpi: int,
) -> Optional[Path]:
    """Generate question type accuracy bar chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    qtype_data = analysis.get('question_type_analysis', {})
    counts = qtype_data.get('counts', {})
    accuracy = qtype_data.get('accuracy', {})
    vqa_accuracy = qtype_data.get('vqa_accuracy', {})
    
    if not counts:
        return None
    
    # Sort by count
    sorted_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in sorted_types]
    
    # Get accuracy values
    acc_values = [accuracy.get(t, 0) * 100 for t in labels]
    vqa_values = [vqa_accuracy.get(t, 0) * 100 for t in labels]
    sample_counts = [counts.get(t, 0) for t in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bar positions
    x = np.arange(len(labels))
    width = 0.35
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, acc_values, width, label='Exact Accuracy', 
                   color=sns.color_palette("Blues_d")[2], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, vqa_values, width, label='VQA Accuracy',
                   color=sns.color_palette("Oranges_d")[2], edgecolor='black', linewidth=0.5)
    
    # Add sample counts as text below x-axis
    for i, count in enumerate(sample_counts):
        ax.annotate(
            f'n={count}',
            xy=(i, 0),
            xytext=(0, -25),
            textcoords="offset points",
            ha='center', va='top',
            fontsize=8, color='gray',
        )
    
    # Labels and title
    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Question Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(loc='upper right')
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'question_type_accuracy.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _plot_answer_length_accuracy(
    analysis: Dict[str, Any],
    output_dir: Path,
    figsize: tuple,
    dpi: int,
) -> Optional[Path]:
    """Generate answer length accuracy bar chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    length_data = analysis.get('answer_length_analysis', {})
    counts = length_data.get('counts', {})
    accuracy = length_data.get('accuracy', {})
    
    if not counts:
        return None
    
    # Order by length
    order = ['1_word', '2_words', '3+_words']
    labels = [l for l in order if l in counts]
    
    # Get values
    acc_values = [accuracy.get(l, 0) * 100 for l in labels]
    sample_counts = [counts.get(l, 0) for l in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color based on accuracy
    colors = sns.color_palette("RdYlGn", len(labels))
    
    # Create bars
    bars = ax.bar(labels, acc_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val, count in zip(bars, acc_values, sample_counts):
        height = bar.get_height()
        ax.annotate(
            f'{val:.1f}%\n(n={count})',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10,
        )
    
    # Labels and title
    ax.set_xlabel('Answer Length', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Answer Length', fontsize=14, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, max(acc_values) * 1.2 if acc_values else 100)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'answer_length_accuracy.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_confusion_heatmap(
    confusions: list,
    output_dir: Union[str, Path],
    top_k: int = 10,
    figsize: tuple = (10, 8),
    dpi: int = 150,
) -> Optional[Path]:
    """
    Generate confusion heatmap for common prediction errors.
    
    Args:
        confusions: List of (prediction, ground_truth, count) tuples
        output_dir: Directory to save plot
        top_k: Number of top confusions to include
        figsize: Figure size
        dpi: Resolution
        
    Returns:
        Path to saved plot or None
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        return None
    
    if not confusions:
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top confusions
    top_confusions = confusions[:top_k]
    
    # Extract unique predictions and ground truths
    preds = list(set(c[0] for c in top_confusions))
    gts = list(set(c[1] for c in top_confusions))
    
    # Create confusion matrix
    matrix = np.zeros((len(preds), len(gts)))
    for pred, gt, count in top_confusions:
        i = preds.index(pred)
        j = gts.index(gt)
        matrix[i, j] = count
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        matrix,
        xticklabels=gts,
        yticklabels=preds,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Count'},
    )
    
    # Labels
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title('Common Prediction Confusions', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'confusion_heatmap.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
