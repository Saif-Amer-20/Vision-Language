"""Evaluation and analysis module."""

from .evaluator import VQAEvaluator
from .metrics import VQAMetrics
from .error_analysis import (
    ErrorAnalyzer,
    ErrorAnalysisResult,
    PredictionRecord,
    analyze_predictions,
    analyze_predictions_file,
)
from .visualizations import plot_error_analysis, plot_confusion_heatmap

__all__ = [
    "VQAEvaluator",
    "VQAMetrics",
    "ErrorAnalyzer",
    "ErrorAnalysisResult",
    "PredictionRecord",
    "analyze_predictions",
    "analyze_predictions_file",
    "plot_error_analysis",
    "plot_confusion_heatmap",
]
