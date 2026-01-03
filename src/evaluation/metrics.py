"""
VQA Metrics Implementation.

Provides standard VQA metrics:
- Exact Match
- Normalized Match
- VQA Accuracy (soft matching)
"""

from typing import List, Dict, Any
from collections import Counter


class VQAMetrics:
    """
    VQA evaluation metrics.
    """
    
    def __init__(self):
        pass
    
    def compute(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> Dict[str, float]:
        """
        Compute all VQA metrics.
        
        Args:
            predictions: Predicted answers
            targets: Ground truth answers
            
        Returns:
            Dictionary of metrics
        """
        from src.data.answer_utils import (
            exact_match,
            normalized_match,
            normalize_answer,
        )
        
        n = len(predictions)
        if n == 0:
            return {
                'exact_match': 0.0,
                'normalized_match': 0.0,
                'vqa_accuracy': 0.0,
                'total_samples': 0,
            }
        
        # Exact match
        exact = sum(exact_match(p, t) for p, t in zip(predictions, targets))
        
        # Normalized match
        normalized = sum(normalized_match(p, t) for p, t in zip(predictions, targets))
        
        # VQA accuracy (with multiple reference handling if available)
        vqa_acc = normalized  # Using normalized as proxy
        
        return {
            'exact_match': exact / n,
            'normalized_match': normalized / n,
            'vqa_accuracy': vqa_acc / n,
            'total_samples': n,
            'correct_exact': exact,
            'correct_normalized': normalized,
        }
    
    def compute_per_type(
        self,
        predictions: List[str],
        targets: List[str],
        question_types: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per question type.
        
        Args:
            predictions: Predicted answers
            targets: Ground truth answers
            question_types: Question type labels
            
        Returns:
            Metrics per question type
        """
        from src.data.answer_utils import normalized_match
        
        # Group by type
        type_results = {}
        for pred, target, qtype in zip(predictions, targets, question_types):
            if qtype not in type_results:
                type_results[qtype] = {'correct': 0, 'total': 0}
            
            type_results[qtype]['total'] += 1
            if normalized_match(pred, target):
                type_results[qtype]['correct'] += 1
        
        # Compute accuracy per type
        metrics_per_type = {}
        for qtype, counts in type_results.items():
            metrics_per_type[qtype] = {
                'accuracy': counts['correct'] / counts['total'],
                'correct': counts['correct'],
                'total': counts['total'],
            }
        
        return metrics_per_type
