"""
Error Analysis for VQA Evaluation.

Provides detailed error analysis with:
- Error categorization
- Sample extraction
- Analysis report generation
"""

from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path


class ErrorAnalyzer:
    """
    Error analyzer for VQA predictions.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize error analyzer.
        
        Args:
            output_dir: Directory to save analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(
        self,
        predictions: List[str],
        targets: List[str],
        questions: List[str],
        question_ids: List[Any],
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Perform error analysis.
        
        Args:
            predictions: Predicted answers
            targets: Ground truth answers
            questions: Questions
            question_ids: Question IDs
            max_samples: Max error samples to save
            
        Returns:
            Error analysis results
        """
        from src.data.answer_utils import normalized_match, normalize_answer
        
        errors = []
        correct = []
        
        for i, (pred, target, question, qid) in enumerate(
            zip(predictions, targets, questions, question_ids)
        ):
            record = {
                'question_id': qid,
                'question': question,
                'prediction': pred,
                'target': target,
                'pred_normalized': normalize_answer(pred),
                'target_normalized': normalize_answer(target),
            }
            
            if normalized_match(pred, target):
                correct.append(record)
            else:
                errors.append(record)
        
        # Analyze errors
        analysis = {
            'total_samples': len(predictions),
            'total_correct': len(correct),
            'total_errors': len(errors),
            'accuracy': len(correct) / len(predictions) if predictions else 0,
        }
        
        # Error categorization
        analysis['error_categories'] = self._categorize_errors(errors)
        
        # Common error patterns
        analysis['common_wrong_predictions'] = Counter(
            e['pred_normalized'] for e in errors
        ).most_common(20)
        
        analysis['common_missed_answers'] = Counter(
            e['target_normalized'] for e in errors
        ).most_common(20)
        
        # Save error samples
        self._save_error_samples(errors[:max_samples])
        
        # Save analysis report
        self._save_analysis_report(analysis)
        
        return analysis
    
    def _categorize_errors(self, errors: List[Dict]) -> Dict[str, int]:
        """Categorize errors by type."""
        categories = {
            'empty_prediction': 0,
            'wrong_type': 0,  # e.g., predicted number when answer is yes/no
            'partial_match': 0,
            'completely_wrong': 0,
        }
        
        for error in errors:
            pred = error['pred_normalized']
            target = error['target_normalized']
            
            if not pred:
                categories['empty_prediction'] += 1
            elif self._is_partial_match(pred, target):
                categories['partial_match'] += 1
            elif self._is_type_mismatch(pred, target):
                categories['wrong_type'] += 1
            else:
                categories['completely_wrong'] += 1
        
        return categories
    
    def _is_partial_match(self, pred: str, target: str) -> bool:
        """Check for partial word overlap."""
        pred_words = set(pred.split())
        target_words = set(target.split())
        return len(pred_words & target_words) > 0
    
    def _is_type_mismatch(self, pred: str, target: str) -> bool:
        """Check for answer type mismatch."""
        yes_no = {'yes', 'no'}
        numbers = set('0123456789')
        
        pred_is_yesno = pred in yes_no
        target_is_yesno = target in yes_no
        
        pred_is_number = any(c in numbers for c in pred)
        target_is_number = any(c in numbers for c in target)
        
        return (pred_is_yesno != target_is_yesno) or (pred_is_number != target_is_number)
    
    def _save_error_samples(self, errors: List[Dict]) -> None:
        """Save error samples to CSV."""
        from src.utils.io_utils import save_csv
        save_csv(errors, str(self.output_dir / "error_samples.csv"))
    
    def _save_analysis_report(self, analysis: Dict) -> None:
        """Save analysis report."""
        from src.utils.io_utils import save_json
        
        # Convert Counter objects to lists for JSON serialization
        report = analysis.copy()
        if 'common_wrong_predictions' in report:
            report['common_wrong_predictions'] = list(report['common_wrong_predictions'])
        if 'common_missed_answers' in report:
            report['common_missed_answers'] = list(report['common_missed_answers'])
        
        save_json(report, str(self.output_dir / "error_analysis.json"))
        
        # Generate markdown report
        self._generate_markdown_report(analysis)
    
    def _generate_markdown_report(self, analysis: Dict) -> None:
        """Generate markdown analysis report."""
        report = f"""# Error Analysis Report

## Summary

| Metric | Value |
|--------|-------|
| Total Samples | {analysis['total_samples']} |
| Correct | {analysis['total_correct']} |
| Errors | {analysis['total_errors']} |
| Accuracy | {analysis['accuracy']:.4f} |

## Error Categories

| Category | Count |
|----------|-------|
"""
        for cat, count in analysis['error_categories'].items():
            report += f"| {cat} | {count} |\n"
        
        report += """
## Common Wrong Predictions

| Prediction | Count |
|------------|-------|
"""
        for pred, count in analysis.get('common_wrong_predictions', [])[:10]:
            report += f"| {pred} | {count} |\n"
        
        report += """
## Common Missed Answers

| Answer | Count |
|--------|-------|
"""
        for ans, count in analysis.get('common_missed_answers', [])[:10]:
            report += f"| {ans} | {count} |\n"
        
        with open(self.output_dir / "error_analysis.md", 'w') as f:
            f.write(report)
        
        print(f"📄 Error analysis report saved to {self.output_dir / 'error_analysis.md'}")
