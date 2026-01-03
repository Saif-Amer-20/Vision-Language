"""
Comprehensive Error Analysis for VQA Evaluation.

This module provides detailed error analysis for VQA predictions including:
- Prediction categorization: correct, incorrect, close_misses
- Error type analysis: exact_wrong, close_miss, type_mismatch, partial_correct
- Question type analysis: what, yes/no, counting, spatial, reasoning
- Answer length analysis: 1_word, 2_words, 3+_words
- Confusion analysis: common (prediction, ground_truth) pairs
- Top error examples ranked by VQA accuracy

Reference:
- "VQA: Visual Question Answering" (Antol et al., ICCV 2015)
- "Making the V in VQA Matter" (Goyal et al., CVPR 2017)
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
import json


# ============================================================================
# Constants
# ============================================================================

# Question type patterns for inference
QUESTION_TYPE_PATTERNS = {
    'yes/no': [
        r'^is\s', r'^are\s', r'^was\s', r'^were\s',
        r'^does\s', r'^do\s', r'^did\s',
        r'^has\s', r'^have\s', r'^had\s',
        r'^can\s', r'^could\s', r'^will\s', r'^would\s',
        r'^should\s', r'^shall\s', r'^may\s', r'^might\s',
    ],
    'counting': [
        r'how many', r'how much', r'count', r'number of',
        r'what number', r'how old', r'what time',
    ],
    'spatial': [
        r'where', r'which side', r'left', r'right', r'top', r'bottom',
        r'above', r'below', r'behind', r'in front', r'next to',
        r'between', r'position', r'location', r'direction',
    ],
    'color': [
        r'what color', r'what colour', r'which color', r'which colour',
    ],
    'what': [
        r'^what\s', r'^what\'s',
    ],
    'who': [
        r'^who\s', r'^who\'s', r'^whose\s',
    ],
    'why': [
        r'^why\s',
    ],
    'how': [
        r'^how\s',
    ],
    'which': [
        r'^which\s',
    ],
}

# Yes/No answer set
YES_NO_ANSWERS = frozenset({'yes', 'no', 'yeah', 'nope', 'yep'})

# Number patterns for type mismatch detection
NUMBER_PATTERN = re.compile(r'^\d+$|^one$|^two$|^three$|^four$|^five$|^six$|^seven$|^eight$|^nine$|^ten$')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PredictionRecord:
    """Single prediction record with all metadata."""
    question_id: str
    question: str
    prediction: str
    ground_truths: List[str]  # Multiple GT answers
    pred_normalized: str
    gt_normalized: List[str]
    vqa_accuracy: float
    is_correct: bool
    is_close_miss: bool
    error_type: Optional[str]
    question_type: str
    answer_length: str
    image_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question_id': self.question_id,
            'question': self.question,
            'prediction': self.prediction,
            'ground_truths': self.ground_truths,
            'pred_normalized': self.pred_normalized,
            'gt_normalized': self.gt_normalized,
            'vqa_accuracy': self.vqa_accuracy,
            'is_correct': self.is_correct,
            'is_close_miss': self.is_close_miss,
            'error_type': self.error_type,
            'question_type': self.question_type,
            'answer_length': self.answer_length,
            'image_id': self.image_id,
        }


@dataclass
class ErrorAnalysisResult:
    """Complete error analysis results."""
    # Summary statistics
    total_samples: int
    correct_count: int
    incorrect_count: int
    close_miss_count: int
    overall_accuracy: float
    overall_vqa_accuracy: float
    
    # Categorized samples
    correct_samples: List[PredictionRecord]
    incorrect_samples: List[PredictionRecord]
    close_miss_samples: List[PredictionRecord]
    
    # Error type breakdown
    error_types: Dict[str, int]
    error_type_percentages: Dict[str, float]
    
    # Question type analysis
    question_type_counts: Dict[str, int]
    question_type_accuracy: Dict[str, float]
    question_type_vqa_accuracy: Dict[str, float]
    
    # Answer length analysis
    answer_length_counts: Dict[str, int]
    answer_length_accuracy: Dict[str, float]
    
    # Confusion analysis
    common_confusions: List[Tuple[str, str, int]]  # (pred, gt, count)
    top_wrong_predictions: List[Tuple[str, int]]
    top_missed_answers: List[Tuple[str, int]]
    
    # Top error examples
    top_errors: List[PredictionRecord]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'summary': {
                'total_samples': self.total_samples,
                'correct_count': self.correct_count,
                'incorrect_count': self.incorrect_count,
                'close_miss_count': self.close_miss_count,
                'overall_accuracy': self.overall_accuracy,
                'overall_vqa_accuracy': self.overall_vqa_accuracy,
            },
            'error_types': {
                'counts': self.error_types,
                'percentages': self.error_type_percentages,
            },
            'question_type_analysis': {
                'counts': self.question_type_counts,
                'accuracy': self.question_type_accuracy,
                'vqa_accuracy': self.question_type_vqa_accuracy,
            },
            'answer_length_analysis': {
                'counts': self.answer_length_counts,
                'accuracy': self.answer_length_accuracy,
            },
            'confusion_analysis': {
                'common_confusions': [
                    {'prediction': p, 'ground_truth': g, 'count': c}
                    for p, g, c in self.common_confusions
                ],
                'top_wrong_predictions': [
                    {'prediction': p, 'count': c}
                    for p, c in self.top_wrong_predictions
                ],
                'top_missed_answers': [
                    {'answer': a, 'count': c}
                    for a, c in self.top_missed_answers
                ],
            },
            'top_errors': [e.to_dict() for e in self.top_errors],
        }


# ============================================================================
# ErrorAnalyzer Class
# ============================================================================

class ErrorAnalyzer:
    """
    Comprehensive error analyzer for VQA predictions.
    
    Features:
    - Categorizes predictions: correct, incorrect, close_misses
    - Analyzes error types: exact_wrong, close_miss, type_mismatch, partial_correct
    - Analyzes by question type: what, yes/no, counting, spatial, reasoning
    - Analyzes by answer length: 1_word, 2_words, 3+_words
    - Finds common confusions: most frequent (prediction, ground_truth) pairs
    - Identifies top error examples sorted by VQA accuracy
    
    Usage:
        analyzer = ErrorAnalyzer(predictions_data)
        result = analyzer.analyze()
        report = analyzer.generate_report()
    """
    
    # Thresholds
    CLOSE_MISS_THRESHOLD = 0.5  # Word overlap threshold for close miss
    VQA_CORRECT_THRESHOLD = 0.99  # VQA accuracy >= this is "correct"
    
    def __init__(
        self,
        predictions: Optional[List[Dict[str, Any]]] = None,
        predictions_file: Optional[str] = None,
        max_samples: int = 500,
        max_error_examples: int = 50,
    ):
        """
        Initialize error analyzer.
        
        Args:
            predictions: List of prediction records with keys:
                - prediction: str
                - ground_truths or ground_truth: str or List[str]
                - question: str
                - question_id: str or int
                - question_type (optional): str
                - image_id (optional): str or int
            predictions_file: Path to JSON file with predictions
            max_samples: Maximum samples to analyze
            max_error_examples: Maximum error examples to include in report
        """
        self.max_samples = max_samples
        self.max_error_examples = max_error_examples
        
        # Load predictions
        if predictions_file:
            with open(predictions_file) as f:
                data = json.load(f)
                # Handle both list format and dict format
                if isinstance(data, list):
                    predictions = data
                elif isinstance(data, dict):
                    # Try to extract predictions from various formats
                    if 'predictions' in data:
                        predictions = self._extract_from_dict(data)
                    else:
                        predictions = [data]
        
        self.predictions = predictions or []
        self._records: List[PredictionRecord] = []
        self._result: Optional[ErrorAnalysisResult] = None
    
    def _extract_from_dict(self, data: Dict) -> List[Dict]:
        """Extract prediction list from dictionary format."""
        preds = data.get('predictions', [])
        gts = data.get('ground_truths', [])
        questions = data.get('questions', [])
        qids = data.get('question_ids', [])
        qtypes = data.get('question_types', [])
        iids = data.get('image_ids', [])
        
        records = []
        for i, pred in enumerate(preds):
            record = {
                'prediction': pred,
                'ground_truths': gts[i] if i < len(gts) else [],
                'question': questions[i] if i < len(questions) else '',
                'question_id': qids[i] if i < len(qids) else str(i),
                'question_type': qtypes[i] if i < len(qtypes) else None,
                'image_id': iids[i] if i < len(iids) else None,
            }
            records.append(record)
        
        return records
    
    def analyze(self) -> ErrorAnalysisResult:
        """
        Perform comprehensive error analysis.
        
        Returns:
            ErrorAnalysisResult with all analysis data
        """
        # Import here to avoid circular imports
        from src.evaluation.metrics import normalize_answer, vqa_accuracy
        
        # Process all predictions into records
        self._records = []
        
        for item in self.predictions[:self.max_samples]:
            record = self._process_prediction(item, normalize_answer, vqa_accuracy)
            self._records.append(record)
        
        # Categorize samples
        correct_samples = []
        incorrect_samples = []
        close_miss_samples = []
        
        for record in self._records:
            if record.is_correct:
                correct_samples.append(record)
            elif record.is_close_miss:
                close_miss_samples.append(record)
            else:
                incorrect_samples.append(record)
        
        # Error type breakdown
        error_types = self._compute_error_types(incorrect_samples + close_miss_samples)
        
        # Question type analysis
        qt_counts, qt_accuracy, qt_vqa = self._compute_question_type_metrics()
        
        # Answer length analysis
        al_counts, al_accuracy = self._compute_answer_length_metrics()
        
        # Confusion analysis
        confusions = self._compute_confusions(incorrect_samples + close_miss_samples)
        wrong_preds = self._compute_top_wrong_predictions(incorrect_samples)
        missed_ans = self._compute_top_missed_answers(incorrect_samples)
        
        # Top errors (sorted by VQA accuracy, ascending)
        all_errors = incorrect_samples + close_miss_samples
        top_errors = sorted(all_errors, key=lambda x: x.vqa_accuracy)[:self.max_error_examples]
        
        # Compute overall metrics
        total = len(self._records)
        correct_count = len(correct_samples)
        overall_acc = correct_count / total if total > 0 else 0.0
        overall_vqa = sum(r.vqa_accuracy for r in self._records) / total if total > 0 else 0.0
        
        # Compute error type percentages
        total_errors = len(incorrect_samples) + len(close_miss_samples)
        error_type_pct = {}
        for etype, count in error_types.items():
            error_type_pct[etype] = (count / total_errors * 100) if total_errors > 0 else 0.0
        
        self._result = ErrorAnalysisResult(
            total_samples=total,
            correct_count=correct_count,
            incorrect_count=len(incorrect_samples),
            close_miss_count=len(close_miss_samples),
            overall_accuracy=overall_acc,
            overall_vqa_accuracy=overall_vqa,
            correct_samples=correct_samples,
            incorrect_samples=incorrect_samples,
            close_miss_samples=close_miss_samples,
            error_types=error_types,
            error_type_percentages=error_type_pct,
            question_type_counts=qt_counts,
            question_type_accuracy=qt_accuracy,
            question_type_vqa_accuracy=qt_vqa,
            answer_length_counts=al_counts,
            answer_length_accuracy=al_accuracy,
            common_confusions=confusions,
            top_wrong_predictions=wrong_preds,
            top_missed_answers=missed_ans,
            top_errors=top_errors,
        )
        
        return self._result
    
    def _process_prediction(
        self,
        item: Dict[str, Any],
        normalize_fn,
        vqa_accuracy_fn,
    ) -> PredictionRecord:
        """Process a single prediction into a PredictionRecord."""
        # Extract fields
        prediction = str(item.get('prediction', ''))
        
        # Handle ground truths (single or multiple)
        gts = item.get('ground_truths', item.get('ground_truth', []))
        if isinstance(gts, str):
            gts = [gts]
        gts = [str(g) for g in gts]
        
        question = str(item.get('question', ''))
        qid = str(item.get('question_id', ''))
        qtype = item.get('question_type')
        image_id = item.get('image_id')
        if image_id is not None:
            image_id = str(image_id)
        
        # Normalize
        pred_norm = normalize_fn(prediction)
        gt_norm = [normalize_fn(g) for g in gts]
        
        # Compute VQA accuracy
        vqa_acc = vqa_accuracy_fn(prediction, gts)
        
        # Determine correctness
        is_correct = vqa_acc >= self.VQA_CORRECT_THRESHOLD
        
        # Determine close miss
        is_close_miss = not is_correct and self._is_partial_correct(pred_norm, gt_norm)
        
        # Infer question type if not provided
        if not qtype:
            qtype = self._infer_question_type(question)
        
        # Determine error type
        error_type = None
        if not is_correct:
            error_type = self._classify_error_type(pred_norm, gt_norm, qtype)
        
        # Determine answer length
        answer_length = self._get_answer_length(prediction)
        
        return PredictionRecord(
            question_id=qid,
            question=question,
            prediction=prediction,
            ground_truths=gts,
            pred_normalized=pred_norm,
            gt_normalized=gt_norm,
            vqa_accuracy=vqa_acc,
            is_correct=is_correct,
            is_close_miss=is_close_miss,
            error_type=error_type,
            question_type=qtype,
            answer_length=answer_length,
            image_id=image_id,
        )
    
    def _infer_question_type(self, question: str) -> str:
        """
        Infer question type from question text.
        
        Categories:
        - yes/no: Questions expecting yes/no answer
        - counting: How many, how much, numbers
        - spatial: Where, position, location
        - color: What color questions
        - what: General what questions
        - who: Who questions
        - why: Why questions
        - how: How questions (non-counting)
        - which: Which questions
        - other: Unclassified
        """
        q_lower = question.lower().strip()
        
        # Check patterns in priority order
        # Color first (subset of 'what')
        for pattern in QUESTION_TYPE_PATTERNS['color']:
            if re.search(pattern, q_lower):
                return 'color'
        
        # Counting before how (how many vs how)
        for pattern in QUESTION_TYPE_PATTERNS['counting']:
            if re.search(pattern, q_lower):
                return 'counting'
        
        # Yes/No
        for pattern in QUESTION_TYPE_PATTERNS['yes/no']:
            if re.search(pattern, q_lower):
                return 'yes/no'
        
        # Spatial
        for pattern in QUESTION_TYPE_PATTERNS['spatial']:
            if re.search(pattern, q_lower):
                return 'spatial'
        
        # Who
        for pattern in QUESTION_TYPE_PATTERNS['who']:
            if re.search(pattern, q_lower):
                return 'who'
        
        # Why
        for pattern in QUESTION_TYPE_PATTERNS['why']:
            if re.search(pattern, q_lower):
                return 'why'
        
        # Which
        for pattern in QUESTION_TYPE_PATTERNS['which']:
            if re.search(pattern, q_lower):
                return 'which'
        
        # How (after counting to avoid 'how many')
        for pattern in QUESTION_TYPE_PATTERNS['how']:
            if re.search(pattern, q_lower):
                return 'how'
        
        # What (catch-all for what questions)
        for pattern in QUESTION_TYPE_PATTERNS['what']:
            if re.search(pattern, q_lower):
                return 'what'
        
        return 'other'
    
    def _is_type_mismatch(self, pred: str, gts: List[str]) -> bool:
        """
        Check if there's a type mismatch between prediction and ground truths.
        
        Type mismatch examples:
        - Prediction is a number, GT is yes/no
        - Prediction is yes/no, GT is a number
        - Prediction is text, GT is a number
        """
        pred_lower = pred.lower().strip()
        
        # Check if prediction is yes/no
        pred_is_yesno = pred_lower in YES_NO_ANSWERS
        
        # Check if prediction is a number
        pred_is_number = bool(NUMBER_PATTERN.match(pred_lower))
        
        # Check ground truths
        gt_is_yesno = any(g.lower().strip() in YES_NO_ANSWERS for g in gts)
        gt_is_number = any(bool(NUMBER_PATTERN.match(g.lower().strip())) for g in gts)
        
        # Type mismatch cases:
        # 1. Prediction is yes/no but GT is not yes/no
        if pred_is_yesno and not gt_is_yesno:
            return True
        
        # 2. Prediction is number but GT is yes/no (different semantic types)
        if pred_is_number and gt_is_yesno:
            return True
        
        # 3. Prediction is number but GT is neither number nor yes/no (text)
        if pred_is_number and not gt_is_number and not gt_is_yesno:
            return True
        
        # 4. Prediction is not yes/no and not number (text) but GT is yes/no
        if not pred_is_yesno and not pred_is_number and gt_is_yesno:
            return True
        
        # 5. Prediction is text but GT is pure number
        if not pred_is_yesno and not pred_is_number and gt_is_number and not gt_is_yesno:
            return True
        
        return False
    
    def _is_partial_correct(self, pred: str, gts: List[str]) -> bool:
        """
        Check if prediction has significant word overlap with any ground truth.
        
        Uses two measures:
        1. Jaccard similarity: |intersection| / |union| >= 0.5
        2. At least one meaningful word overlaps
        """
        pred_words = set(pred.lower().split())
        
        if not pred_words:
            return False
        
        for gt in gts:
            gt_words = set(gt.lower().split())
            if not gt_words:
                continue
            
            intersection = pred_words & gt_words
            union = pred_words | gt_words
            
            # If any non-trivial word overlaps, consider it partial
            # Filter out very short words that are likely articles/prepositions
            meaningful_overlap = [w for w in intersection if len(w) > 2]
            
            if meaningful_overlap:
                return True
            
            # Also check Jaccard similarity for cases with short words
            if len(union) > 0:
                jaccard = len(intersection) / len(union)
                if jaccard >= self.CLOSE_MISS_THRESHOLD:
                    return True
        
        return False
    
    def _classify_error_type(
        self,
        pred_norm: str,
        gt_norm: List[str],
        question_type: str,
    ) -> str:
        """
        Classify the type of error.
        
        Error types:
        - exact_wrong: No overlap, completely wrong answer
        - close_miss: High word overlap but not matching
        - type_mismatch: Wrong answer type (number vs text vs yes/no)
        - partial_correct: Some word overlap, partial credit
        """
        # Check type mismatch first
        if self._is_type_mismatch(pred_norm, gt_norm):
            return 'type_mismatch'
        
        # Check partial correct
        if self._is_partial_correct(pred_norm, gt_norm):
            return 'partial_correct'
        
        # Check close miss (high overlap but not quite)
        pred_words = set(pred_norm.split())
        for gt in gt_norm:
            gt_words = set(gt.split())
            if pred_words and gt_words:
                overlap = len(pred_words & gt_words) / max(len(pred_words), len(gt_words))
                if overlap >= 0.3:  # Lower threshold for close miss
                    return 'close_miss'
        
        # Default: completely wrong
        return 'exact_wrong'
    
    def _get_answer_length(self, answer: str) -> str:
        """Categorize answer by word count."""
        words = answer.strip().split()
        n_words = len(words)
        
        if n_words <= 1:
            return '1_word'
        elif n_words == 2:
            return '2_words'
        else:
            return '3+_words'
    
    def _compute_error_types(
        self,
        error_samples: List[PredictionRecord],
    ) -> Dict[str, int]:
        """Compute error type breakdown."""
        counts = Counter(
            s.error_type for s in error_samples if s.error_type
        )
        return dict(counts)
    
    def _compute_question_type_metrics(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """Compute metrics by question type."""
        type_samples = defaultdict(list)
        
        for record in self._records:
            type_samples[record.question_type].append(record)
        
        counts = {}
        accuracy = {}
        vqa_accuracy = {}
        
        for qtype, samples in type_samples.items():
            counts[qtype] = len(samples)
            correct = sum(1 for s in samples if s.is_correct)
            accuracy[qtype] = correct / len(samples) if samples else 0.0
            vqa_accuracy[qtype] = sum(s.vqa_accuracy for s in samples) / len(samples) if samples else 0.0
        
        return counts, accuracy, vqa_accuracy
    
    def _compute_answer_length_metrics(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Compute metrics by answer length."""
        length_samples = defaultdict(list)
        
        for record in self._records:
            length_samples[record.answer_length].append(record)
        
        counts = {}
        accuracy = {}
        
        for length, samples in length_samples.items():
            counts[length] = len(samples)
            correct = sum(1 for s in samples if s.is_correct)
            accuracy[length] = correct / len(samples) if samples else 0.0
        
        return counts, accuracy
    
    def _compute_confusions(
        self,
        error_samples: List[PredictionRecord],
        top_k: int = 20,
    ) -> List[Tuple[str, str, int]]:
        """Find most common (prediction, ground_truth) pairs."""
        confusion_pairs = Counter()
        
        for sample in error_samples:
            pred = sample.pred_normalized
            # Use most common GT (first one)
            gt = sample.gt_normalized[0] if sample.gt_normalized else ''
            if pred and gt:
                confusion_pairs[(pred, gt)] += 1
        
        return [(p, g, c) for (p, g), c in confusion_pairs.most_common(top_k)]
    
    def _compute_top_wrong_predictions(
        self,
        error_samples: List[PredictionRecord],
        top_k: int = 20,
    ) -> List[Tuple[str, int]]:
        """Find most common wrong predictions."""
        wrong_preds = Counter(
            s.pred_normalized for s in error_samples if s.pred_normalized
        )
        return list(wrong_preds.most_common(top_k))
    
    def _compute_top_missed_answers(
        self,
        error_samples: List[PredictionRecord],
        top_k: int = 20,
    ) -> List[Tuple[str, int]]:
        """Find most commonly missed ground truth answers."""
        missed = Counter()
        for sample in error_samples:
            if sample.gt_normalized:
                missed[sample.gt_normalized[0]] += 1
        return list(missed.most_common(top_k))
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive markdown error analysis report.
        
        Returns:
            Markdown formatted report string
        """
        if self._result is None:
            self.analyze()
        
        result = self._result
        
        report = []
        report.append("# VQA Error Analysis Report\n")
        report.append("---\n")
        
        # Summary Statistics
        report.append("## 📊 Summary Statistics\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Total Samples | {result.total_samples} |")
        report.append(f"| Correct | {result.correct_count} ({result.overall_accuracy*100:.1f}%) |")
        report.append(f"| Incorrect | {result.incorrect_count} |")
        report.append(f"| Close Misses | {result.close_miss_count} |")
        report.append(f"| Overall Accuracy | {result.overall_accuracy*100:.2f}% |")
        report.append(f"| VQA Accuracy | {result.overall_vqa_accuracy*100:.2f}% |")
        report.append("")
        
        # Error Type Breakdown
        report.append("## 🔍 Error Type Breakdown\n")
        report.append("| Error Type | Count | Percentage |")
        report.append("|------------|-------|------------|")
        for etype in ['exact_wrong', 'close_miss', 'type_mismatch', 'partial_correct']:
            count = result.error_types.get(etype, 0)
            pct = result.error_type_percentages.get(etype, 0.0)
            report.append(f"| {etype} | {count} | {pct:.1f}% |")
        report.append("")
        
        # Question Type Analysis
        report.append("## 📝 Performance by Question Type\n")
        report.append("| Question Type | Count | Accuracy | VQA Accuracy |")
        report.append("|---------------|-------|----------|--------------|")
        
        # Sort by count
        sorted_types = sorted(
            result.question_type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for qtype, count in sorted_types:
            acc = result.question_type_accuracy.get(qtype, 0.0)
            vqa = result.question_type_vqa_accuracy.get(qtype, 0.0)
            report.append(f"| {qtype} | {count} | {acc*100:.1f}% | {vqa*100:.1f}% |")
        report.append("")
        
        # Answer Length Analysis
        report.append("## 📏 Performance by Answer Length\n")
        report.append("| Answer Length | Count | Accuracy |")
        report.append("|---------------|-------|----------|")
        for length in ['1_word', '2_words', '3+_words']:
            count = result.answer_length_counts.get(length, 0)
            acc = result.answer_length_accuracy.get(length, 0.0)
            report.append(f"| {length} | {count} | {acc*100:.1f}% |")
        report.append("")
        
        # Common Confusions
        report.append("## 🔄 Common Confusions\n")
        report.append("| Prediction | Ground Truth | Count |")
        report.append("|------------|--------------|-------|")
        for pred, gt, count in result.common_confusions[:15]:
            # Escape pipe characters in table
            pred_safe = pred.replace('|', '\\|')
            gt_safe = gt.replace('|', '\\|')
            report.append(f"| {pred_safe} | {gt_safe} | {count} |")
        report.append("")
        
        # Top Wrong Predictions
        report.append("## ❌ Most Frequent Wrong Predictions\n")
        report.append("| Prediction | Count |")
        report.append("|------------|-------|")
        for pred, count in result.top_wrong_predictions[:10]:
            pred_safe = pred.replace('|', '\\|')
            report.append(f"| {pred_safe} | {count} |")
        report.append("")
        
        # Top Missed Answers
        report.append("## 🎯 Most Commonly Missed Answers\n")
        report.append("| Answer | Count |")
        report.append("|--------|-------|")
        for ans, count in result.top_missed_answers[:10]:
            ans_safe = ans.replace('|', '\\|')
            report.append(f"| {ans_safe} | {count} |")
        report.append("")
        
        # Top Error Examples
        report.append("## 🔬 Top 10 Error Examples\n")
        report.append("Sorted by VQA accuracy (lowest first):\n")
        
        for i, error in enumerate(result.top_errors[:10], 1):
            report.append(f"### Example {i}")
            report.append(f"- **Question ID**: {error.question_id}")
            report.append(f"- **Question**: {error.question}")
            report.append(f"- **Prediction**: `{error.prediction}`")
            report.append(f"- **Ground Truths**: {', '.join(f'`{g}`' for g in error.ground_truths[:5])}")
            report.append(f"- **VQA Accuracy**: {error.vqa_accuracy*100:.1f}%")
            report.append(f"- **Error Type**: {error.error_type}")
            report.append(f"- **Question Type**: {error.question_type}")
            report.append("")
        
        return '\n'.join(report)
    
    def save_analysis(
        self,
        output_dir: str,
        prefix: str = "error_analysis",
    ) -> Dict[str, Path]:
        """
        Save analysis results to files.
        
        Args:
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self._result is None:
            self.analyze()
        
        saved_files = {}
        
        # Save markdown report
        report_path = output_dir / f"{prefix}.md"
        with open(report_path, 'w') as f:
            f.write(self.generate_report())
        saved_files['report'] = report_path
        
        # Save JSON analysis
        json_path = output_dir / f"{prefix}.json"
        with open(json_path, 'w') as f:
            json.dump(self._result.to_dict(), f, indent=2)
        saved_files['json'] = json_path
        
        # Save top errors as CSV
        csv_path = output_dir / "top_errors.csv"
        self._save_errors_csv(csv_path)
        saved_files['csv'] = csv_path
        
        return saved_files
    
    def _save_errors_csv(self, path: Path) -> None:
        """Save top errors to CSV."""
        import csv
        
        if self._result is None:
            return
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'question_id', 'question', 'prediction', 'ground_truths',
                'vqa_accuracy', 'error_type', 'question_type', 'answer_length'
            ])
            
            for error in self._result.top_errors:
                writer.writerow([
                    error.question_id,
                    error.question,
                    error.prediction,
                    '|'.join(error.ground_truths),
                    f"{error.vqa_accuracy:.4f}",
                    error.error_type,
                    error.question_type,
                    error.answer_length,
                ])


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_predictions(
    predictions: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    max_samples: int = 500,
) -> ErrorAnalysisResult:
    """
    Convenience function to analyze predictions.
    
    Args:
        predictions: List of prediction records
        output_dir: Optional output directory for saving results
        max_samples: Maximum samples to analyze
        
    Returns:
        ErrorAnalysisResult
    """
    analyzer = ErrorAnalyzer(predictions=predictions, max_samples=max_samples)
    result = analyzer.analyze()
    
    if output_dir:
        analyzer.save_analysis(output_dir)
    
    return result


def analyze_predictions_file(
    predictions_file: str,
    output_dir: Optional[str] = None,
    max_samples: int = 500,
) -> ErrorAnalysisResult:
    """
    Convenience function to analyze predictions from file.
    
    Args:
        predictions_file: Path to JSON file with predictions
        output_dir: Optional output directory for saving results
        max_samples: Maximum samples to analyze
        
    Returns:
        ErrorAnalysisResult
    """
    analyzer = ErrorAnalyzer(predictions_file=predictions_file, max_samples=max_samples)
    result = analyzer.analyze()
    
    if output_dir:
        analyzer.save_analysis(output_dir)
    
    return result
