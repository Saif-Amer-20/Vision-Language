"""
VQA Metrics Implementation - Official VQAv2 Protocol.

This module implements the official VQAv2 evaluation metrics as described in:
- "VQA: Visual Question Answering" (Antol et al., ICCV 2015)
- "Making the V in VQA Matter" (Goyal et al., CVPR 2017)

VQA Accuracy Formula:
=====================
For each question, 10 human annotators provide answers. The official VQA accuracy
for a predicted answer is computed as:

    accuracy = min(#humans_that_provided_that_answer / 3, 1.0)

This gives:
- 0 humans agree → 0.0 (0%)
- 1 human agrees → 0.33 (33%)
- 2 humans agree → 0.67 (67%)
- 3+ humans agree → 1.0 (100%)

The intuition is that an answer is "correct" if at least 3 out of 10 humans
agree with it, accounting for human disagreement on subjective questions.

Answer Normalization (per official evaluation script):
======================================================
1. Convert to lowercase
2. Convert number words to digits ("one" → "1")
3. Remove punctuation except apostrophes in contractions
4. Remove articles (a, an, the)
5. Collapse multiple spaces to single space
6. Strip leading/trailing whitespace

Reference:
- https://visualqa.org/evaluation.html
- https://github.com/GT-Vision-Lab/VQA (official evaluation code)
"""

import re
import string
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json


# ============================================================================
# Constants
# ============================================================================

# Articles to remove during normalization
ARTICLES = frozenset({'a', 'an', 'the'})

# Period used in numbers (for decimal handling)
PERIOD_STRIP = re.compile(r'(?!<=\d)(\.)(?!\d)')

# Comma used in numbers
COMMA_STRIP = re.compile(r'(\d)(\,)(\d)')

# Punctuation to remove (keeping apostrophes for contractions)
PUNCTUATION = set(string.punctuation) - {"'"}
PUNCT_REGEX = re.compile(r"[{}]".format(re.escape(''.join(PUNCTUATION))))

# Number word to digit mapping
NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
    'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100',
    'thousand': '1000', 'million': '1000000',
}

# Common contractions for expansion
CONTRACTIONS = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


# ============================================================================
# Answer Normalization Functions
# ============================================================================

def process_digit_article(text: str) -> str:
    """
    Process articles and convert number words to digits.
    
    This follows the official VQA evaluation script logic.
    """
    words = []
    for word in text.split():
        # Check if it's a number word
        if word in NUMBER_WORDS:
            words.append(NUMBER_WORDS[word])
        # Skip articles
        elif word not in ARTICLES:
            words.append(word)
    return ' '.join(words)


def process_punctuation(text: str) -> str:
    """
    Remove punctuation except apostrophes in contractions.
    
    Handles:
    - Periods in decimal numbers (keeps them)
    - Commas in large numbers (keeps them)
    - All other punctuation (removes)
    """
    # Handle periods not in numbers
    text = PERIOD_STRIP.sub('', text)
    
    # Handle commas in numbers (remove the comma but keep digits together)
    text = COMMA_STRIP.sub(r'\1\3', text)
    
    # Remove other punctuation
    text = PUNCT_REGEX.sub(' ', text)
    
    return text


def expand_contractions(text: str) -> str:
    """
    Expand common contractions.
    
    Args:
        text: Input text
        
    Returns:
        Text with contractions expanded
    """
    words = text.split()
    result = []
    for word in words:
        if word.lower() in CONTRACTIONS:
            result.append(CONTRACTIONS[word.lower()])
        else:
            result.append(word)
    return ' '.join(result)


def normalize_answer(answer: str, expand_contractions_flag: bool = True) -> str:
    """
    Normalize answer following official VQAv2 evaluation protocol.
    
    Processing steps:
    1. Convert to lowercase
    2. Expand contractions (optional)
    3. Remove punctuation (except in numbers)
    4. Convert number words to digits
    5. Remove articles (a, an, the)
    6. Collapse multiple spaces
    7. Strip whitespace
    
    Args:
        answer: Raw answer string
        expand_contractions_flag: Whether to expand contractions
        
    Returns:
        Normalized answer string
    
    Examples:
        >>> normalize_answer("The cat")
        'cat'
        >>> normalize_answer("It's a dog!")
        'it is dog'
        >>> normalize_answer("Three apples")
        '3 apples'
        >>> normalize_answer("  Hello,   World!  ")
        'hello world'
    """
    if not answer:
        return ''
    
    # Step 1: Lowercase
    answer = answer.lower()
    
    # Step 2: Expand contractions
    if expand_contractions_flag:
        answer = expand_contractions(answer)
    
    # Step 3: Remove punctuation
    answer = process_punctuation(answer)
    
    # Step 4 & 5: Convert number words and remove articles
    answer = process_digit_article(answer)
    
    # Step 6 & 7: Collapse spaces and strip
    answer = ' '.join(answer.split())
    
    return answer


# ============================================================================
# Metric Functions
# ============================================================================

def exact_match(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """
    Compute exact match score.
    
    Checks if prediction exactly matches any ground truth (case-insensitive,
    whitespace-normalized).
    
    Args:
        prediction: Model's predicted answer
        ground_truths: Single answer or list of valid answers
        
    Returns:
        1.0 if exact match found, 0.0 otherwise
    
    Examples:
        >>> exact_match("cat", "Cat")
        1.0
        >>> exact_match("cat", "dog")
        0.0
        >>> exact_match("cat", ["cat", "kitty", "feline"])
        1.0
    """
    # Normalize input to list
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    pred_normalized = prediction.strip().lower()
    
    for gt in ground_truths:
        if pred_normalized == gt.strip().lower():
            return 1.0
    
    return 0.0


def normalized_match(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """
    Compute normalized match score.
    
    Like exact match but applies full answer normalization before comparison.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: Single answer or list of valid answers
        
    Returns:
        1.0 if normalized match found, 0.0 otherwise
    
    Examples:
        >>> normalized_match("The cat", "cat")
        1.0
        >>> normalized_match("3 dogs", "three dogs")
        1.0
        >>> normalized_match("it's blue", "it is blue")
        1.0
    """
    # Normalize input to list
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    pred_normalized = normalize_answer(prediction)
    
    for gt in ground_truths:
        if pred_normalized == normalize_answer(gt):
            return 1.0
    
    return 0.0


def vqa_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute official VQA accuracy score.
    
    This implements the official VQAv2 evaluation formula:
        accuracy = min(count / 3.0, 1.0)
    
    Where 'count' is the number of ground truth annotators who provided
    the same answer as the prediction (after normalization).
    
    For VQAv2, each question has 10 human annotators. The formula gives
    partial credit when multiple humans agree with the prediction:
    - 0 agree → 0.00 (0%)
    - 1 agrees → 0.33 (33%)
    - 2 agree → 0.67 (67%)
    - 3+ agree → 1.00 (100%)
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of ground truth answers (typically 10 per question)
        
    Returns:
        VQA accuracy score between 0.0 and 1.0
    
    Examples:
        >>> vqa_accuracy("cat", ["cat", "cat", "cat", "dog", "dog"])
        1.0  # 3 annotators said "cat", min(3/3, 1) = 1.0
        
        >>> vqa_accuracy("cat", ["cat", "cat", "dog", "dog", "dog"])
        0.6666...  # 2 annotators said "cat", min(2/3, 1) = 0.67
        
        >>> vqa_accuracy("cat", ["cat", "dog", "dog", "dog", "dog"])
        0.3333...  # 1 annotator said "cat", min(1/3, 1) = 0.33
        
        >>> vqa_accuracy("bird", ["cat", "dog", "cat", "dog", "cat"])
        0.0  # 0 annotators said "bird", min(0/3, 1) = 0.0
    """
    if not ground_truths:
        return 0.0
    
    # Normalize prediction
    pred_normalized = normalize_answer(prediction)
    
    # Count how many ground truths match the prediction
    count = 0
    for gt in ground_truths:
        if normalize_answer(gt) == pred_normalized:
            count += 1
    
    # Official VQA formula: min(count / 3, 1)
    return min(count / 3.0, 1.0)


def soft_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute soft accuracy using word overlap.
    
    Useful as an auxiliary metric when VQA accuracy is too strict.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of ground truth answers
        
    Returns:
        Maximum word overlap score with any ground truth
    """
    if not ground_truths:
        return 0.0
    
    pred_words = set(normalize_answer(prediction).split())
    
    if not pred_words:
        return 0.0
    
    max_score = 0.0
    for gt in ground_truths:
        gt_words = set(normalize_answer(gt).split())
        if not gt_words:
            continue
        
        # Compute F1-style overlap
        overlap = len(pred_words & gt_words)
        precision = overlap / len(pred_words) if pred_words else 0
        recall = overlap / len(gt_words) if gt_words else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            max_score = max(max_score, f1)
    
    return max_score


# ============================================================================
# VQA Metrics Class
# ============================================================================

@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    
    # Core metrics (percentages)
    exact_match: float = 0.0
    normalized_match: float = 0.0
    vqa_accuracy: float = 0.0
    soft_accuracy: float = 0.0
    
    # Counts
    total_samples: int = 0
    correct_exact: int = 0
    correct_normalized: int = 0
    
    # Per-type breakdown
    per_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Answer distribution
    answer_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'exact_match': self.exact_match,
            'normalized_match': self.normalized_match,
            'vqa_accuracy': self.vqa_accuracy,
            'soft_accuracy': self.soft_accuracy,
            'total_samples': self.total_samples,
            'correct_exact': self.correct_exact,
            'correct_normalized': self.correct_normalized,
            'per_type': self.per_type,
            'answer_distribution': self.answer_distribution,
        }
    
    def __repr__(self) -> str:
        return (
            f"MetricsResult(\n"
            f"  exact_match={self.exact_match:.2f}%,\n"
            f"  normalized_match={self.normalized_match:.2f}%,\n"
            f"  vqa_accuracy={self.vqa_accuracy:.2f}%,\n"
            f"  soft_accuracy={self.soft_accuracy:.2f}%,\n"
            f"  total_samples={self.total_samples}\n"
            f")"
        )


class VQAMetrics:
    """
    Complete VQA evaluation metrics following official VQAv2 protocol.
    
    Computes:
    - Exact Match: Strict string equality (case-insensitive)
    - Normalized Match: String equality after normalization
    - VQA Accuracy: Official formula min(count/3, 1)
    - Soft Accuracy: Word overlap-based score
    
    Also provides:
    - Per-question-type breakdown
    - Answer distribution analysis
    - Error analysis utilities
    
    Usage:
        metrics = VQAMetrics()
        
        # Single sample
        score = metrics.vqa_accuracy("cat", ["cat", "cat", "dog", "cat"])
        
        # Batch evaluation
        results = metrics.compute_metrics(
            predictions=["cat", "dog"],
            ground_truths=[["cat", "cat", "cat"], ["dog", "cat", "cat"]],
            question_types=["what", "what"]
        )
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize VQA metrics calculator.
        
        Args:
            verbose: Print detailed progress during computation
        """
        self.verbose = verbose
    
    # Expose module-level functions as methods
    normalize_answer = staticmethod(normalize_answer)
    exact_match = staticmethod(exact_match)
    normalized_match = staticmethod(normalized_match)
    vqa_accuracy = staticmethod(vqa_accuracy)
    soft_accuracy = staticmethod(soft_accuracy)
    
    def compute_metrics(
        self,
        predictions: List[str],
        ground_truths: List[Union[str, List[str]]],
        question_types: Optional[List[str]] = None,
        compute_distribution: bool = True,
    ) -> MetricsResult:
        """
        Compute all VQA metrics for a batch of predictions.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers (each can be str or list)
            question_types: Optional list of question types for breakdown
            compute_distribution: Whether to compute answer distribution
            
        Returns:
            MetricsResult with all computed metrics
        """
        n = len(predictions)
        
        if n == 0:
            return MetricsResult()
        
        if len(ground_truths) != n:
            raise ValueError(
                f"Length mismatch: {n} predictions vs {len(ground_truths)} ground truths"
            )
        
        # Normalize ground truths to lists
        normalized_gts = []
        for gt in ground_truths:
            if isinstance(gt, str):
                normalized_gts.append([gt])
            else:
                normalized_gts.append(list(gt))
        
        # Compute individual scores
        exact_scores = []
        normalized_scores = []
        vqa_scores = []
        soft_scores = []
        
        for pred, gts in zip(predictions, normalized_gts):
            exact_scores.append(exact_match(pred, gts))
            normalized_scores.append(normalized_match(pred, gts))
            vqa_scores.append(vqa_accuracy(pred, gts))
            soft_scores.append(soft_accuracy(pred, gts))
        
        # Aggregate metrics
        result = MetricsResult(
            exact_match=100.0 * sum(exact_scores) / n,
            normalized_match=100.0 * sum(normalized_scores) / n,
            vqa_accuracy=100.0 * sum(vqa_scores) / n,
            soft_accuracy=100.0 * sum(soft_scores) / n,
            total_samples=n,
            correct_exact=int(sum(exact_scores)),
            correct_normalized=int(sum(normalized_scores)),
        )
        
        # Per-type breakdown
        if question_types is not None:
            result.per_type = self._compute_per_type(
                predictions, normalized_gts, question_types, vqa_scores
            )
        
        # Answer distribution
        if compute_distribution:
            result.answer_distribution = self._compute_distribution(predictions)
        
        return result
    
    def _compute_per_type(
        self,
        predictions: List[str],
        ground_truths: List[List[str]],
        question_types: List[str],
        vqa_scores: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics breakdown by question type."""
        type_data = defaultdict(lambda: {
            'scores': [],
            'exact': 0,
            'normalized': 0,
            'total': 0
        })
        
        for pred, gts, qtype, vqa_score in zip(
            predictions, ground_truths, question_types, vqa_scores
        ):
            type_data[qtype]['scores'].append(vqa_score)
            type_data[qtype]['total'] += 1
            type_data[qtype]['exact'] += exact_match(pred, gts)
            type_data[qtype]['normalized'] += normalized_match(pred, gts)
        
        result = {}
        for qtype, data in type_data.items():
            n = data['total']
            result[qtype] = {
                'vqa_accuracy': 100.0 * sum(data['scores']) / n,
                'exact_match': 100.0 * data['exact'] / n,
                'normalized_match': 100.0 * data['normalized'] / n,
                'total': n,
            }
        
        return result
    
    def _compute_distribution(
        self,
        predictions: List[str],
        top_k: int = 20,
    ) -> Dict[str, int]:
        """Compute answer distribution."""
        counter = Counter(normalize_answer(p) for p in predictions)
        return dict(counter.most_common(top_k))
    
    # Backwards compatibility alias
    def compute(
        self,
        predictions: List[str],
        targets: Union[List[str], List[List[str]]],
    ) -> Dict[str, float]:
        """
        Compute metrics (backwards-compatible interface).
        
        Args:
            predictions: Predicted answers
            targets: Ground truth answers
            
        Returns:
            Dictionary of metrics
        """
        result = self.compute_metrics(predictions, targets)
        return result.to_dict()


# ============================================================================
# Error Analysis Utilities
# ============================================================================

@dataclass
class ErrorAnalysis:
    """Container for error analysis results."""
    
    # Error categories
    incorrect_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error patterns
    common_errors: Dict[str, int] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Statistics
    error_rate_by_type: Dict[str, float] = field(default_factory=dict)
    hardest_questions: List[Dict[str, Any]] = field(default_factory=list)


def analyze_errors(
    predictions: List[str],
    ground_truths: List[List[str]],
    questions: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    question_ids: Optional[List[str]] = None,
    max_samples: int = 100,
) -> ErrorAnalysis:
    """
    Perform detailed error analysis on predictions.
    
    Args:
        predictions: Model predictions
        ground_truths: Ground truth answers (list of lists)
        questions: Optional question texts
        question_types: Optional question type labels
        question_ids: Optional question IDs
        max_samples: Maximum incorrect samples to collect
        
    Returns:
        ErrorAnalysis with detailed error breakdown
    """
    analysis = ErrorAnalysis()
    
    n = len(predictions)
    questions = questions or [''] * n
    question_types = question_types or ['unknown'] * n
    question_ids = question_ids or [str(i) for i in range(n)]
    
    # Collect incorrect predictions
    error_counts = defaultdict(int)
    type_errors = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, (pred, gts, q, qtype, qid) in enumerate(
        zip(predictions, ground_truths, questions, question_types, question_ids)
    ):
        vqa_score = vqa_accuracy(pred, gts)
        type_errors[qtype]['total'] += 1
        
        if vqa_score < 1.0:
            # This is an error (or partial error)
            if len(analysis.incorrect_samples) < max_samples:
                analysis.incorrect_samples.append({
                    'question_id': qid,
                    'question': q,
                    'prediction': pred,
                    'ground_truths': gts,
                    'vqa_score': vqa_score,
                    'question_type': qtype,
                })
            
            # Track error patterns
            pred_norm = normalize_answer(pred)
            gt_norm = normalize_answer(gts[0]) if gts else ''
            error_key = f"{pred_norm} → {gt_norm}"
            error_counts[error_key] += 1
            
            # Confusion matrix
            if pred_norm not in analysis.confusion_matrix:
                analysis.confusion_matrix[pred_norm] = defaultdict(int)
            analysis.confusion_matrix[pred_norm][gt_norm] += 1
        else:
            type_errors[qtype]['correct'] += 1
    
    # Compute error rates by type
    for qtype, data in type_errors.items():
        if data['total'] > 0:
            analysis.error_rate_by_type[qtype] = 100.0 * (
                1 - data['correct'] / data['total']
            )
    
    # Get most common errors
    analysis.common_errors = dict(
        sorted(error_counts.items(), key=lambda x: -x[1])[:20]
    )
    
    return analysis


# ============================================================================
# Utility Functions
# ============================================================================

def format_metrics_table(result: MetricsResult) -> str:
    """Format metrics as a nice table string."""
    lines = [
        "=" * 50,
        "VQA Evaluation Results",
        "=" * 50,
        f"  Total Samples:     {result.total_samples:,}",
        "-" * 50,
        f"  Exact Match:       {result.exact_match:.2f}%",
        f"  Normalized Match:  {result.normalized_match:.2f}%",
        f"  VQA Accuracy:      {result.vqa_accuracy:.2f}%",
        f"  Soft Accuracy:     {result.soft_accuracy:.2f}%",
    ]
    
    if result.per_type:
        lines.append("-" * 50)
        lines.append("  Per Question Type:")
        for qtype, data in sorted(result.per_type.items()):
            lines.append(
                f"    {qtype}: {data['vqa_accuracy']:.2f}% "
                f"(n={data['total']})"
            )
    
    lines.append("=" * 50)
    return '\n'.join(lines)


def save_metrics_json(
    result: MetricsResult,
    filepath: str,
    include_distribution: bool = True,
) -> None:
    """Save metrics to JSON file."""
    data = result.to_dict()
    
    if not include_distribution:
        data.pop('answer_distribution', None)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_metrics_json(filepath: str) -> MetricsResult:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return MetricsResult(**data)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core functions
    'normalize_answer',
    'exact_match',
    'normalized_match',
    'vqa_accuracy',
    'soft_accuracy',
    # Classes
    'VQAMetrics',
    'MetricsResult',
    'ErrorAnalysis',
    # Analysis
    'analyze_errors',
    # Utilities
    'format_metrics_table',
    'save_metrics_json',
    'load_metrics_json',
]
