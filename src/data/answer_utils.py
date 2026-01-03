"""
Answer normalization and vocabulary utilities for VQA.

Provides answer normalization, matching functions, and optional vocabulary building.
"""

import re
import string
from typing import List, Dict, Optional
from collections import Counter
import json


# Common articles to remove
ARTICLES = {'a', 'an', 'the'}

# Punctuation translation table
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Strip whitespace
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer
    """
    # Lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = answer.translate(PUNCT_TABLE)
    
    # Remove articles
    words = [w for w in answer.split() if w not in ARTICLES]
    
    # Rejoin and normalize whitespace
    answer = ' '.join(words).strip()
    answer = re.sub(r'\s+', ' ', answer)
    
    return answer


def exact_match(pred: str, target: str) -> bool:
    """Check exact match (case-insensitive, stripped)."""
    return pred.strip().lower() == target.strip().lower()


def normalized_match(pred: str, target: str) -> bool:
    """Check normalized match."""
    return normalize_answer(pred) == normalize_answer(target)


def soft_match(pred: str, target: str) -> float:
    """
    Compute soft match score using word overlap.
    
    Returns:
        Score between 0 and 1
    """
    pred_words = set(normalize_answer(pred).split())
    target_words = set(normalize_answer(target).split())
    
    if not target_words:
        return 1.0 if not pred_words else 0.0
    
    overlap = len(pred_words & target_words)
    return overlap / len(target_words)


def compute_vqa_accuracy(
    predictions: List[str],
    targets: List[str]
) -> Dict[str, float]:
    """
    Compute VQA accuracy metrics.
    
    Args:
        predictions: Predicted answers
        targets: Ground truth answers
        
    Returns:
        Dictionary with accuracy metrics
    """
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0, "normalized_match": 0.0, "total": 0}
    
    exact = sum(exact_match(p, t) for p, t in zip(predictions, targets))
    normalized = sum(normalized_match(p, t) for p, t in zip(predictions, targets))
    
    return {
        "exact_match": exact / n,
        "normalized_match": normalized / n,
        "total": n,
    }


class AnswerVocabulary:
    """
    Answer vocabulary for classification-based VQA (optional).
    """
    
    def __init__(
        self,
        min_freq: int = 5,
        max_vocab_size: Optional[int] = 3000,
        unk_token: str = "<UNK>"
    ):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.unk_token = unk_token
        
        self.answer_to_idx: Dict[str, int] = {}
        self.idx_to_answer: Dict[int, str] = {}
        self.answer_freq: Counter = Counter()
    
    def build(self, answers: List[str]) -> 'AnswerVocabulary':
        """Build vocabulary from answers."""
        normalized = [normalize_answer(a) for a in answers]
        self.answer_freq = Counter(normalized)
        
        # Filter by frequency
        filtered = [(a, c) for a, c in self.answer_freq.most_common()
                   if c >= self.min_freq]
        
        # Limit size
        if self.max_vocab_size:
            filtered = filtered[:self.max_vocab_size - 1]
        
        # Build mappings
        self.answer_to_idx = {self.unk_token: 0}
        self.idx_to_answer = {0: self.unk_token}
        
        for idx, (answer, _) in enumerate(filtered, start=1):
            self.answer_to_idx[answer] = idx
            self.idx_to_answer[idx] = answer
        
        print(f"📖 Built vocabulary: {len(self)} answers")
        return self
    
    def encode(self, answer: str) -> int:
        """Encode answer to index."""
        return self.answer_to_idx.get(normalize_answer(answer), 0)
    
    def decode(self, idx: int) -> str:
        """Decode index to answer."""
        return self.idx_to_answer.get(idx, self.unk_token)
    
    def __len__(self) -> int:
        return len(self.answer_to_idx)
    
    def save(self, path: str) -> None:
        """Save vocabulary."""
        with open(path, 'w') as f:
            json.dump({
                'answer_to_idx': self.answer_to_idx,
                'min_freq': self.min_freq,
                'max_vocab_size': self.max_vocab_size,
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AnswerVocabulary':
        """Load vocabulary."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        vocab = cls(
            min_freq=data.get('min_freq', 5),
            max_vocab_size=data.get('max_vocab_size')
        )
        vocab.answer_to_idx = data['answer_to_idx']
        vocab.idx_to_answer = {int(v): k for k, v in vocab.answer_to_idx.items()}
        return vocab
