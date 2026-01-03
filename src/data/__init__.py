"""Data loading and preprocessing module."""

from .vqa_dataset import (
    VQADataset,
    VQADatasetConfig,
    create_dataloaders,
    vqa_collate_fn,
    get_sample_batch,
)
from .answer_utils import (
    normalize_answer,
    exact_match,
    normalized_match,
    soft_match,
    compute_vqa_accuracy,
    AnswerVocabulary,
)

__all__ = [
    # Dataset
    "VQADataset",
    "VQADatasetConfig",
    "create_dataloaders", 
    "vqa_collate_fn",
    "get_sample_batch",
    # Answer utilities
    "normalize_answer",
    "exact_match",
    "normalized_match",
    "soft_match",
    "compute_vqa_accuracy",
    "AnswerVocabulary",
]
