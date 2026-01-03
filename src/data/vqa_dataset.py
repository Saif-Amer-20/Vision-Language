"""
VQA Dataset Module with Proper Label Masking for Generative Training.

CRITICAL DESIGN:
- For generative VQA, we need to train the model to generate answers
- Input: Question prompt + Image
- Labels: Full sequence with question tokens MASKED to -100
- Only answer tokens should contribute to the loss

Prompt Template: "Question: {question} Answer:"
Full Training Sequence: "Question: {question} Answer: {answer}"
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class VQADatasetConfig:
    """Configuration for VQA dataset."""
    dataset_name: str = "HuggingFaceM4/VQAv2"
    train_split: str = "train"
    val_split: str = "validation"
    max_length: int = 128
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    max_samples: Optional[int] = None  # Limit samples for dev/debugging
    prompt_template: str = "Question: {question} Answer:"
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'VQADatasetConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class VQADataset(Dataset):
    """
    VQA Dataset with proper label masking for generative training.
    
    Key Features:
    - Loads VQAv2 from HuggingFace
    - Handles multiple answer formats
    - CRITICAL: Masks question tokens to -100 in labels
    - Memory-efficient image loading
    - Robust error handling with fallbacks
    """
    
    def __init__(
        self,
        processor,
        split: str = "train",
        dataset_name: str = "HuggingFaceM4/VQAv2",
        max_length: int = 128,
        max_samples: Optional[int] = None,
        prompt_template: str = "Question: {question} Answer:",
    ):
        """
        Initialize VQA Dataset.
        
        Args:
            processor: BLIP-2 processor (handles both image and text)
            split: Dataset split ('train', 'validation', 'test')
            dataset_name: HuggingFace dataset name
            max_length: Maximum sequence length for tokenization
            max_samples: Limit number of samples (for debugging)
            prompt_template: Template for question prompt
        """
        self.processor = processor
        self.split = split
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.max_samples = max_samples
        self.prompt_template = prompt_template
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Detect column names (different datasets have different schemas)
        self._detect_columns()
        
        logger.info(f"Loaded {len(self)} samples from {dataset_name} ({split})")
    
    def _load_dataset(self):
        """Load dataset with fallback strategies."""
        from datasets import load_dataset
        
        try:
            # Try loading with streaming for large datasets
            ds = load_dataset(
                self.dataset_name,
                split=self.split,
                trust_remote_code=True
            )
            
            # Apply sample limit if specified
            if self.max_samples is not None and self.max_samples < len(ds):
                ds = ds.select(range(self.max_samples))
                logger.info(f"Limited dataset to {self.max_samples} samples")
            
            return ds
            
        except Exception as e:
            logger.warning(f"Failed to load {self.dataset_name}: {e}")
            
            # Fallback: try loading the official VQAv2
            try:
                ds = load_dataset("lmms-lab/VQAv2", split=self.split)
                if self.max_samples is not None:
                    ds = ds.select(range(min(self.max_samples, len(ds))))
                return ds
            except Exception as e2:
                logger.error(f"All dataset loading attempts failed: {e2}")
                raise RuntimeError(f"Cannot load VQA dataset: {e}, {e2}")
    
    def _detect_columns(self):
        """Detect column names for flexible dataset support."""
        columns = self.dataset.column_names
        
        # Question column detection
        question_candidates = ['question', 'questions', 'text', 'query']
        self.question_col = next(
            (c for c in question_candidates if c in columns),
            columns[0] if columns else 'question'
        )
        
        # Answer column detection
        answer_candidates = ['answer', 'answers', 'multiple_choice_answer', 'label']
        self.answer_col = next(
            (c for c in answer_candidates if c in columns),
            None
        )
        
        # Image column detection
        image_candidates = ['image', 'images', 'image_path', 'image_bytes']
        self.image_col = next(
            (c for c in image_candidates if c in columns),
            'image'
        )
        
        logger.info(f"Detected columns - Question: {self.question_col}, "
                   f"Answer: {self.answer_col}, Image: {self.image_col}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single VQA sample with proper label masking.
        
        CRITICAL: Labels have question tokens set to -100 so only
        answer tokens contribute to the loss.
        
        Returns:
            Dict with:
                - pixel_values: Image tensor [3, H, W]
                - input_ids: Token IDs for question [seq_len]
                - attention_mask: Attention mask [seq_len]
                - labels: Labels with question masked to -100 [seq_len]
                - question: Raw question string
                - answer: Raw answer string
        """
        item = self.dataset[idx]
        
        # Get question
        question = self._get_question(item)
        
        # Get answer (handle multiple answer formats)
        answer = self._get_answer(item)
        
        # Get image
        image = self._load_image(item)
        
        # Create prompt and full sequence
        prompt = self.prompt_template.format(question=question)
        full_text = f"{prompt} {answer}"
        
        # Process image
        image_encoding = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        # CRITICAL: Tokenize with proper label masking
        # Step 1: Tokenize the full sequence (prompt + answer)
        full_encoding = self.processor.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Step 2: Tokenize just the prompt to find where answer starts
        prompt_encoding = self.processor.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt"
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]
        
        # Step 3: Create labels with question tokens masked to -100
        labels = full_encoding["input_ids"].clone()
        
        # Mask the prompt tokens (set to -100 so they don't contribute to loss)
        labels[0, :prompt_length] = -100
        
        # Also mask padding tokens
        padding_mask = full_encoding["attention_mask"] == 0
        labels[padding_mask] = -100
        
        return {
            "pixel_values": image_encoding["pixel_values"].squeeze(0),
            "input_ids": full_encoding["input_ids"].squeeze(0),
            "attention_mask": full_encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "question": question,
            "answer": answer,
        }
    
    def _get_question(self, item: Dict) -> str:
        """Extract question from item."""
        question = item.get(self.question_col, "")
        
        if isinstance(question, list):
            question = question[0] if question else ""
        
        return str(question).strip()
    
    def _get_answer(self, item: Dict) -> str:
        """
        Extract answer from item, handling multiple formats.
        
        VQAv2 has 10 annotated answers per question.
        For training, we typically use:
        1. multiple_choice_answer (most common human answer)
        2. Or randomly sample from answers list
        """
        # Try multiple_choice_answer first (most common answer)
        if 'multiple_choice_answer' in item:
            return str(item['multiple_choice_answer']).strip()
        
        # Try answers list
        if self.answer_col and self.answer_col in item:
            answer_data = item[self.answer_col]
            
            # Handle list of answer dicts
            if isinstance(answer_data, list):
                if len(answer_data) > 0:
                    if isinstance(answer_data[0], dict):
                        # VQAv2 format: [{"answer": "yes", "answer_confidence": ...}, ...]
                        answers = [a.get('answer', '') for a in answer_data]
                        # Randomly sample one answer for training diversity
                        return str(random.choice(answers)).strip()
                    else:
                        # Simple list of strings
                        return str(random.choice(answer_data)).strip()
            
            # Handle dict with 'answer' key
            if isinstance(answer_data, dict):
                return str(answer_data.get('answer', answer_data)).strip()
            
            # Handle string directly
            return str(answer_data).strip()
        
        # Fallback
        logger.warning(f"No answer found for item, using empty string")
        return ""
    
    def _load_image(self, item: Dict) -> Image.Image:
        """Load image from various formats with error handling."""
        try:
            image_data = item.get(self.image_col)
            
            # Already a PIL Image
            if isinstance(image_data, Image.Image):
                return image_data.convert("RGB")
            
            # Bytes
            if isinstance(image_data, bytes):
                return Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Dict with bytes (HuggingFace format)
            if isinstance(image_data, dict):
                if 'bytes' in image_data:
                    return Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
                if 'path' in image_data:
                    return Image.open(image_data['path']).convert("RGB")
            
            # File path
            if isinstance(image_data, (str, Path)):
                return Image.open(image_data).convert("RGB")
            
            # Unknown format - create placeholder
            logger.warning(f"Unknown image format: {type(image_data)}, using placeholder")
            return Image.new("RGB", (224, 224), color=(128, 128, 128))
            
        except Exception as e:
            logger.warning(f"Error loading image: {e}, using placeholder")
            return Image.new("RGB", (224, 224), color=(128, 128, 128))


def vqa_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for VQA batches.
    
    Handles:
    - Stacking tensors
    - Keeping raw strings as lists
    - Proper padding (already done in __getitem__)
    """
    # Stack tensors
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Keep raw strings as lists
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "questions": questions,
        "answers": answers,
    }


def create_dataloaders(
    processor,
    data_config: VQADatasetConfig,
    training_config: Any,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        processor: BLIP-2 processor
        data_config: Data configuration
        training_config: Training configuration (for batch_size)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Get batch size from training config
    batch_size = getattr(training_config, 'batch_size', 1)
    
    # Create datasets
    train_dataset = VQADataset(
        processor=processor,
        split=data_config.train_split,
        dataset_name=data_config.dataset_name,
        max_length=data_config.max_length,
        max_samples=data_config.max_samples,
        prompt_template=data_config.prompt_template,
    )
    
    val_dataset = VQADataset(
        processor=processor,
        split=data_config.val_split,
        dataset_name=data_config.dataset_name,
        max_length=data_config.max_length,
        max_samples=data_config.max_samples,  # Also limit val for dev mode
        prompt_template=data_config.prompt_template,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        collate_fn=vqa_collate_fn,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor if data_config.num_workers > 0 else None,
        drop_last=True,  # Drop incomplete batches for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        collate_fn=vqa_collate_fn,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor if data_config.num_workers > 0 else None,
        drop_last=False,
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, "
               f"Val: {len(val_loader)} batches (batch_size={batch_size})")
    
    return train_loader, val_loader


# Convenience function for quick testing
def get_sample_batch(processor, num_samples: int = 2) -> Dict[str, torch.Tensor]:
    """Get a sample batch for testing."""
    dataset = VQADataset(
        processor=processor,
        split="validation",
        max_samples=num_samples,
    )
    
    batch = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    return vqa_collate_fn(batch)
