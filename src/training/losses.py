"""
Loss functions for VQA training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class VQALoss(nn.Module):
    """
    Loss function for VQA training.
    
    Supports:
    - Cross-entropy for generative VQA
    - Label smoothing
    - Optional auxiliary losses
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Initialize loss function.
        
        Args:
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            logits: Model logits [B, seq_len, vocab_size]
            labels: Target labels [B, seq_len]
            
        Returns:
            Dictionary with loss values
        """
        # Reshape for cross-entropy
        B, S, V = logits.shape
        loss = self.cross_entropy(
            logits.view(B * S, V),
            labels.view(B * S)
        )
        
        return {
            'loss': loss,
            'ce_loss': loss,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        return focal_loss.mean()
