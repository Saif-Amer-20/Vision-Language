"""
Learning rate schedulers for training.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from typing import Optional


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        name: Scheduler name (cosine, linear, constant)
        optimizer: Optimizer
        num_warmup_steps: Warmup steps
        num_training_steps: Total training steps
        
    Returns:
        Learning rate scheduler
    """
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LRScheduler:
    """Cosine scheduler with warmup."""
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LRScheduler:
    """Linear scheduler with warmup."""
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        return max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
        )
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LRScheduler:
    """Constant scheduler with warmup."""
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)
