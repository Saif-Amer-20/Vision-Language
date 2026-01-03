"""
Seed utilities for reproducibility.

Ensures deterministic behavior across PyTorch, NumPy, and Random.
"""

import random
import numpy as np
import torch
import os
from typing import Callable


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Use deterministic CUDA algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except RuntimeError:
                pass
    else:
        torch.backends.cudnn.benchmark = True
    
    print(f"🎲 Random seed: {seed} (deterministic={deterministic})")


def get_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Get worker init function for DataLoader reproducibility."""
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return worker_init_fn


def get_generator(seed: int) -> torch.Generator:
    """Get seeded generator for DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
