"""
I/O utilities for checkpoints, JSON, and CSV handling.
"""

import torch
import json
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def save_checkpoint(
    state_dict: Dict[str, Any],
    path: str,
    is_best: bool = False,
    keep_last_n: int = 3
) -> None:
    """
    Save model checkpoint.
    
    Args:
        state_dict: State dictionary to save
        path: Save path
        is_best: Whether this is the best checkpoint
        keep_last_n: Number of recent checkpoints to keep
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    state_dict['timestamp'] = datetime.now().isoformat()
    torch.save(state_dict, path)
    print(f"💾 Saved checkpoint: {path}")
    
    if is_best:
        best_path = str(path).replace('.pt', '_best.pt')
        shutil.copy(path, best_path)
        print(f"⭐ Saved best: {best_path}")
    
    _cleanup_old_checkpoints(os.path.dirname(path), keep_last_n)


def load_checkpoint(
    path: str,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """Load checkpoint from path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(path, map_location=map_location)
    print(f"📂 Loaded checkpoint: {path}")
    
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}, Step: {checkpoint.get('global_step', 'N/A')}")
    
    return checkpoint


def _cleanup_old_checkpoints(directory: str, keep_n: int) -> None:
    """Remove old checkpoints, keeping only recent ones."""
    if not os.path.exists(directory):
        return
    
    checkpoints = []
    for f in os.listdir(directory):
        if f.endswith('.pt') and 'best' not in f:
            path = os.path.join(directory, f)
            checkpoints.append((path, os.path.getmtime(path)))
    
    checkpoints.sort(key=lambda x: x[1])
    
    while len(checkpoints) > keep_n:
        old_path, _ = checkpoints.pop(0)
        os.remove(old_path)
        print(f"🗑️ Removed old checkpoint: {old_path}")


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    print(f"📄 Saved JSON: {path}")


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_csv(
    data: List[Dict[str, Any]],
    path: str,
    fieldnames: Optional[List[str]] = None
) -> None:
    """Save list of dicts to CSV file."""
    if not data:
        return
    
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fieldnames = fieldnames or list(data[0].keys())
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"📊 Saved CSV: {path} ({len(data)} rows)")


def load_csv(path: str) -> List[Dict[str, Any]]:
    """Load CSV file to list of dicts."""
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def ensure_dir(path: str) -> str:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def get_experiment_dir(base_dir: str, name: str, timestamp: bool = True) -> str:
    """Create experiment directory with optional timestamp."""
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(base_dir, f"{name}_{ts}")
    else:
        exp_dir = os.path.join(base_dir, name)
    
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
