"""
Logging utilities for experiment tracking.

Supports TensorBoard (default) and optional Weights & Biases.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
import torch


class ExperimentLogger:
    """Unified logger for TensorBoard and W&B."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.step = 0
        
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(log_dir, "tensorboard", experiment_name)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            print(f"📊 TensorBoard: {tb_dir}")
        
        # W&B
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "vlm-vqa",
                    entity=wandb_entity,
                    name=experiment_name,
                    config=config,
                    dir=log_dir
                )
                print(f"📊 W&B: {self.wandb_run.url}")
            except ImportError:
                print("⚠️ wandb not installed")
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log scalar value."""
        step = step if step is not None else self.step
        
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        
        if self.wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """Log dictionary of metrics."""
        step = step if step is not None else self.step
        
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log text."""
        step = step if step is not None else self.step
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
    
    def log_gpu_memory(self, step: Optional[int] = None) -> Optional[float]:
        """Log GPU memory usage."""
        if not torch.cuda.is_available():
            return None
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
        self.log_scalar("system/gpu_memory_gb", memory_gb, step)
        return memory_gb
    
    def set_step(self, step: int) -> None:
        """Set current step."""
        self.step = step
    
    def close(self) -> None:
        """Close loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics for printing."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }
