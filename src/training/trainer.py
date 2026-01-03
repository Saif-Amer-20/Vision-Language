"""
VQA Training Pipeline with Accelerate Integration.

Provides a complete training loop with:
- HuggingFace Accelerate for distributed training
- Mixed precision (fp16/bf16) support
- Gradient accumulation
- Checkpoint saving/resuming
- Early stopping
- TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import os
from tqdm import tqdm
from datetime import datetime

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


class VQATrainer:
    """
    Trainer for VQA models with Accelerate integration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        logger=None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: VQA model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration object
            logger: Experiment logger
        """
        self.config = config
        self.logger = logger
        
        # Setup accelerator
        self.accelerator = self._setup_accelerator()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer(model)
        self.scheduler = self._setup_scheduler()
        
        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler
        ) = self.accelerator.prepare(
            model, self.optimizer, train_loader, val_loader, self.scheduler
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # Output directory
        self.output_dir = Path(config.logging.output_dir) / config.logging.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(str(self.output_dir / "config.yaml"))
    
    def _setup_accelerator(self) -> 'Accelerator':
        """Setup Accelerate."""
        if not ACCELERATE_AVAILABLE:
            raise ImportError("accelerate is required. Install with: pip install accelerate")
        
        mixed_precision = None
        if self.config.training.fp16:
            mixed_precision = "fp16"
        elif self.config.training.bf16:
            mixed_precision = "bf16"
        
        return Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
        )
    
    def _setup_optimizer(self, model: nn.Module) -> AdamW:
        """Setup optimizer with parameter groups."""
        # Separate parameters with/without weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        params = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.training.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters()
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        return AdamW(params, lr=self.config.training.learning_rate)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        from src.training.schedulers import get_scheduler
        
        total_steps = self._get_total_steps()
        warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        
        return get_scheduler(
            name=self.config.training.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    
    def _get_total_steps(self) -> int:
        """Calculate total training steps."""
        if self.config.training.max_steps:
            return self.config.training.max_steps
        
        steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulation_steps
        return steps_per_epoch * self.config.training.num_epochs
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Training results dictionary
        """
        print(f"\n{'='*60}")
        print(f"🚀 STARTING TRAINING")
        print(f"{'='*60}")
        print(f"   Epochs: {self.config.training.num_epochs}")
        print(f"   Batch: {self.config.training.batch_size} x {self.config.training.gradient_accumulation_steps}")
        print(f"   LR: {self.config.training.learning_rate}")
        print(f"   Device: {self.accelerator.device}")
        print(f"   Mixed Precision: {self.accelerator.mixed_precision or 'disabled'}")
        print(f"{'='*60}\n")
        
        total_steps = self._get_total_steps()
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Log
            self._log_epoch(train_metrics, val_metrics)
            
            # Checkpoint
            is_best = val_metrics['loss'] < self.best_metric
            if is_best:
                self.best_metric = val_metrics['loss']
            
            if self.config.training.save_strategy == "epoch":
                self._save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config.training.early_stopping:
                if self._should_stop_early():
                    print("⏹️ Early stopping triggered")
                    break
            
            # Max steps check
            if self.config.training.max_steps and self.global_step >= self.config.training.max_steps:
                print("⏹️ Max steps reached")
                break
        
        return self._get_training_summary()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config.training.num_epochs}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                
                loss = outputs['loss']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Step logging
            if self.global_step % self.config.logging.log_every_n_steps == 0:
                if self.logger:
                    self.logger.log_scalar("train/loss", loss.item(), self.global_step)
                    self.logger.log_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
            
            # Step checkpoint
            if (self.config.training.save_strategy == "steps" and
                self.global_step % self.config.training.save_steps == 0):
                self._save_checkpoint()
            
            # Max steps check
            if self.config.training.max_steps and self.global_step >= self.config.training.max_steps:
                break
        
        return {
            'loss': total_loss / max(num_batches, 1),
        }
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        progress = tqdm(
            self.val_loader,
            desc="Validation",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch in progress:
            # Forward pass
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            # Generate predictions
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            predictions = unwrapped_model.generate(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(batch['answers'])
        
        # Compute metrics
        from src.data.answer_utils import compute_vqa_accuracy
        accuracy_metrics = compute_vqa_accuracy(all_predictions, all_targets)
        
        metrics = {
            'loss': total_loss / max(num_batches, 1),
            **accuracy_metrics,
        }
        
        return metrics
    
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log epoch results."""
        print(f"\n📊 Epoch {self.epoch + 1} Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}")
        print(f"   Val Loss:   {val_metrics['loss']:.4f}")
        print(f"   Val Exact:  {val_metrics.get('exact_match', 0):.4f}")
        print(f"   Val Norm:   {val_metrics.get('normalized_match', 0):.4f}")
        
        if self.logger:
            self.logger.log_metrics(train_metrics, self.global_step, prefix="train")
            self.logger.log_metrics(val_metrics, self.global_step, prefix="val")
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save checkpoint."""
        if not self.accelerator.is_local_main_process:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
        }
        
        from src.utils.io_utils import save_checkpoint
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{self.epoch + 1}.pt"
        save_checkpoint(state, str(checkpoint_path), is_best=is_best)
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        from src.utils.io_utils import load_checkpoint
        
        checkpoint = load_checkpoint(checkpoint_path, map_location=self.accelerator.device)
        
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint['model_state_dict']
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        print(f"📂 Resumed from epoch {self.epoch + 1}, step {self.global_step}")
    
    def _should_stop_early(self) -> bool:
        """Check early stopping condition."""
        # Simple implementation - would need patience tracking for full implementation
        return False
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'epochs_completed': self.epoch + 1,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'output_dir': str(self.output_dir),
        }
