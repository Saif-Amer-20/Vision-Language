#!/usr/bin/env python3
"""
Training Script for VLM-VQA Research Project
=============================================

Entry point for training BLIP-2 baseline and proposed models with Scene Reasoning.

Usage:
    # Full training on Colab
    python scripts/train.py --config configs/proposed.yaml
    
    # Mac development (auto-limited)
    python scripts/train.py --config configs/mac_dev.yaml
    
    # Smoke test
    python scripts/train.py --config configs/smoke_test.yaml
    
    # With overrides
    python scripts/train.py --config configs/baseline.yaml \\
        --training.learning_rate 2e-5 \\
        --training.num_epochs 5

Author: VLM Research Team
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils.config import (
    load_config,
    ExperimentConfig,
    ExecutionProfile,
    parse_cli_overrides,
)
from src.utils.seed import set_seed
from src.utils.io_utils import save_json, ensure_dir
from src.utils.logging_utils import ExperimentLogger
from src.utils.device_utils import DeviceManager, get_device_manager
from src.data.vqa_dataset import create_dataloaders, VQADatasetConfig
from src.models.blip2_wrapper import create_model
from src.training.trainer import VQATrainer
from src.evaluation.evaluator import VQAEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VLM-VQA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--execution_profile",
        type=str,
        choices=["colab_train", "mac_dev", "eval_only"],
        default=None,
        help="Execution profile (overrides config)",
    )
    
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run quick smoke test",
    )
    
    parser.add_argument(
        "--sync_to_drive",
        action="store_true",
        help="Sync outputs to Google Drive (Colab only)",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    # Allow arbitrary config overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format: --key.subkey value",
    )
    
    args, unknown = parser.parse_known_args()
    
    # Parse unknown args as config overrides
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]  # Remove --
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1
    
    args.overrides = overrides
    return args


def detect_execution_profile() -> ExecutionProfile:
    """Auto-detect execution environment."""
    try:
        import google.colab
        return ExecutionProfile.COLAB_TRAIN
    except ImportError:
        pass
    
    if sys.platform == "darwin":
        return ExecutionProfile.MAC_DEV
    
    if torch.cuda.is_available():
        return ExecutionProfile.COLAB_TRAIN
    
    return ExecutionProfile.MAC_DEV


def apply_profile_limits(config: ExperimentConfig, profile: ExecutionProfile) -> ExperimentConfig:
    """Apply safety limits based on execution profile."""
    if profile == ExecutionProfile.MAC_DEV:
        print("⚠️  MAC_DEV profile: Applying safety limits")
        
        # Enforce limits
        max_steps = config.runtime.mac_dev_max_steps
        max_samples = config.runtime.mac_dev_max_samples
        
        if config.training.max_steps is None or config.training.max_steps > max_steps:
            print(f"   → Limiting max_steps to {max_steps}")
            config.training.max_steps = max_steps
        
        if config.data.max_train_samples is None or config.data.max_train_samples > max_samples:
            print(f"   → Limiting max_train_samples to {max_samples}")
            config.data.max_train_samples = max_samples
        
        if config.data.max_val_samples is None or config.data.max_val_samples > max_samples // 2:
            config.data.max_val_samples = max_samples // 2
        
        # Disable fp16 on MPS
        if config.training.fp16:
            print("   → Disabling fp16 (MPS limitation)")
            config.training.fp16 = False
        
        # Don't save checkpoints
        if not config.runtime.mac_dev_save_checkpoints:
            config.training.save_total_limit = 0
    
    return config


def setup_directories(config: ExperimentConfig) -> dict:
    """Create output directories and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(config.logging.output_dir) / config.logging.experiment_name / timestamp
    
    dirs = {
        "experiment": experiment_dir,
        "checkpoints": experiment_dir / "checkpoints",
        "logs": experiment_dir / "logs",
        "results": experiment_dir / "results",
    }
    
    for dir_path in dirs.values():
        ensure_dir(dir_path)
    
    return dirs


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_args()
    
    # Parse CLI overrides (dot-notation like --training.learning_rate 2e-5)
    overrides = args.overrides if hasattr(args, 'overrides') else {}
    
    # Determine execution profile
    profile_str = args.execution_profile if args.execution_profile else None
    
    # Load configuration with all options
    print(f"📋 Loading config from: {args.config}")
    config = load_config(
        config_path=args.config,
        overrides=overrides,
        execution_profile=profile_str,
        smoke_test=args.smoke_test,
        apply_constraints=True,
        validate=True,
    )
    
    # Get the effective profile
    profile = config.runtime.get_profile()
    print(f"🔧 Execution profile: {profile.value}")
    
    # Resume checkpoint override
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    
    # Set seed
    set_seed(config.seed)
    print(f"🎲 Seed: {config.seed}")
    
    # Setup directories
    dirs = setup_directories(config)
    print(f"📁 Output directory: {dirs['experiment']}")
    
    # Save config
    save_json(config.to_dict(), dirs["experiment"] / "config.json")
    
    # Initialize DeviceManager for unified device placement
    # This replaces manual device detection and model.to(device)
    device_manager = get_device_manager(config, log_info=True)
    device = device_manager.device
    
    # Log device info to experiment config
    save_json(device_manager.get_info_dict(), dirs["experiment"] / "device_info.json")
    
    # Initialize logger
    logger = ExperimentLogger(
        log_dir=str(dirs["logs"]),
        experiment_name=config.logging.experiment_name,
        use_tensorboard=config.logging.use_tensorboard,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        wandb_entity=config.logging.wandb_entity,
        config=config.to_dict(),
    )
    
    print("\n" + "=" * 60)
    print("📦 Loading Dataset...")
    print("=" * 60)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config.data, config.training)
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    
    print("\n" + "=" * 60)
    print("🤖 Loading Model...")
    print("=" * 60)
    
    # Create model - loaded on CPU initially
    model = create_model(config)
    
    # Use DeviceManager for proper device placement
    # This avoids the device_map="auto" conflict with .to(device)
    model = device_manager.prepare_model(model)
    
    # Print memory estimate
    device_manager.print_memory_estimate(model, config.training.per_device_train_batch_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    print("\n" + "=" * 60)
    print("🏋️ Starting Training...")
    print("=" * 60)
    
    # Create trainer
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        output_dir=dirs["experiment"],
        device=device,
    )
    
    # Train
    train_results = trainer.train()
    
    print("\n" + "=" * 60)
    print("📊 Final Evaluation...")
    print("=" * 60)
    
    # Final evaluation
    evaluator = VQAEvaluator(
        model=model,
        dataloader=val_loader,
        config=config.evaluation,
        device=device,
    )
    
    eval_results = evaluator.evaluate(
        save_predictions=True,
        output_dir=dirs["results"],
    )
    
    # Log final metrics
    logger.log_metrics(eval_results, step=trainer.global_step, prefix="final")
    
    # Save results
    save_json(
        {
            "train_results": train_results,
            "eval_results": eval_results,
            "config": config.to_dict(),
        },
        dirs["results"] / "final_results.json",
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"   Best checkpoint: {trainer.best_checkpoint_path}")
    print(f"   Results saved to: {dirs['results']}")
    
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
    
    # Sync to Drive if requested
    if args.sync_to_drive or config.runtime.sync_to_drive:
        try:
            import shutil
            drive_path = Path(config.runtime.drive_output_path) / config.logging.experiment_name
            drive_path.mkdir(parents=True, exist_ok=True)
            
            # Copy results
            shutil.copytree(dirs["experiment"], drive_path / dirs["experiment"].name)
            print(f"\n☁️ Synced to Drive: {drive_path}")
        except Exception as e:
            print(f"\n⚠️ Drive sync failed: {e}")
    
    # Close logger
    logger.close()
    
    return eval_results


if __name__ == "__main__":
    main()
