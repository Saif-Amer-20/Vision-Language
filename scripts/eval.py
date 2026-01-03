#!/usr/bin/env python3
"""
Evaluation Script for VLM-VQA Research Project
==============================================

Entry point for evaluating trained BLIP-2 models.

Usage:
    # Evaluate a checkpoint
    python scripts/eval.py --checkpoint outputs/proposed/best_model.pt --config configs/proposed.yaml
    
    # Evaluate with different dataset split
    python scripts/eval.py --checkpoint outputs/proposed/best_model.pt \\
        --config configs/proposed.yaml --split test
    
    # Generate error analysis
    python scripts/eval.py --checkpoint outputs/proposed/best_model.pt \\
        --config configs/proposed.yaml --error_analysis

Author: VLM Research Team
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.io_utils import load_checkpoint, save_json, ensure_dir
from src.data.vqa_dataset import create_dataloaders
from src.models.blip2_wrapper import create_model
from src.evaluation.evaluator import VQAEvaluator
from src.evaluation.error_analysis import ErrorAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM-VQA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to evaluate on",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint dir)",
    )
    
    parser.add_argument(
        "--error_analysis",
        action="store_true",
        help="Generate detailed error analysis report",
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for evaluation",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Load configuration
    print(f"📋 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override data split
    if args.split == "test":
        config.data.val_split = "test"
    
    # Override max samples
    if args.max_samples:
        config.data.max_val_samples = args.max_samples
    
    # Set seed
    set_seed(config.seed)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_path = Path(args.checkpoint)
        output_dir = checkpoint_path.parent / "eval_results"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"eval_{args.split}_{timestamp}"
    ensure_dir(output_dir)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"💻 Device: {device}")
    
    print("\n" + "=" * 60)
    print("📦 Loading Dataset...")
    print("=" * 60)
    
    # Create dataloader (only validation)
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    _, val_loader = create_dataloaders(config.data, config.training)
    print(f"   Eval samples: {len(val_loader.dataset)}")
    
    print("\n" + "=" * 60)
    print("🤖 Loading Model...")
    print("=" * 60)
    
    # Create model
    model = create_model(config.model)
    
    # Load checkpoint
    print(f"   Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"   Checkpoint epoch: {checkpoint['epoch']}")
    if "global_step" in checkpoint:
        print(f"   Checkpoint step: {checkpoint['global_step']}")
    if "best_metric" in checkpoint:
        print(f"   Checkpoint best metric: {checkpoint['best_metric']:.4f}")
    
    print("\n" + "=" * 60)
    print("📊 Running Evaluation...")
    print("=" * 60)
    
    # Create evaluator
    evaluator = VQAEvaluator(
        model=model,
        dataloader=val_loader,
        config=config.evaluation,
        device=device,
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        save_predictions=True,
        output_dir=output_dir,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 Evaluation Results")
    print("=" * 60)
    
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"   {metric}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.4f}")
    
    # Save results
    save_json(results, output_dir / "eval_results.json")
    
    # Error analysis
    if args.error_analysis or config.evaluation.save_error_analysis:
        print("\n" + "=" * 60)
        print("🔍 Generating Error Analysis...")
        print("=" * 60)
        
        # Load predictions
        predictions_path = output_dir / "predictions.json"
        if predictions_path.exists():
            import json
            with open(predictions_path) as f:
                predictions = json.load(f)
            
            # Create analyzer
            analyzer = ErrorAnalyzer(
                predictions=predictions,
                max_samples=config.evaluation.error_analysis_samples,
            )
            
            # Generate report
            report = analyzer.generate_report()
            
            # Save report
            report_path = output_dir / "error_analysis.md"
            with open(report_path, "w") as f:
                f.write(report)
            
            print(f"   Error analysis saved to: {report_path}")
            
            # Save detailed analysis
            analysis = analyzer.analyze()
            save_json(analysis, output_dir / "error_analysis.json")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    print(f"   Results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
