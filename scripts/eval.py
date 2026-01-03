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

from src.utils.config import load_config, ExperimentConfig
from src.utils.seed import set_seed
from src.utils.io_utils import load_checkpoint, save_json, ensure_dir
from src.data.vqa_dataset import create_dataloaders, VQADatasetConfig
from src.models.blip2_wrapper import create_model
from src.evaluation.evaluator import VQAEvaluator
from src.evaluation.error_analysis import ErrorAnalyzer, analyze_predictions_file
from src.evaluation.visualizations import plot_error_analysis


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
        print("🔍 Generating Comprehensive Error Analysis...")
        print("=" * 60)
        
        # Load predictions
        predictions_path = output_dir / "predictions.json"
        if predictions_path.exists():
            import json
            with open(predictions_path) as f:
                predictions_data = json.load(f)
            
            # Prepare prediction records for analyzer
            # Handle both list format and dict format from evaluator
            if isinstance(predictions_data, dict):
                # Extract from evaluator output format
                preds = predictions_data.get('predictions', [])
                gts = predictions_data.get('ground_truths', [])
                questions = predictions_data.get('questions', [])
                qids = predictions_data.get('question_ids', [])
                qtypes = predictions_data.get('question_types', [])
                iids = predictions_data.get('image_ids', [])
                
                prediction_records = []
                for i in range(len(preds)):
                    record = {
                        'prediction': preds[i] if i < len(preds) else '',
                        'ground_truths': gts[i] if i < len(gts) else [],
                        'question': questions[i] if i < len(questions) else '',
                        'question_id': qids[i] if i < len(qids) else str(i),
                        'question_type': qtypes[i] if i < len(qtypes) else None,
                        'image_id': iids[i] if i < len(iids) else None,
                    }
                    prediction_records.append(record)
            else:
                prediction_records = predictions_data
            
            # Get max samples from config
            max_samples = getattr(config.evaluation, 'error_analysis_samples', 500)
            
            # Create analyzer
            analyzer = ErrorAnalyzer(
                predictions=prediction_records,
                max_samples=max_samples,
            )
            
            # Run analysis
            print("   Analyzing predictions...")
            analysis_result = analyzer.analyze()
            
            # Save all outputs
            print("   Saving analysis files...")
            saved_files = analyzer.save_analysis(output_dir)
            
            print(f"   ✅ Markdown report: {saved_files.get('report', 'N/A')}")
            print(f"   ✅ JSON analysis: {saved_files.get('json', 'N/A')}")
            print(f"   ✅ Top errors CSV: {saved_files.get('csv', 'N/A')}")
            
            # Generate visualizations
            print("\n   Generating visualizations...")
            try:
                plot_paths = plot_error_analysis(
                    analysis=analysis_result,
                    output_dir=output_dir,
                )
                for name, path in plot_paths.items():
                    print(f"   ✅ {name}: {path}")
            except Exception as e:
                print(f"   ⚠️ Visualization generation failed: {e}")
                print("      (Install matplotlib and seaborn for plots)")
            
            # Print summary
            print("\n   📊 Error Analysis Summary:")
            print(f"      Total samples: {analysis_result.total_samples}")
            print(f"      Correct: {analysis_result.correct_count} ({analysis_result.overall_accuracy*100:.1f}%)")
            print(f"      Close misses: {analysis_result.close_miss_count}")
            print(f"      VQA Accuracy: {analysis_result.overall_vqa_accuracy*100:.2f}%")
            
            if analysis_result.error_types:
                print("\n   📉 Error Types:")
                for etype, count in analysis_result.error_types.items():
                    pct = analysis_result.error_type_percentages.get(etype, 0)
                    print(f"      {etype}: {count} ({pct:.1f}%)")
        else:
            print(f"   ⚠️ Predictions file not found: {predictions_path}")
            print("      Run evaluation with save_predictions=True first.")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    print(f"   Results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
