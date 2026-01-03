#!/usr/bin/env python3
"""
Report Generation Script for VLM-VQA Research Project
======================================================

Generates thesis-ready comparison tables, summaries, and visualizations.

Usage:
    # Generate report from all experiments
    python scripts/make_report.py --results_dir outputs/
    
    # Compare specific experiments
    python scripts/make_report.py --experiments baseline proposed ablation_no_spatial
    
    # Generate LaTeX tables
    python scripts/make_report.py --results_dir outputs/ --latex

Author: VLM Research Team
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import save_json, load_json, save_csv


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate research reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs",
        help="Directory containing experiment outputs",
    )
    
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiment names to include",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="thesis_assets",
        help="Output directory for reports",
    )
    
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX-formatted tables",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "html", "latex", "all"],
        default="markdown",
        help="Output format for reports",
    )
    
    return parser.parse_args()


def find_experiment_results(results_dir: Path, experiment_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Find all experiment results in the results directory."""
    results = {}
    
    if not results_dir.exists():
        print(f"⚠️ Results directory not found: {results_dir}")
        return results
    
    # Look for experiment directories
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Filter by experiment names if specified
        if experiment_names and exp_name not in experiment_names:
            continue
        
        # Find the latest run
        runs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        
        if not runs:
            continue
        
        latest_run = runs[0]
        
        # Look for results file
        results_file = latest_run / "results" / "final_results.json"
        if not results_file.exists():
            results_file = latest_run / "results" / "eval_results.json"
        
        if results_file.exists():
            try:
                with open(results_file) as f:
                    exp_results = json.load(f)
                
                results[exp_name] = {
                    "path": str(latest_run),
                    "results": exp_results,
                    "timestamp": latest_run.name,
                }
            except Exception as e:
                print(f"⚠️ Error loading {results_file}: {e}")
    
    return results


def extract_metrics(results: Dict) -> Dict[str, float]:
    """Extract key metrics from results."""
    metrics = {}
    
    # Handle different result structures
    if "eval_results" in results:
        results = results["eval_results"]
    
    # Standard metrics
    metric_keys = [
        "exact_match",
        "normalized_match",
        "vqa_accuracy",
        "accuracy",
        "loss",
    ]
    
    for key in metric_keys:
        if key in results:
            metrics[key] = results[key]
    
    return metrics


def generate_comparison_table_markdown(experiments: Dict[str, Dict]) -> str:
    """Generate a Markdown comparison table."""
    if not experiments:
        return "No experiments found."
    
    # Collect all metrics
    all_metrics = set()
    for exp_data in experiments.values():
        metrics = extract_metrics(exp_data.get("results", {}))
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Build table
    lines = []
    lines.append("# Experiment Comparison Results\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Header
    header = "| Experiment |"
    separator = "|------------|"
    for metric in all_metrics:
        header += f" {metric.replace('_', ' ').title()} |"
        separator += "--------|"
    
    lines.append(header)
    lines.append(separator)
    
    # Rows
    for exp_name, exp_data in sorted(experiments.items()):
        metrics = extract_metrics(exp_data.get("results", {}))
        row = f"| {exp_name} |"
        for metric in all_metrics:
            value = metrics.get(metric, "-")
            if isinstance(value, float):
                row += f" {value:.4f} |"
            else:
                row += f" {value} |"
        lines.append(row)
    
    # Add summary
    lines.append("\n## Summary\n")
    
    # Find best for each metric
    for metric in all_metrics:
        values = []
        for exp_name, exp_data in experiments.items():
            exp_metrics = extract_metrics(exp_data.get("results", {}))
            if metric in exp_metrics:
                values.append((exp_name, exp_metrics[metric]))
        
        if values:
            if "loss" in metric:
                best = min(values, key=lambda x: x[1])
            else:
                best = max(values, key=lambda x: x[1])
            lines.append(f"- **Best {metric.replace('_', ' ').title()}**: {best[0]} ({best[1]:.4f})")
    
    return "\n".join(lines)


def generate_comparison_table_latex(experiments: Dict[str, Dict]) -> str:
    """Generate a LaTeX comparison table."""
    if not experiments:
        return "% No experiments found."
    
    # Collect all metrics
    all_metrics = set()
    for exp_data in experiments.values():
        metrics = extract_metrics(exp_data.get("results", {}))
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Build table
    lines = []
    lines.append("% Experiment Comparison Table")
    lines.append("% Generated automatically - do not edit")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of VQA Model Variants}")
    lines.append("\\label{tab:results}")
    
    # Column specification
    col_spec = "l" + "c" * len(all_metrics)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header
    header = "Model & " + " & ".join([m.replace("_", " ").title() for m in all_metrics]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Find best values for each metric (for bold formatting)
    best_values = {}
    for metric in all_metrics:
        values = []
        for exp_data in experiments.values():
            exp_metrics = extract_metrics(exp_data.get("results", {}))
            if metric in exp_metrics:
                values.append(exp_metrics[metric])
        if values:
            if "loss" in metric:
                best_values[metric] = min(values)
            else:
                best_values[metric] = max(values)
    
    # Rows
    for exp_name, exp_data in sorted(experiments.items()):
        metrics = extract_metrics(exp_data.get("results", {}))
        
        # Format experiment name
        display_name = exp_name.replace("_", " ").title()
        if "proposed" in exp_name.lower():
            display_name = f"\\textbf{{{display_name}}}"
        
        row = display_name
        for metric in all_metrics:
            value = metrics.get(metric)
            if value is not None:
                if abs(value - best_values.get(metric, 0)) < 1e-6:
                    row += f" & \\textbf{{{value:.4f}}}"
                else:
                    row += f" & {value:.4f}"
            else:
                row += " & -"
        row += " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_detailed_report(experiments: Dict[str, Dict]) -> str:
    """Generate a detailed markdown report."""
    lines = []
    
    lines.append("# VLM-VQA Research Report\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    lines.append("## Overview\n")
    lines.append(f"This report compares {len(experiments)} experimental configurations:\n")
    
    for exp_name in sorted(experiments.keys()):
        lines.append(f"- **{exp_name}**")
    
    lines.append("\n---\n")
    
    # Results table
    lines.append("## Results Comparison\n")
    lines.append(generate_comparison_table_markdown(experiments))
    
    lines.append("\n---\n")
    
    # Per-experiment details
    lines.append("## Detailed Results\n")
    
    for exp_name, exp_data in sorted(experiments.items()):
        lines.append(f"### {exp_name.replace('_', ' ').title()}\n")
        
        lines.append(f"- **Run path**: `{exp_data.get('path', 'N/A')}`")
        lines.append(f"- **Timestamp**: {exp_data.get('timestamp', 'N/A')}")
        lines.append("")
        
        metrics = extract_metrics(exp_data.get("results", {}))
        if metrics:
            lines.append("**Metrics:**\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    lines.append(f"- {metric.replace('_', ' ').title()}: {value}")
        
        lines.append("")
    
    lines.append("\n---\n")
    
    # Architecture diagram
    lines.append("## Model Architecture\n")
    lines.append("```")
    lines.append("┌─────────────────────────────────────────────────────────────┐")
    lines.append("│                        BLIP-2 VQA Model                     │")
    lines.append("├─────────────────────────────────────────────────────────────┤")
    lines.append("│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │")
    lines.append("│  │   Vision    │    │   Q-Former  │    │       LLM       │  │")
    lines.append("│  │   Encoder   │───▶│  (Querying  │───▶│   (OPT-2.7B)    │  │")
    lines.append("│  │  (ViT-G)    │    │ Transformer)│    │                 │  │")
    lines.append("│  └─────────────┘    └──────┬──────┘    └─────────────────┘  │")
    lines.append("│                            │                                │")
    lines.append("│                     ┌──────▼──────┐                         │")
    lines.append("│                     │   Scene     │  ◀── Proposed Module    │")
    lines.append("│                     │  Reasoning  │                         │")
    lines.append("│                     │   Module    │                         │")
    lines.append("│                     └─────────────┘                         │")
    lines.append("└─────────────────────────────────────────────────────────────┘")
    lines.append("```\n")
    
    # Ablation analysis
    ablation_exps = [e for e in experiments if "ablation" in e.lower()]
    if ablation_exps:
        lines.append("## Ablation Study\n")
        lines.append("The following ablation experiments were conducted:\n")
        
        for exp_name in ablation_exps:
            metrics = extract_metrics(experiments[exp_name].get("results", {}))
            accuracy = metrics.get("exact_match", metrics.get("accuracy", "N/A"))
            if isinstance(accuracy, float):
                accuracy = f"{accuracy:.4f}"
            lines.append(f"- **{exp_name}**: Accuracy = {accuracy}")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_csv_results(experiments: Dict[str, Dict], output_path: Path):
    """Generate CSV file with all results."""
    # Collect all metrics
    all_metrics = set()
    for exp_data in experiments.values():
        metrics = extract_metrics(exp_data.get("results", {}))
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Build rows
    rows = []
    for exp_name, exp_data in sorted(experiments.items()):
        metrics = extract_metrics(exp_data.get("results", {}))
        row = {"experiment": exp_name, "timestamp": exp_data.get("timestamp", "")}
        for metric in all_metrics:
            row[metric] = metrics.get(metric, "")
        rows.append(row)
    
    # Save
    fieldnames = ["experiment", "timestamp"] + list(all_metrics)
    save_csv(rows, output_path, fieldnames=fieldnames)


def main():
    """Main report generation entry point."""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Results directory: {results_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Find experiments
    print("\n🔍 Finding experiments...")
    experiments = find_experiment_results(results_dir, args.experiments)
    
    if not experiments:
        print("⚠️ No experiments found!")
        return
    
    print(f"   Found {len(experiments)} experiments:")
    for exp_name in experiments:
        print(f"      - {exp_name}")
    
    # Generate reports
    print("\n📝 Generating reports...")
    
    # Markdown report
    if args.format in ["markdown", "all"]:
        report = generate_detailed_report(experiments)
        report_path = output_dir / "experiment_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"   ✅ Markdown report: {report_path}")
    
    # LaTeX table
    if args.latex or args.format in ["latex", "all"]:
        latex_table = generate_comparison_table_latex(experiments)
        latex_path = output_dir / "results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        print(f"   ✅ LaTeX table: {latex_path}")
    
    # CSV results
    csv_path = output_dir / "experiment_results.csv"
    generate_csv_results(experiments, csv_path)
    print(f"   ✅ CSV results: {csv_path}")
    
    # JSON summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "num_experiments": len(experiments),
        "experiments": {
            name: {
                "timestamp": data.get("timestamp"),
                "metrics": extract_metrics(data.get("results", {})),
            }
            for name, data in experiments.items()
        },
    }
    json_path = output_dir / "experiment_summary.json"
    save_json(summary, json_path)
    print(f"   ✅ JSON summary: {json_path}")
    
    print("\n✅ Report generation complete!")


if __name__ == "__main__":
    main()
