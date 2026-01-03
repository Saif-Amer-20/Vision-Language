"""
VQA Evaluation Pipeline - Complete Evaluation System.

Provides comprehensive evaluation with:
- Official VQAv2 metrics (exact match, normalized match, VQA accuracy)
- Multiple ground truth answer handling
- Per-question-type breakdown
- Error analysis and diagnostics
- Result saving (JSON and CSV)
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm
from pathlib import Path
import json
import csv
from collections import Counter

from src.evaluation.metrics import (
    VQAMetrics,
    MetricsResult,
    ErrorAnalysis,
    analyze_errors,
    format_metrics_table,
    normalize_answer,
    vqa_accuracy,
)


class VQAEvaluator:
    """
    Complete evaluator for VQA models following official VQAv2 protocol.
    
    Features:
    - Handles multiple ground truth answers per question
    - Computes official VQA accuracy: min(count/3, 1.0)
    - Per-question-type metrics breakdown
    - Detailed error analysis
    - Exports predictions to JSON and CSV
    - Answer distribution analysis
    
    Usage:
        evaluator = VQAEvaluator(model, dataloader, config, device)
        results = evaluator.evaluate(save_predictions=True, output_dir="./results")
        
        # Access metrics
        print(f"VQA Accuracy: {results['vqa_accuracy']:.2f}%")
    """
    
    def __init__(
        self,
        model,
        dataloader: DataLoader,
        config,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: VQA model with .generate() method
            dataloader: Evaluation dataloader
            config: Configuration (EvaluationConfig or full config)
            device: Device for evaluation
        """
        self.model = model
        self.dataloader = dataloader
        
        # Extract evaluation config if nested
        if hasattr(config, 'evaluation'):
            self.eval_config = config.evaluation
            self.full_config = config
        else:
            self.eval_config = config
            self.full_config = config
        
        # Handle device types
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
        
        # Initialize metrics calculator
        self.metrics = VQAMetrics(verbose=False)
        
        # Set model to eval mode and move to device if needed
        self.model.eval()
        try:
            param_device = next(model.parameters()).device
            if param_device != self.device:
                self.model.to(self.device)
        except StopIteration:
            # Model has no parameters (unlikely but handle gracefully)
            pass
    
    @torch.no_grad()
    def evaluate(
        self,
        save_predictions: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        compute_error_analysis: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            save_predictions: Save predictions to files
            output_dir: Directory to save results (uses config if None)
            compute_error_analysis: Perform detailed error analysis
            verbose: Print progress and results
            
        Returns:
            Dictionary containing:
            - Core metrics (exact_match, normalized_match, vqa_accuracy)
            - Per-type breakdown (if question types available)
            - Error analysis (if requested)
            - All predictions and ground truths
        """
        if verbose:
            print("\n" + "=" * 60)
            print("🔍 VQA Evaluation")
            print("=" * 60)
        
        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        all_questions = []
        all_question_ids = []
        all_question_types = []
        all_image_ids = []
        
        progress_bar = tqdm(
            self.dataloader,
            desc="Evaluating",
            disable=not verbose,
        )
        
        for batch in progress_bar:
            # Move inputs to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Generate predictions
            predictions = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Collect results
            all_predictions.extend(predictions)
            
            # Handle ground truths (can be single string or list of strings)
            if 'answers' in batch:
                answers = batch['answers']
                # Normalize to list of lists for multiple GT handling
                for ans in answers:
                    if isinstance(ans, str):
                        all_ground_truths.append([ans])
                    elif isinstance(ans, list):
                        all_ground_truths.append(ans)
                    else:
                        all_ground_truths.append([str(ans)])
            elif 'answer' in batch:
                # Single answer format
                for ans in batch['answer']:
                    all_ground_truths.append([ans] if isinstance(ans, str) else ans)
            
            # Optional metadata
            if 'questions' in batch:
                all_questions.extend(batch['questions'])
            elif 'question' in batch:
                all_questions.extend(batch['question'])
            
            if 'question_ids' in batch:
                all_question_ids.extend(batch['question_ids'])
            elif 'question_id' in batch:
                all_question_ids.extend(batch['question_id'])
            
            if 'question_types' in batch:
                all_question_types.extend(batch['question_types'])
            elif 'question_type' in batch:
                all_question_types.extend(batch['question_type'])
            elif 'answer_type' in batch:
                all_question_types.extend(batch['answer_type'])
            
            if 'image_ids' in batch:
                all_image_ids.extend(batch['image_ids'])
            elif 'image_id' in batch:
                all_image_ids.extend(batch['image_id'])
        
        # Compute metrics
        if verbose:
            print("\n📊 Computing metrics...")
        
        question_types_for_metrics = all_question_types if all_question_types else None
        
        metrics_result = self.metrics.compute_metrics(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            question_types=question_types_for_metrics,
            compute_distribution=True,
        )
        
        # Print results
        if verbose:
            print(format_metrics_table(metrics_result))
        
        # Build results dictionary
        results = metrics_result.to_dict()
        
        # Add raw data
        results['predictions'] = all_predictions
        results['ground_truths'] = all_ground_truths
        results['questions'] = all_questions
        results['question_ids'] = all_question_ids
        results['question_types'] = all_question_types
        results['image_ids'] = all_image_ids
        
        # Error analysis
        if compute_error_analysis and getattr(self.eval_config, 'save_error_analysis', True):
            if verbose:
                print("\n🔬 Performing error analysis...")
            
            error_analysis = analyze_errors(
                predictions=all_predictions,
                ground_truths=all_ground_truths,
                questions=all_questions,
                question_types=all_question_types if all_question_types else None,
                question_ids=[str(qid) for qid in all_question_ids] if all_question_ids else None,
                max_samples=100,
            )
            
            results['error_analysis'] = {
                'incorrect_samples': error_analysis.incorrect_samples,
                'common_errors': error_analysis.common_errors,
                'error_rate_by_type': error_analysis.error_rate_by_type,
            }
            
            if verbose and error_analysis.common_errors:
                print("\n  Most common errors:")
                for i, (error, count) in enumerate(
                    list(error_analysis.common_errors.items())[:5]
                ):
                    print(f"    {i+1}. {error}: {count} times")
        
        # Answer distribution analysis
        results['predicted_answer_distribution'] = self._analyze_predictions(all_predictions)
        results['gt_answer_distribution'] = self._analyze_ground_truths(all_ground_truths)
        
        # Save results
        if save_predictions:
            if output_dir is None:
                if hasattr(self.full_config, 'logging'):
                    output_dir = Path(self.full_config.logging.output_dir) / \
                                 self.full_config.logging.experiment_name / "evaluation"
                else:
                    output_dir = Path("./outputs/evaluation")
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(results, output_dir, verbose)
        
        return results
    
    def _analyze_predictions(
        self,
        predictions: List[str],
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Analyze prediction distribution."""
        normalized = [normalize_answer(p) for p in predictions]
        counter = Counter(normalized)
        
        return {
            'total_unique': len(counter),
            'top_answers': dict(counter.most_common(top_k)),
            'single_occurrence': sum(1 for c in counter.values() if c == 1),
        }
    
    def _analyze_ground_truths(
        self,
        ground_truths: List[List[str]],
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Analyze ground truth distribution."""
        # Use first GT answer for distribution analysis
        first_answers = [normalize_answer(gts[0]) if gts else '' for gts in ground_truths]
        counter = Counter(first_answers)
        
        # Compute agreement statistics
        agreement_scores = []
        for gts in ground_truths:
            if len(gts) > 1:
                normalized_gts = [normalize_answer(gt) for gt in gts]
                most_common = Counter(normalized_gts).most_common(1)[0][1]
                agreement_scores.append(most_common / len(gts))
        
        avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 1.0
        
        return {
            'total_unique': len(counter),
            'top_answers': dict(counter.most_common(top_k)),
            'avg_annotator_agreement': avg_agreement,
        }
    
    def _save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        verbose: bool = True,
    ) -> None:
        """Save evaluation results to files."""
        if verbose:
            print(f"\n💾 Saving results to: {output_dir}")
        
        # Prepare prediction records
        predictions_records = []
        n_samples = len(results['predictions'])
        
        for i in range(n_samples):
            pred = results['predictions'][i]
            gts = results['ground_truths'][i]
            
            # Compute per-sample metrics
            score = vqa_accuracy(pred, gts)
            
            record = {
                'index': i,
                'prediction': pred,
                'ground_truths': gts,
                'primary_gt': gts[0] if gts else '',
                'vqa_score': score,
                'is_correct': score >= 1.0,
            }
            
            # Add optional fields
            if results['question_ids'] and i < len(results['question_ids']):
                record['question_id'] = results['question_ids'][i]
            if results['questions'] and i < len(results['questions']):
                record['question'] = results['questions'][i]
            if results['question_types'] and i < len(results['question_types']):
                record['question_type'] = results['question_types'][i]
            if results['image_ids'] and i < len(results['image_ids']):
                record['image_id'] = results['image_ids'][i]
            
            predictions_records.append(record)
        
        # Save predictions as JSON
        if getattr(self.eval_config, 'output_json', True):
            json_path = output_dir / "predictions.json"
            with open(json_path, 'w') as f:
                json.dump(predictions_records, f, indent=2)
            if verbose:
                print(f"  ✓ Predictions JSON: {json_path}")
        
        # Save predictions as CSV
        if getattr(self.eval_config, 'output_csv', True):
            csv_path = output_dir / "predictions.csv"
            
            # Flatten for CSV (convert lists to strings)
            csv_records = []
            for rec in predictions_records:
                csv_rec = rec.copy()
                if isinstance(csv_rec.get('ground_truths'), list):
                    csv_rec['ground_truths'] = ' | '.join(csv_rec['ground_truths'])
                csv_records.append(csv_rec)
            
            if csv_records:
                fieldnames = list(csv_records[0].keys())
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_records)
                if verbose:
                    print(f"  ✓ Predictions CSV: {csv_path}")
        
        # Save metrics summary
        metrics_path = output_dir / "metrics.json"
        metrics_only = {
            k: v for k, v in results.items()
            if k not in ['predictions', 'ground_truths', 'questions', 
                        'question_ids', 'question_types', 'image_ids']
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_only, f, indent=2)
        if verbose:
            print(f"  ✓ Metrics JSON: {metrics_path}")
        
        # Save error analysis if present
        if 'error_analysis' in results and getattr(self.eval_config, 'save_error_analysis', True):
            error_path = output_dir / "error_analysis.json"
            with open(error_path, 'w') as f:
                json.dump(results['error_analysis'], f, indent=2)
            if verbose:
                print(f"  ✓ Error analysis: {error_path}")
    
    def compute_metrics_from_file(
        self,
        predictions_file: Union[str, Path],
        ground_truths_file: Optional[Union[str, Path]] = None,
    ) -> MetricsResult:
        """
        Compute metrics from saved prediction files.
        
        Useful for re-evaluating without running inference.
        
        Args:
            predictions_file: Path to predictions JSON
            ground_truths_file: Optional separate ground truths file
            
        Returns:
            MetricsResult with computed metrics
        """
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        if isinstance(predictions_data, list):
            predictions = [r['prediction'] for r in predictions_data]
            ground_truths = [r['ground_truths'] for r in predictions_data]
            question_types = [r.get('question_type') for r in predictions_data]
            question_types = question_types if all(question_types) else None
        else:
            predictions = predictions_data['predictions']
            ground_truths = predictions_data['ground_truths']
            question_types = predictions_data.get('question_types')
        
        if ground_truths_file:
            with open(ground_truths_file, 'r') as f:
                ground_truths = json.load(f)
        
        return self.metrics.compute_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            question_types=question_types,
        )


def evaluate_vqa(
    model,
    dataloader: DataLoader,
    config,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for VQA evaluation.
    
    Args:
        model: VQA model
        dataloader: Evaluation dataloader
        config: Configuration
        device: Compute device
        output_dir: Output directory
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = VQAEvaluator(model, dataloader, config, device)
    return evaluator.evaluate(
        save_predictions=True,
        output_dir=output_dir,
    )


__all__ = [
    'VQAEvaluator',
    'evaluate_vqa',
]
