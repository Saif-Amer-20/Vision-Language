"""
VQA Evaluation Pipeline.

Provides comprehensive evaluation with:
- Multiple metrics (Exact Match, Normalized Match, VQA Accuracy)
- Batch generation
- Result saving
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from pathlib import Path


class VQAEvaluator:
    """
    Evaluator for VQA models.
    """
    
    def __init__(
        self,
        model,
        dataloader: DataLoader,
        config,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: VQA model
            dataloader: Evaluation dataloader
            config: Configuration
            device: Device for evaluation (can be string or torch.device)
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config
        
        # Handle both string and torch.device
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
        
        self.model.eval()
        # Only move if not already on the correct device
        # This avoids conflicts when model was prepared by DeviceManager/Accelerate
        if next(model.parameters()).device != self.device:
            self.model.to(self.device)
    
    @torch.no_grad()
    def evaluate(self, save_predictions: bool = True) -> Dict[str, Any]:
        """
        Run evaluation.
        
        Args:
            save_predictions: Save predictions to file
            
        Returns:
            Evaluation results
        """
        all_predictions = []
        all_targets = []
        all_questions = []
        all_question_ids = []
        
        print("🔍 Running evaluation...")
        
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Generate predictions
            predictions = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(batch['answers'])
            all_questions.extend(batch['questions'])
            all_question_ids.extend(batch['question_ids'])
        
        # Compute metrics
        from src.evaluation.metrics import VQAMetrics
        metrics = VQAMetrics()
        results = metrics.compute(all_predictions, all_targets)
        
        # Add detailed results
        results['predictions'] = all_predictions
        results['targets'] = all_targets
        results['questions'] = all_questions
        results['question_ids'] = all_question_ids
        
        # Save predictions
        if save_predictions:
            self._save_predictions(results)
        
        return results
    
    def _save_predictions(self, results: Dict[str, Any]) -> None:
        """Save predictions to file."""
        output_dir = Path(self.config.logging.output_dir) / self.config.logging.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare prediction records
        predictions_list = []
        for i in range(len(results['predictions'])):
            predictions_list.append({
                'question_id': results['question_ids'][i],
                'question': results['questions'][i],
                'prediction': results['predictions'][i],
                'target': results['targets'][i],
                'correct': results['predictions'][i].lower().strip() == results['targets'][i].lower().strip(),
            })
        
        # Save as CSV
        from src.utils.io_utils import save_csv, save_json
        save_csv(predictions_list, str(output_dir / "predictions.csv"))
        
        # Save metrics as JSON
        metrics_only = {k: v for k, v in results.items() 
                       if k not in ['predictions', 'targets', 'questions', 'question_ids']}
        save_json(metrics_only, str(output_dir / "metrics.json"))
