"""
Configuration System for VLM-VQA Research Project.

Provides YAML-based configuration with:
- Type-safe dataclass configuration objects
- CLI argument overrides
- Execution profiles (colab_train, mac_dev, eval_only)
- Environment auto-detection
- Safety guards for local development
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import yaml
import os
import sys
import platform
import argparse
from pathlib import Path


class ExecutionProfile(Enum):
    """Execution profile for different environments."""
    COLAB_TRAIN = "colab_train"   # Full training on Colab GPU
    MAC_DEV = "mac_dev"           # Local development (smoke tests only)
    EVAL_ONLY = "eval_only"       # Evaluation only (no training)
    
    @classmethod
    def from_string(cls, s: str) -> 'ExecutionProfile':
        """Create from string value."""
        mapping = {
            'colab_train': cls.COLAB_TRAIN,
            'mac_dev': cls.MAC_DEV,
            'eval_only': cls.EVAL_ONLY,
        }
        if s.lower() not in mapping:
            raise ValueError(f"Unknown profile: {s}. Valid: {list(mapping.keys())}")
        return mapping[s.lower()]


@dataclass
class RuntimeConfig:
    """Runtime configuration for execution environment."""
    execution_profile: str = "colab_train"
    sync_to_drive: bool = False
    drive_output_path: str = "/content/drive/MyDrive/VLM_Thesis_Outputs"
    
    # Safety limits for mac_dev profile
    mac_dev_max_steps: int = 10
    mac_dev_max_samples: int = 50
    mac_dev_save_checkpoints: bool = False
    
    def get_profile(self) -> ExecutionProfile:
        return ExecutionProfile.from_string(self.execution_profile)
    
    def is_full_training_allowed(self) -> bool:
        return self.get_profile() == ExecutionProfile.COLAB_TRAIN


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "HuggingFaceM4/VQAv2"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    
    # Preprocessing
    image_size: int = 224
    max_question_length: int = 32
    max_answer_length: int = 16
    
    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True
    
    # Prompt template for generative VQA
    prompt_template: str = "Question: {question} Answer:"
    
    # Cache
    cache_dir: Optional[str] = None


@dataclass 
class ModelConfig:
    """Model configuration."""
    # Base model
    model_name: str = "Salesforce/blip2-opt-2.7b"
    torch_dtype: str = "float16"  # float16, bfloat16, float32
    
    # Component freezing
    freeze_vision_encoder: bool = True
    freeze_llm: bool = True
    freeze_qformer: bool = False
    
    # Scene Reasoning Module
    use_scene_reasoning: bool = False
    scene_hidden_dim: int = 768
    scene_num_heads: int = 8
    scene_num_layers: int = 2
    scene_mlp_ratio: float = 4.0
    scene_dropout: float = 0.1
    
    # Ablation controls
    use_spatial_encoding: bool = True
    use_relation_attention: bool = True
    spatial_encoding_dim: int = 64
    
    # Generation settings
    max_new_tokens: int = 16
    num_beams: int = 3
    do_sample: bool = False
    temperature: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Batch and accumulation
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 8  # batch_size * gradient_accumulation_steps
    
    # Optimization
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Schedule
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant
    
    # Precision
    fp16: bool = True
    bf16: bool = False
    
    # Memory optimization
    gradient_checkpointing: bool = False
    
    # Device
    device: str = "auto"  # auto, cuda, mps, cpu
    
    # Checkpointing
    save_strategy: str = "epoch"  # epoch, steps
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3
    
    # Smoke test
    smoke_test: bool = False
    smoke_test_samples: int = 32
    smoke_test_steps: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    output_dir: str = "./outputs"
    experiment_name: str = "vqa_experiment"
    
    # Logging backends
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "vlm-vqa-research"
    wandb_entity: Optional[str] = None
    
    # Logging frequency
    log_every_n_steps: int = 10
    
    # Save artifacts
    save_predictions: bool = True
    save_attention_maps: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Metrics
    compute_exact_match: bool = True
    compute_normalized_match: bool = True
    compute_vqa_accuracy: bool = True
    
    # Error analysis
    save_error_analysis: bool = True
    error_analysis_samples: int = 500
    
    # Output formats
    output_csv: bool = True
    output_json: bool = True


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict or {})
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**d.get('data', {})),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            logging=LoggingConfig(**d.get('logging', {})),
            evaluation=EvaluationConfig(**d.get('evaluation', {})),
            runtime=RuntimeConfig(**d.get('runtime', {})),
            seed=d.get('seed', 42),
        )
    
    def apply_cli_overrides(self, args: argparse.Namespace) -> 'Config':
        """Apply CLI argument overrides to config."""
        # Execution profile
        if hasattr(args, 'execution_profile') and args.execution_profile:
            self.runtime.execution_profile = args.execution_profile
        
        # Smoke test
        if hasattr(args, 'smoke_test') and args.smoke_test:
            self.training.smoke_test = True
            self.training.max_steps = self.training.smoke_test_steps
            self.data.max_train_samples = self.training.smoke_test_samples
            self.data.max_val_samples = self.training.smoke_test_samples // 2
        
        # Common overrides
        if hasattr(args, 'batch_size') and args.batch_size:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'lr') and args.lr:
            self.training.learning_rate = args.lr
        if hasattr(args, 'epochs') and args.epochs:
            self.training.num_epochs = args.epochs
        if hasattr(args, 'output_dir') and args.output_dir:
            self.logging.output_dir = args.output_dir
        if hasattr(args, 'experiment_name') and args.experiment_name:
            self.logging.experiment_name = args.experiment_name
        if hasattr(args, 'resume') and args.resume:
            self.training.resume_from_checkpoint = args.resume
        if hasattr(args, 'sync_to_drive') and args.sync_to_drive:
            self.runtime.sync_to_drive = True
        
        # Apply profile constraints
        self._apply_profile_constraints()
        
        return self
    
    def _apply_profile_constraints(self) -> None:
        """Apply constraints based on execution profile."""
        profile = self.runtime.get_profile()
        
        if profile == ExecutionProfile.MAC_DEV:
            # Safety limits
            if self.training.max_steps is None or self.training.max_steps > self.runtime.mac_dev_max_steps:
                print(f"⚠️ mac_dev: Limiting max_steps to {self.runtime.mac_dev_max_steps}")
                self.training.max_steps = self.runtime.mac_dev_max_steps
            
            if self.data.max_train_samples is None or self.data.max_train_samples > self.runtime.mac_dev_max_samples:
                print(f"⚠️ mac_dev: Limiting train samples to {self.runtime.mac_dev_max_samples}")
                self.data.max_train_samples = self.runtime.mac_dev_max_samples
                self.data.max_val_samples = self.runtime.mac_dev_max_samples // 2
            
            # Device and precision
            if self.training.device == "auto":
                self.training.device = "mps" if platform.system() == "Darwin" else "cpu"
            if self.training.fp16:
                print("⚠️ mac_dev: Disabling fp16 (MPS limitation)")
                self.training.fp16 = False
        
        elif profile == ExecutionProfile.EVAL_ONLY:
            self.training.num_epochs = 0
            self.training.max_steps = 0
    
    def validate(self) -> None:
        """Validate configuration values."""
        assert self.training.batch_size >= 1
        assert self.training.gradient_accumulation_steps >= 1
        assert self.training.learning_rate > 0
        
        # Profile-specific validation
        if self.runtime.get_profile() == ExecutionProfile.MAC_DEV:
            self._validate_mac_safety()
    
    def _validate_mac_safety(self) -> None:
        """Validate mac_dev safety constraints."""
        if (self.training.max_steps is None or 
            self.training.max_steps > self.runtime.mac_dev_max_steps):
            raise ValueError(
                f"mac_dev profile: max_steps must be <= {self.runtime.mac_dev_max_steps}. "
                "Use --execution_profile colab_train for full training."
            )
    
    def get_device(self) -> str:
        """Get effective device."""
        import torch
        
        if self.training.device != "auto":
            return self.training.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        profile = self.runtime.get_profile()
        print(f"\n{'='*60}")
        print(f"📋 CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Execution Profile: {profile.value}")
        print(f"  Model: {self.model.model_name}")
        print(f"  Scene Reasoning: {'Enabled' if self.model.use_scene_reasoning else 'Disabled'}")
        print(f"  Dataset: {self.data.dataset_name}")
        print(f"  Batch Size: {self.training.batch_size} x {self.training.gradient_accumulation_steps}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Epochs: {self.training.num_epochs}")
        if self.training.max_steps:
            print(f"  Max Steps: {self.training.max_steps}")
        print(f"  Device: {self.get_device()}")
        print(f"  FP16: {self.training.fp16}")
        print(f"  Output: {self.logging.output_dir}")
        print(f"{'='*60}\n")


def detect_environment() -> str:
    """Auto-detect execution environment."""
    try:
        import google.colab
        return "colab_train"
    except ImportError:
        pass
    
    if platform.system() == "Darwin":
        return "mac_dev"
    
    return "eval_only"


def get_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="VLM VQA Research - Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    
    # Execution
    parser.add_argument("--execution_profile", type=str,
                       choices=['colab_train', 'mac_dev', 'eval_only'],
                       help="Execution profile (default: auto-detect)")
    parser.add_argument("--sync_to_drive", action='store_true',
                       help="Sync outputs to Google Drive")
    
    # Training overrides
    parser.add_argument("--smoke_test", action='store_true',
                       help="Run quick smoke test")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, help="Max training steps")
    
    # Paths
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Evaluation
    parser.add_argument("--checkpoint", type=str, help="Checkpoint for evaluation")
    
    return parser


def load_config(config_path: str, args: Optional[argparse.Namespace] = None) -> Config:
    """Load config from YAML and apply CLI overrides."""
    config = Config.from_yaml(config_path)
    
    if args is not None:
        config = config.apply_cli_overrides(args)
    
    config.validate()
    return config
