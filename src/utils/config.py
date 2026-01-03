"""
Configuration System for VLM-VQA Research Project.

Provides YAML-based configuration with:
- Type-safe dataclass configuration objects
- CLI argument overrides with nested key support
- Execution profiles (colab_train, mac_dev, eval_only)
- Environment auto-detection
- Safety guards for local development

IMPORTANT: All field names EXACTLY match YAML keys in configs/*.yaml

Usage:
    from src.utils.config import load_config, ExperimentConfig
    
    # Load from YAML
    config = load_config("configs/baseline.yaml")
    
    # Load with CLI overrides
    config = load_config("configs/baseline.yaml", overrides={
        "training.learning_rate": 2e-5,
        "model.use_scene_reasoning": True,
    })
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Dict, Any, List, Union, Type, TypeVar
from enum import Enum
import yaml
import os
import sys
import platform
import logging
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# ENUMS
# =============================================================================

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
        s_lower = s.lower()
        if s_lower not in mapping:
            raise ValueError(f"Unknown profile: {s}. Valid: {list(mapping.keys())}")
        return mapping[s_lower]


# =============================================================================
# DATA CONFIG
# =============================================================================

@dataclass
class DataConfig:
    """
    Dataset configuration.
    
    All field names match YAML keys under 'data:' section.
    """
    # Dataset source
    dataset_name: str = "HuggingFaceM4/VQAv2"
    
    # Splits - matches YAML: train_split, val_split
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    
    # Sample limits
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
    prefetch_factor: int = 2
    
    # Prompt template for generative VQA
    prompt_template: str = "Question: {question} Answer:"
    
    # Cache directory
    cache_dir: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataConfig':
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# MODEL CONFIG
# =============================================================================

@dataclass 
class ModelConfig:
    """
    Model configuration.
    
    All field names match YAML keys under 'model:' section.
    Includes all scene reasoning fields.
    """
    # Base model
    model_name: str = "Salesforce/blip2-opt-2.7b"
    torch_dtype: str = "float16"  # float16, bfloat16, float32
    
    # Component freezing - matches YAML exactly
    freeze_vision_encoder: bool = True
    freeze_llm: bool = True
    freeze_qformer: bool = False
    
    # Scene Reasoning Module - matches YAML exactly
    use_scene_reasoning: bool = False
    scene_hidden_dim: int = 768
    scene_num_heads: int = 8
    scene_num_layers: int = 2
    scene_mlp_ratio: float = 4.0
    scene_dropout: float = 0.1
    
    # Ablation controls - matches YAML exactly
    use_spatial_encoding: bool = True
    use_relation_attention: bool = True
    spatial_encoding_dim: int = 64
    
    # Generation settings
    max_new_tokens: int = 16
    num_beams: int = 3
    do_sample: bool = False
    temperature: float = 1.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration.
    
    All field names match YAML keys under 'training:' section.
    """
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
    
    # Evaluation during training
    eval_strategy: str = "epoch"  # epoch, steps
    eval_steps: int = 500
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3
    
    # Smoke test settings
    smoke_test: bool = False
    smoke_test_samples: int = 32
    smoke_test_steps: int = 5
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# LOGGING CONFIG
# =============================================================================

@dataclass
class LoggingConfig:
    """
    Logging configuration.
    
    All field names match YAML keys under 'logging:' section.
    """
    # Output paths
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
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LoggingConfig':
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# EVALUATION CONFIG (NEW)
# =============================================================================

@dataclass
class EvaluationConfig:
    """
    Evaluation configuration.
    
    All field names match YAML keys under 'evaluation:' section.
    """
    # Metrics to compute
    compute_exact_match: bool = True
    compute_normalized_match: bool = True
    compute_vqa_accuracy: bool = True
    
    # Error analysis
    save_error_analysis: bool = True
    error_analysis_samples: int = 500
    
    # Output formats
    output_csv: bool = True
    output_json: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvaluationConfig':
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# RUNTIME CONFIG
# =============================================================================

@dataclass
class RuntimeConfig:
    """
    Runtime configuration for execution environment.
    
    All field names match YAML keys under 'runtime:' section.
    """
    # Execution profile
    execution_profile: str = "colab_train"
    
    # Google Drive sync (Colab)
    sync_to_drive: bool = False
    drive_output_path: str = "/content/drive/MyDrive/VLM_Thesis_Outputs"
    
    # Safety limits for mac_dev profile
    mac_dev_max_steps: int = 10
    mac_dev_max_samples: int = 50
    mac_dev_save_checkpoints: bool = False
    
    def get_profile(self) -> ExecutionProfile:
        """Get execution profile as enum."""
        return ExecutionProfile.from_string(self.execution_profile)
    
    def is_full_training_allowed(self) -> bool:
        """Check if full training is allowed."""
        return self.get_profile() == ExecutionProfile.COLAB_TRAIN
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RuntimeConfig':
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# EXPERIMENT CONFIG (Main class)
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Main experiment configuration combining all sub-configs.
    
    This is the primary configuration class used throughout the codebase.
    Structure matches YAML config files exactly:
    
        seed: 42
        data:
            dataset_name: ...
        model:
            model_name: ...
        training:
            batch_size: ...
        logging:
            output_dir: ...
        evaluation:
            compute_exact_match: ...
        runtime:
            execution_profile: ...
    """
    # Global seed (top-level in YAML)
    seed: int = 42
    
    # Sub-configurations (match YAML sections)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict or {})
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            seed=d.get('seed', 42),
            data=DataConfig.from_dict(d.get('data', {})),
            model=ModelConfig.from_dict(d.get('model', {})),
            training=TrainingConfig.from_dict(d.get('training', {})),
            logging=LoggingConfig.from_dict(d.get('logging', {})),
            evaluation=EvaluationConfig.from_dict(d.get('evaluation', {})),
            runtime=RuntimeConfig.from_dict(d.get('runtime', {})),
        )
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Apply nested overrides to config.
        
        Supports dot notation for nested keys:
            {"training.learning_rate": 2e-5}
            {"model.use_scene_reasoning": True}
        
        Args:
            overrides: Dictionary of overrides with dot-notation keys
            
        Returns:
            Self for chaining
        """
        for key, value in overrides.items():
            self._set_nested(key, value)
        return self
    
    def _set_nested(self, key: str, value: Any) -> None:
        """Set a nested config value using dot notation."""
        parts = key.split('.')
        
        if len(parts) == 1:
            # Top-level key (e.g., "seed")
            if hasattr(self, key):
                setattr(self, key, self._convert_type(getattr(self, key), value))
            else:
                logger.warning(f"Unknown config key: {key}")
        elif len(parts) == 2:
            # Nested key (e.g., "training.learning_rate")
            section, field_name = parts
            if hasattr(self, section):
                sub_config = getattr(self, section)
                if hasattr(sub_config, field_name):
                    current_value = getattr(sub_config, field_name)
                    setattr(sub_config, field_name, self._convert_type(current_value, value))
                else:
                    logger.warning(f"Unknown field: {section}.{field_name}")
            else:
                logger.warning(f"Unknown section: {section}")
        else:
            logger.warning(f"Deep nesting not supported: {key}")
    
    def _convert_type(self, current: Any, new: Any) -> Any:
        """Convert new value to match current value's type."""
        if current is None:
            return new
        
        current_type = type(current)
        
        if current_type == bool and isinstance(new, str):
            return new.lower() in ('true', '1', 'yes')
        
        try:
            return current_type(new)
        except (ValueError, TypeError):
            return new
    
    def apply_profile_constraints(self) -> 'ExperimentConfig':
        """Apply constraints based on execution profile."""
        profile = self.runtime.get_profile()
        
        if profile == ExecutionProfile.MAC_DEV:
            # Safety limits
            max_steps = self.runtime.mac_dev_max_steps
            max_samples = self.runtime.mac_dev_max_samples
            
            if self.training.max_steps is None or self.training.max_steps > max_steps:
                logger.info(f"mac_dev: Limiting max_steps to {max_steps}")
                self.training.max_steps = max_steps
            
            if self.data.max_train_samples is None or self.data.max_train_samples > max_samples:
                logger.info(f"mac_dev: Limiting train samples to {max_samples}")
                self.data.max_train_samples = max_samples
                self.data.max_val_samples = max_samples // 2
            
            # Device and precision
            if self.training.device == "auto":
                self.training.device = "mps" if platform.system() == "Darwin" else "cpu"
            
            if self.training.fp16:
                logger.info("mac_dev: Disabling fp16 (MPS limitation)")
                self.training.fp16 = False
            
            # Don't save checkpoints
            if not self.runtime.mac_dev_save_checkpoints:
                self.training.save_total_limit = 0
        
        elif profile == ExecutionProfile.EVAL_ONLY:
            self.training.num_epochs = 0
            self.training.max_steps = 0
        
        return self
    
    def apply_smoke_test(self) -> 'ExperimentConfig':
        """Apply smoke test settings."""
        self.training.smoke_test = True
        self.training.max_steps = self.training.smoke_test_steps
        self.data.max_train_samples = self.training.smoke_test_samples
        self.data.max_val_samples = self.training.smoke_test_samples // 2
        return self
    
    def validate(self) -> 'ExperimentConfig':
        """Validate configuration values."""
        # Training validations
        assert self.training.batch_size >= 1, "batch_size must be >= 1"
        assert self.training.gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be >= 1"
        assert self.training.learning_rate > 0, "learning_rate must be > 0"
        assert self.training.num_epochs >= 0, "num_epochs must be >= 0"
        assert self.training.warmup_ratio >= 0 and self.training.warmup_ratio <= 1, "warmup_ratio must be in [0, 1]"
        
        # Model validations
        assert self.model.scene_hidden_dim > 0, "scene_hidden_dim must be > 0"
        assert self.model.scene_num_heads > 0, "scene_num_heads must be > 0"
        assert self.model.scene_num_layers >= 0, "scene_num_layers must be >= 0"
        
        # Profile-specific validation
        if self.runtime.get_profile() == ExecutionProfile.MAC_DEV:
            self._validate_mac_safety()
        
        return self
    
    def _validate_mac_safety(self) -> None:
        """Validate mac_dev safety constraints."""
        max_steps = self.runtime.mac_dev_max_steps
        
        if self.training.max_steps is None or self.training.max_steps > max_steps:
            raise ValueError(
                f"mac_dev profile: max_steps must be <= {max_steps}. "
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


# Alias for backward compatibility
Config = ExperimentConfig


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def detect_environment() -> str:
    """
    Auto-detect execution environment.
    
    Returns:
        Execution profile string: 'colab_train', 'mac_dev', or 'eval_only'
    """
    try:
        import google.colab
        return "colab_train"
    except ImportError:
        pass
    
    if platform.system() == "Darwin":
        return "mac_dev"
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "colab_train"
    except ImportError:
        pass
    
    return "eval_only"


def detect_execution_profile() -> ExecutionProfile:
    """Auto-detect execution environment as enum."""
    return ExecutionProfile.from_string(detect_environment())


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    execution_profile: Optional[str] = None,
    smoke_test: bool = False,
    apply_constraints: bool = True,
    validate: bool = True,
) -> ExperimentConfig:
    """
    Load configuration from YAML with optional overrides.
    
    This is the main entry point for loading configurations.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of overrides with dot-notation keys
                   e.g., {"training.learning_rate": 2e-5}
        execution_profile: Override execution profile
        smoke_test: Enable smoke test mode
        apply_constraints: Apply profile-based constraints
        validate: Validate configuration
    
    Returns:
        Loaded and configured ExperimentConfig
    
    Example:
        config = load_config(
            "configs/baseline.yaml",
            overrides={"training.learning_rate": 2e-5},
            smoke_test=True,
        )
    """
    # Load from YAML
    config = ExperimentConfig.from_yaml(config_path)
    
    # Apply CLI overrides
    if overrides:
        config.apply_overrides(overrides)
    
    # Override execution profile
    if execution_profile:
        config.runtime.execution_profile = execution_profile
    
    # Apply smoke test
    if smoke_test:
        config.apply_smoke_test()
    
    # Apply profile constraints
    if apply_constraints:
        config.apply_profile_constraints()
    
    # Validate
    if validate:
        config.validate()
    
    return config


def parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    Parse CLI arguments into override dictionary.
    
    Handles arguments like:
        --training.learning_rate 2e-5
        --model.use_scene_reasoning true
    
    Args:
        args: List of command line arguments
    
    Returns:
        Dictionary of overrides
    """
    overrides = {}
    i = 0
    
    while i < len(args):
        arg = args[i]
        
        if arg.startswith('--'):
            key = arg[2:]  # Remove --
            
            # Check if next arg is a value
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                value = args[i + 1]
                
                # Try to parse as number or bool
                value = _parse_value(value)
                
                overrides[key] = value
                i += 2
            else:
                # Flag without value = True
                overrides[key] = True
                i += 1
        else:
            i += 1
    
    return overrides


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate type."""
    # Boolean
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False
    
    # None
    if value.lower() in ('null', 'none'):
        return None
    
    # Number
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # String
    return value


# =============================================================================
# ARGPARSE INTEGRATION
# =============================================================================

def get_argument_parser() -> 'argparse.ArgumentParser':
    """Create standard CLI argument parser."""
    import argparse
    
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


def config_from_args(args: 'argparse.Namespace', extra_overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """
    Create configuration from parsed argparse namespace.
    
    Args:
        args: Parsed argument namespace
        extra_overrides: Additional overrides to apply
    
    Returns:
        ExperimentConfig
    """
    # Build overrides from args
    overrides = {}
    
    if hasattr(args, 'batch_size') and args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr:
        overrides['training.learning_rate'] = args.lr
    if hasattr(args, 'epochs') and args.epochs:
        overrides['training.num_epochs'] = args.epochs
    if hasattr(args, 'max_steps') and args.max_steps:
        overrides['training.max_steps'] = args.max_steps
    if hasattr(args, 'output_dir') and args.output_dir:
        overrides['logging.output_dir'] = args.output_dir
    if hasattr(args, 'experiment_name') and args.experiment_name:
        overrides['logging.experiment_name'] = args.experiment_name
    if hasattr(args, 'resume') and args.resume:
        overrides['training.resume_from_checkpoint'] = args.resume
    if hasattr(args, 'sync_to_drive') and args.sync_to_drive:
        overrides['runtime.sync_to_drive'] = True
    
    # Merge with extra overrides
    if extra_overrides:
        overrides.update(extra_overrides)
    
    # Determine profile
    profile = None
    if hasattr(args, 'execution_profile') and args.execution_profile:
        profile = args.execution_profile
    
    # Determine smoke test
    smoke_test = hasattr(args, 'smoke_test') and args.smoke_test
    
    return load_config(
        config_path=args.config,
        overrides=overrides,
        execution_profile=profile,
        smoke_test=smoke_test,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_config() -> ExperimentConfig:
    """Create a default configuration."""
    return ExperimentConfig()


def merge_configs(base: ExperimentConfig, override: ExperimentConfig) -> ExperimentConfig:
    """Merge two configurations, with override taking precedence."""
    base_dict = base.to_dict()
    override_dict = override.to_dict()
    
    def deep_merge(a: Dict, b: Dict) -> Dict:
        result = a.copy()
        for key, value in b.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged = deep_merge(base_dict, override_dict)
    return ExperimentConfig.from_dict(merged)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main config class
    'ExperimentConfig',
    'Config',  # Alias
    
    # Sub-config classes
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'LoggingConfig',
    'EvaluationConfig',
    'RuntimeConfig',
    
    # Enum
    'ExecutionProfile',
    
    # Functions
    'load_config',
    'detect_environment',
    'detect_execution_profile',
    'get_argument_parser',
    'config_from_args',
    'parse_cli_overrides',
    'create_default_config',
    'merge_configs',
]
