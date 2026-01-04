"""
Test VQATrainer Initialization.

Validates that the trainer can be initialized correctly without AttributeError,
particularly that the scheduler setup works with proper loader access.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import VQATrainer
from src.utils.config import ExperimentConfig, TrainingConfig, LoggingConfig


def create_mock_model():
    """Create a simple mock model for testing."""
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, **kwargs):
            # Mock forward that returns loss and outputs
            batch_size = kwargs.get('pixel_values', torch.tensor([1])).shape[0] if 'pixel_values' in kwargs else 1
            return {
                'loss': torch.tensor(0.5, requires_grad=True),
                'logits': torch.randn(batch_size, 10),
            }
        
        def generate(self, **kwargs):
            # Mock generate for evaluation
            batch_size = kwargs.get('pixel_values', torch.tensor([1])).shape[0] if 'pixel_values' in kwargs else 1
            return ["answer"] * batch_size
    
    return MockModel()


def create_mock_dataloader(num_samples=10, batch_size=2):
    """Create a mock dataloader for testing."""
    # Create dummy tensors
    pixel_values = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(0, 100, (num_samples, 20))
    attention_mask = torch.ones(num_samples, 20)
    labels = torch.randint(0, 100, (num_samples, 20))
    
    # Create dataset
    dataset = TensorDataset(pixel_values, input_ids, attention_mask, labels)
    
    # Create dataloader
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_mock_config():
    """Create a minimal config for testing."""
    config = ExperimentConfig()
    
    # Set minimal training config
    config.training.num_epochs = 2
    config.training.max_steps = None
    config.training.gradient_accumulation_steps = 1
    config.training.learning_rate = 1e-4
    config.training.warmup_ratio = 0.1
    config.training.lr_scheduler_type = "linear"
    config.training.batch_size = 2
    config.training.per_device_train_batch_size = 2
    config.training.weight_decay = 0.01
    config.training.max_grad_norm = 1.0
    config.training.fp16 = False
    config.training.bf16 = False
    config.training.save_strategy = "no"
    config.training.early_stopping = False
    
    # Set logging config
    config.logging.output_dir = "/tmp/test_trainer"
    config.logging.experiment_name = "test"
    config.logging.use_tensorboard = False
    config.logging.use_wandb = False
    config.logging.log_every_n_steps = 10
    
    return config


def test_trainer_initialization():
    """Test that trainer can be initialized without AttributeError."""
    print("=" * 60)
    print("Test: Trainer Initialization")
    print("=" * 60)
    
    # Create mock components
    model = create_mock_model()
    train_loader = create_mock_dataloader(num_samples=10, batch_size=2)
    val_loader = create_mock_dataloader(num_samples=6, batch_size=2)
    config = create_mock_config()
    
    # Test initialization
    try:
        trainer = VQATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            logger=None,
        )
        print("  ✓ Trainer initialized successfully")
        
        # Verify that loaders are set
        assert hasattr(trainer, 'train_loader'), "Trainer should have train_loader"
        assert hasattr(trainer, 'val_loader'), "Trainer should have val_loader"
        assert trainer.train_loader is not None, "train_loader should not be None"
        assert trainer.val_loader is not None, "val_loader should not be None"
        print("  ✓ Loaders are properly set")
        
        # Verify that scheduler is set
        assert hasattr(trainer, 'scheduler'), "Trainer should have scheduler"
        assert trainer.scheduler is not None, "Scheduler should not be None"
        print("  ✓ Scheduler is properly set")
        
        # Verify that optimizer is set
        assert hasattr(trainer, 'optimizer'), "Trainer should have optimizer"
        assert trainer.optimizer is not None, "Optimizer should not be None"
        print("  ✓ Optimizer is properly set")
        
        print("\n✅ All initialization checks passed!")
        return True
        
    except AttributeError as e:
        print(f"\n❌ AttributeError during initialization: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error during initialization: {e}")
        raise


def test_scheduler_setup():
    """Test that scheduler setup can access train_loader correctly."""
    print("\n" + "=" * 60)
    print("Test: Scheduler Setup with train_loader")
    print("=" * 60)
    
    # Create mock components
    model = create_mock_model()
    train_loader = create_mock_dataloader(num_samples=20, batch_size=4)
    val_loader = create_mock_dataloader(num_samples=8, batch_size=4)
    config = create_mock_config()
    config.training.max_steps = None  # Use epochs to calculate steps
    
    try:
        trainer = VQATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            logger=None,
        )
        
        # Calculate expected total steps
        steps_per_epoch = len(train_loader) // config.training.gradient_accumulation_steps
        expected_total_steps = steps_per_epoch * config.training.num_epochs
        
        print(f"  ✓ Train loader length: {len(train_loader)}")
        print(f"  ✓ Steps per epoch: {steps_per_epoch}")
        print(f"  ✓ Expected total steps: {expected_total_steps}")
        print(f"  ✓ Scheduler initialized successfully")
        
        print("\n✅ Scheduler setup test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during scheduler setup: {e}")
        raise


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VQATrainer Initialization Tests")
    print("=" * 60 + "\n")
    
    try:
        test_trainer_initialization()
        test_scheduler_setup()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
