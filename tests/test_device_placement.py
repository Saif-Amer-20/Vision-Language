#!/usr/bin/env python3
"""
Test Device Placement Strategy.

Verifies that DeviceManager provides unified device placement
without conflicts from device_map="auto".
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


def test_device_detection():
    """Test automatic device detection."""
    print("\n" + "=" * 60)
    print("Testing: Device Detection")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    # Test auto detection
    dm = DeviceManager("auto")
    
    print(f"  Requested: auto")
    print(f"  Detected: {dm.device}")
    print(f"  Device type: {dm.device_info.device_type}")
    print(f"  Dtype: {dm.dtype}")
    
    # Verify detection makes sense
    if torch.cuda.is_available():
        assert dm.device_info.device_type == "cuda", "Should detect CUDA when available"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        assert dm.device_info.device_type == "mps", "Should detect MPS on Apple Silicon"
    else:
        assert dm.device_info.device_type == "cpu", "Should fall back to CPU"
    
    print("  ✓ Device detection passed")


def test_dtype_selection():
    """Test optimal dtype selection per device."""
    print("\n" + "=" * 60)
    print("Testing: Dtype Selection")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    dm = DeviceManager("auto")
    
    print(f"  Device: {dm.device_info.device_type}")
    print(f"  Selected dtype: {dm.dtype}")
    print(f"  Supports fp16: {dm.device_info.supports_fp16}")
    print(f"  Supports bf16: {dm.device_info.supports_bf16}")
    
    # Verify dtype is appropriate for device
    if dm.device_info.device_type == "cuda":
        assert dm.dtype in (torch.float16, torch.bfloat16), "CUDA should use fp16/bf16"
    else:
        assert dm.dtype == torch.float32, "MPS/CPU should use fp32"
    
    print("  ✓ Dtype selection passed")


def test_simple_model_placement():
    """Test model placement with a simple model."""
    print("\n" + "=" * 60)
    print("Testing: Simple Model Placement")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    
    print(f"  Initial device: {next(model.parameters()).device}")
    
    dm = DeviceManager("auto")
    model = dm.prepare_model(model)
    
    actual_device = next(model.parameters()).device
    print(f"  After prepare: {actual_device}")
    
    # Verify model is on correct device
    assert str(actual_device).startswith(dm.device_info.device_type), \
        f"Model should be on {dm.device_info.device_type}, got {actual_device}"
    
    print("  ✓ Simple model placement passed")


def test_batch_preparation():
    """Test batch tensor placement."""
    print("\n" + "=" * 60)
    print("Testing: Batch Preparation")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    dm = DeviceManager("auto")
    
    # Create a batch with mixed data types
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 32)),
        "attention_mask": torch.ones(2, 32),
        "pixel_values": torch.randn(2, 3, 224, 224),
        "labels": torch.randint(0, 1000, (2, 16)),
        "question_text": ["What color is the car?", "How many people?"],
    }
    
    print(f"  Before prepare:")
    print(f"    input_ids device: {batch['input_ids'].device}")
    print(f"    question_text type: {type(batch['question_text'])}")
    
    prepared = dm.prepare_batch(batch)
    
    print(f"  After prepare:")
    print(f"    input_ids device: {prepared['input_ids'].device}")
    print(f"    attention_mask device: {prepared['attention_mask'].device}")
    print(f"    pixel_values device: {prepared['pixel_values'].device}")
    print(f"    question_text preserved: {prepared['question_text'] == batch['question_text']}")
    
    # Verify all tensors moved
    for key in ["input_ids", "attention_mask", "pixel_values", "labels"]:
        assert str(prepared[key].device).startswith(dm.device_info.device_type), \
            f"{key} should be on {dm.device_info.device_type}"
    
    # Verify strings preserved
    assert prepared["question_text"] == batch["question_text"], "Strings should be unchanged"
    
    print("  ✓ Batch preparation passed")


def test_autocast_context():
    """Test mixed precision context manager."""
    print("\n" + "=" * 60)
    print("Testing: Autocast Context")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    dm = DeviceManager("auto")
    
    print(f"  AMP enabled: {dm.use_amp}")
    
    model = nn.Linear(64, 64)
    model = dm.prepare_model(model)
    
    x = torch.randn(2, 64)
    x = x.to(dm.device)
    
    # Test forward pass with autocast
    with dm.autocast():
        y = model(x)
        print(f"  Input dtype: {x.dtype}")
        print(f"  Output dtype: {y.dtype}")
    
    print("  ✓ Autocast context passed")


def test_memory_estimation():
    """Test memory usage estimation."""
    print("\n" + "=" * 60)
    print("Testing: Memory Estimation")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    dm = DeviceManager("auto")
    
    # Create a model with known parameter count
    model = nn.Sequential(
        nn.Linear(1024, 2048),  # 2M params
        nn.ReLU(),
        nn.Linear(2048, 1024),  # 2M params
    )
    
    estimate = dm.estimate_memory_usage(model, batch_size=8)
    
    print(f"  Model parameters: ~4M")
    print(f"  Estimated model memory: {estimate['model_gb']:.4f} GB")
    print(f"  Estimated gradient memory: {estimate['gradient_gb']:.4f} GB")
    print(f"  Estimated optimizer memory: {estimate['optimizer_gb']:.4f} GB")
    print(f"  Total estimate: {estimate['total_gb']:.4f} GB")
    
    # Verify estimate is reasonable (4M params * 2 bytes = ~8MB for fp16)
    assert estimate['model_gb'] > 0, "Model memory should be positive"
    assert estimate['total_gb'] > estimate['model_gb'], "Total should include gradients/optimizer"
    
    print("  ✓ Memory estimation passed")


def test_config_integration():
    """Test integration with configuration system."""
    print("\n" + "=" * 60)
    print("Testing: Config Integration")
    print("=" * 60)
    
    from src.utils.device_utils import get_device_manager
    from src.utils.config import TrainingConfig
    
    # Create a config-like object
    class MockConfig:
        def __init__(self):
            self.training = TrainingConfig()
            self.training.device = "auto"
            self.training.fp16 = torch.cuda.is_available()  # Only fp16 on CUDA
    
    config = MockConfig()
    dm = get_device_manager(config, log_info=False)
    
    print(f"  Config device: {config.training.device}")
    print(f"  Config fp16: {config.training.fp16}")
    print(f"  DeviceManager device: {dm.device}")
    print(f"  DeviceManager dtype: {dm.dtype}")
    
    # Verify config is respected
    if config.training.fp16 and torch.cuda.is_available():
        assert dm.dtype == torch.float16, "Should use fp16 when requested on CUDA"
    
    print("  ✓ Config integration passed")


def test_gradient_checkpointing_decision():
    """Test gradient checkpointing auto-decision."""
    print("\n" + "=" * 60)
    print("Testing: Gradient Checkpointing Decision")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    # Test with default threshold
    dm_default = DeviceManager("auto", memory_threshold_gb=16.0)
    should_checkpoint = dm_default.should_use_gradient_checkpointing()
    
    print(f"  Device: {dm_default.device_info.device_name}")
    print(f"  Memory: {dm_default.device_info.memory_gb} GB")
    print(f"  Threshold: 16.0 GB")
    print(f"  Should checkpoint: {should_checkpoint}")
    
    # Test with forced setting
    dm_forced = DeviceManager("auto", enable_gradient_checkpointing=True)
    assert dm_forced.should_use_gradient_checkpointing() == True, "Should respect forced True"
    
    dm_forced_off = DeviceManager("auto", enable_gradient_checkpointing=False)
    assert dm_forced_off.should_use_gradient_checkpointing() == False, "Should respect forced False"
    
    print("  ✓ Gradient checkpointing decision passed")


def test_info_dict():
    """Test device info dictionary export."""
    print("\n" + "=" * 60)
    print("Testing: Info Dict Export")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    dm = DeviceManager("auto")
    info = dm.get_info_dict()
    
    print(f"  Device info keys: {list(info.keys())}")
    
    required_keys = [
        "device", "device_type", "device_name", "dtype",
        "supports_fp16", "supports_bf16", "use_amp", "gradient_checkpointing"
    ]
    
    for key in required_keys:
        assert key in info, f"Missing required key: {key}"
        print(f"    {key}: {info[key]}")
    
    print("  ✓ Info dict export passed")


def test_no_device_map_conflict():
    """Test that model loading without device_map works."""
    print("\n" + "=" * 60)
    print("Testing: No device_map Conflict")
    print("=" * 60)
    
    from src.utils.device_utils import DeviceManager
    
    # Simulate what blip2_wrapper now does
    class MockHFModel(nn.Module):
        """Simulates a HuggingFace model loaded without device_map."""
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(256, 256)
            # Model starts on CPU (like low_cpu_mem_usage=True)
    
    model = MockHFModel()
    initial_device = next(model.parameters()).device
    print(f"  Model loaded on: {initial_device}")
    assert str(initial_device) == "cpu", "Model should load on CPU"
    
    dm = DeviceManager("auto")
    model = dm.prepare_model(model)
    
    final_device = next(model.parameters()).device
    print(f"  After DeviceManager: {final_device}")
    
    # Key test: no error was raised (device_map conflict would have thrown)
    print("  ✓ No device_map conflict (model moved successfully)")


def run_all_tests():
    """Run all device placement tests."""
    print("\n" + "=" * 60)
    print("🧪 Device Placement Tests")
    print("=" * 60)
    
    tests = [
        test_device_detection,
        test_dtype_selection,
        test_simple_model_placement,
        test_batch_preparation,
        test_autocast_context,
        test_memory_estimation,
        test_config_integration,
        test_gradient_checkpointing_decision,
        test_info_dict,
        test_no_device_map_conflict,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    if failed > 0:
        print(f"❌ {failed} tests failed")
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == "__main__":
    exit(run_all_tests())
