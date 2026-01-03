"""
Test Configuration System.

Validates that:
1. All YAML config files load successfully
2. Field names match between YAML and dataclasses
3. ExperimentConfig structure is correct
4. Override system works correctly
5. Profile constraints are applied
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    ExperimentConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    LoggingConfig,
    EvaluationConfig,
    RuntimeConfig,
    ExecutionProfile,
    load_config,
    detect_execution_profile,
    parse_cli_overrides,
)


def test_dataclass_creation():
    """Test that all dataclasses can be created with defaults."""
    print("=" * 60)
    print("Test 1: Dataclass Creation")
    print("=" * 60)
    
    # Test individual configs
    data = DataConfig()
    print(f"  ✓ DataConfig created: dataset={data.dataset_name}")
    
    model = ModelConfig()
    print(f"  ✓ ModelConfig created: model={model.model_name}")
    
    training = TrainingConfig()
    print(f"  ✓ TrainingConfig created: lr={training.learning_rate}")
    
    logging = LoggingConfig()
    print(f"  ✓ LoggingConfig created: output={logging.output_dir}")
    
    evaluation = EvaluationConfig()
    print(f"  ✓ EvaluationConfig created: exact_match={evaluation.compute_exact_match}")
    
    runtime = RuntimeConfig()
    print(f"  ✓ RuntimeConfig created: profile={runtime.execution_profile}")
    
    # Test main config
    config = ExperimentConfig()
    print(f"  ✓ ExperimentConfig created: seed={config.seed}")
    
    # Test alias
    config2 = Config()
    print(f"  ✓ Config alias works: seed={config2.seed}")
    
    print("\n✅ All dataclasses created successfully\n")


def test_yaml_loading():
    """Test that all YAML config files load successfully."""
    print("=" * 60)
    print("Test 2: YAML Config Loading")
    print("=" * 60)
    
    config_dir = PROJECT_ROOT / "configs"
    yaml_files = list(config_dir.glob("*.yaml"))
    
    print(f"  Found {len(yaml_files)} YAML files in configs/\n")
    
    for yaml_file in yaml_files:
        try:
            config = load_config(
                str(yaml_file),
                apply_constraints=False,  # Don't apply constraints during test
                validate=False,  # Don't validate during load test
            )
            print(f"  ✓ {yaml_file.name}")
            print(f"      seed: {config.seed}")
            print(f"      model: {config.model.model_name}")
            print(f"      scene_reasoning: {config.model.use_scene_reasoning}")
            print(f"      profile: {config.runtime.execution_profile}")
        except Exception as e:
            print(f"  ✗ {yaml_file.name} - FAILED: {e}")
            raise
    
    print(f"\n✅ All {len(yaml_files)} YAML files loaded successfully\n")


def test_field_matching():
    """Test that config field names match YAML keys."""
    print("=" * 60)
    print("Test 3: Field Name Matching")
    print("=" * 60)
    
    config = load_config(
        str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        apply_constraints=False,
        validate=False,
    )
    
    # Check critical fields exist and are correct type
    checks = [
        ("seed", int),
        ("data.dataset_name", str),
        ("data.train_split", str),
        ("data.val_split", str),
        ("model.model_name", str),
        ("model.use_scene_reasoning", bool),
        ("model.scene_hidden_dim", int),
        ("model.scene_num_heads", int),
        ("model.scene_num_layers", int),
        ("model.scene_mlp_ratio", float),
        ("model.scene_dropout", float),
        ("model.use_spatial_encoding", bool),
        ("model.use_relation_attention", bool),
        ("model.spatial_encoding_dim", int),
        ("training.batch_size", int),
        ("training.learning_rate", float),
        ("training.num_epochs", int),
        ("logging.output_dir", str),
        ("logging.experiment_name", str),
        ("evaluation.compute_exact_match", bool),
        ("evaluation.compute_normalized_match", bool),
        ("evaluation.compute_vqa_accuracy", bool),
        ("evaluation.save_error_analysis", bool),
        ("evaluation.output_csv", bool),
        ("evaluation.output_json", bool),
        ("runtime.execution_profile", str),
        ("runtime.mac_dev_max_steps", int),
    ]
    
    for field_path, expected_type in checks:
        parts = field_path.split(".")
        value = config
        for part in parts:
            value = getattr(value, part)
        
        assert isinstance(value, expected_type), f"{field_path} should be {expected_type}, got {type(value)}"
        print(f"  ✓ {field_path}: {value} ({expected_type.__name__})")
    
    print(f"\n✅ All {len(checks)} field checks passed\n")


def test_cli_overrides():
    """Test CLI override parsing and application."""
    print("=" * 60)
    print("Test 4: CLI Overrides")
    print("=" * 60)
    
    # Test parse_cli_overrides
    args = [
        "--training.learning_rate", "2e-5",
        "--model.use_scene_reasoning", "true",
        "--training.num_epochs", "5",
    ]
    
    overrides = parse_cli_overrides(args)
    print(f"  Parsed overrides: {overrides}")
    
    assert overrides["training.learning_rate"] == 2e-5
    assert overrides["model.use_scene_reasoning"] == True
    assert overrides["training.num_epochs"] == 5
    print("  ✓ CLI parsing works correctly")
    
    # Test applying overrides
    config = load_config(
        str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        overrides=overrides,
        apply_constraints=False,
        validate=False,
    )
    
    assert config.training.learning_rate == 2e-5
    assert config.model.use_scene_reasoning == True
    assert config.training.num_epochs == 5
    print("  ✓ Overrides applied correctly")
    
    print(f"      learning_rate: {config.training.learning_rate}")
    print(f"      use_scene_reasoning: {config.model.use_scene_reasoning}")
    print(f"      num_epochs: {config.training.num_epochs}")
    
    print("\n✅ CLI override system works correctly\n")


def test_profile_constraints():
    """Test execution profile constraints."""
    print("=" * 60)
    print("Test 5: Profile Constraints")
    print("=" * 60)
    
    # Test mac_dev constraints
    config = load_config(
        str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        execution_profile="mac_dev",
        apply_constraints=True,
        validate=False,
    )
    
    max_steps = config.runtime.mac_dev_max_steps
    max_samples = config.runtime.mac_dev_max_samples
    
    assert config.training.max_steps == max_steps
    assert config.data.max_train_samples == max_samples
    assert config.training.fp16 == False  # Disabled for MPS
    
    print(f"  ✓ mac_dev profile applied:")
    print(f"      max_steps: {config.training.max_steps}")
    print(f"      max_train_samples: {config.data.max_train_samples}")
    print(f"      fp16: {config.training.fp16}")
    
    # Test eval_only constraints
    config2 = load_config(
        str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        execution_profile="eval_only",
        apply_constraints=True,
        validate=False,
    )
    
    assert config2.training.num_epochs == 0
    assert config2.training.max_steps == 0
    print(f"  ✓ eval_only profile applied:")
    print(f"      num_epochs: {config2.training.num_epochs}")
    print(f"      max_steps: {config2.training.max_steps}")
    
    print("\n✅ Profile constraints work correctly\n")


def test_smoke_test_mode():
    """Test smoke test mode."""
    print("=" * 60)
    print("Test 6: Smoke Test Mode")
    print("=" * 60)
    
    config = load_config(
        str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        smoke_test=True,
        apply_constraints=False,
        validate=False,
    )
    
    assert config.training.smoke_test == True
    assert config.training.max_steps == config.training.smoke_test_steps
    assert config.data.max_train_samples == config.training.smoke_test_samples
    
    print(f"  ✓ Smoke test settings applied:")
    print(f"      smoke_test: {config.training.smoke_test}")
    print(f"      max_steps: {config.training.max_steps}")
    print(f"      max_train_samples: {config.data.max_train_samples}")
    
    print("\n✅ Smoke test mode works correctly\n")


def test_to_dict_and_save():
    """Test config serialization."""
    print("=" * 60)
    print("Test 7: Serialization")
    print("=" * 60)
    
    config = ExperimentConfig()
    
    # Test to_dict
    config_dict = config.to_dict()
    
    assert "seed" in config_dict
    assert "data" in config_dict
    assert "model" in config_dict
    assert "training" in config_dict
    assert "logging" in config_dict
    assert "evaluation" in config_dict
    assert "runtime" in config_dict
    
    print(f"  ✓ to_dict() returns all sections")
    print(f"      Keys: {list(config_dict.keys())}")
    
    # Test from_dict roundtrip
    config2 = ExperimentConfig.from_dict(config_dict)
    assert config2.seed == config.seed
    assert config2.model.model_name == config.model.model_name
    
    print(f"  ✓ from_dict() roundtrip works")
    
    print("\n✅ Serialization works correctly\n")


def test_baseline_vs_proposed():
    """Compare baseline and proposed configs."""
    print("=" * 60)
    print("Test 8: Baseline vs Proposed Comparison")
    print("=" * 60)
    
    baseline = load_config(
        str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        apply_constraints=False,
        validate=False,
    )
    
    proposed = load_config(
        str(PROJECT_ROOT / "configs" / "proposed.yaml"),
        apply_constraints=False,
        validate=False,
    )
    
    # Key difference should be use_scene_reasoning
    assert baseline.model.use_scene_reasoning == False
    assert proposed.model.use_scene_reasoning == True
    
    print(f"  ✓ Baseline: use_scene_reasoning={baseline.model.use_scene_reasoning}")
    print(f"  ✓ Proposed: use_scene_reasoning={proposed.model.use_scene_reasoning}")
    
    # Other fields should be the same
    assert baseline.model.scene_hidden_dim == proposed.model.scene_hidden_dim
    assert baseline.training.learning_rate == proposed.training.learning_rate
    
    print(f"  ✓ Scene config matches:")
    print(f"      scene_hidden_dim: {baseline.model.scene_hidden_dim}")
    print(f"      scene_num_heads: {baseline.model.scene_num_heads}")
    print(f"      scene_num_layers: {baseline.model.scene_num_layers}")
    
    print("\n✅ Baseline/Proposed comparison correct\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 CONFIGURATION SYSTEM TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_dataclass_creation()
        test_yaml_loading()
        test_field_matching()
        test_cli_overrides()
        test_profile_constraints()
        test_smoke_test_mode()
        test_to_dict_and_save()
        test_baseline_vs_proposed()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
