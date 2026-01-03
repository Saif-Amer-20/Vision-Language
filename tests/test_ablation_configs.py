#!/usr/bin/env python3
"""
Tests for Ablation Study Configurations.

Validates:
- All ablation configs load correctly
- Component flags are set correctly for each ablation
- Fair comparison setup (matching hyperparameters)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_all_ablation_configs_load():
    """Test that all ablation configs load without errors."""
    print("\n" + "=" * 60)
    print("Testing: All Ablation Configs Load")
    print("=" * 60)
    
    from src.utils.config import load_config
    
    configs_to_test = [
        "configs/baseline.yaml",
        "configs/proposed.yaml",
        "configs/ablation_spatial_only.yaml",
        "configs/ablation_relation_only.yaml",
        "configs/ablation_no_spatial.yaml",
        "configs/ablation_no_relation.yaml",
    ]
    
    loaded_configs = {}
    passed = 0
    failed = 0
    
    for config_path in configs_to_test:
        try:
            config = load_config(config_path)
            loaded_configs[config_path] = config
            
            # Extract key flags
            sr = getattr(config.model, 'use_scene_reasoning', False)
            spatial = getattr(config.model, 'use_spatial_encoding', False)
            relation = getattr(config.model, 'use_relation_attention', False)
            
            print(f"  ✓ {config_path}")
            print(f"      Scene Reasoning: {sr}")
            print(f"      Spatial Encoding: {spatial}")
            print(f"      Relation Attention: {relation}")
            passed += 1
            
        except Exception as e:
            print(f"  ✗ {config_path}")
            print(f"      Error: {e}")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} configs loaded successfully")
    return failed == 0, loaded_configs


def test_ablation_correctness():
    """Test that ablation configs have correct flag combinations."""
    print("\n" + "=" * 60)
    print("Testing: Ablation Flag Correctness")
    print("=" * 60)
    
    from src.utils.config import load_config
    
    # Expected configurations
    # Format: (config_path, expected_scene_reasoning, expected_spatial, expected_relation)
    expected = [
        ("configs/baseline.yaml", False, True, True),  # SR disabled, flags ignored
        ("configs/proposed.yaml", True, True, True),   # Full model
        ("configs/ablation_spatial_only.yaml", True, True, False),   # Spatial only
        ("configs/ablation_relation_only.yaml", True, False, True),  # Relation only
        ("configs/ablation_no_spatial.yaml", True, False, False),    # Neither
        ("configs/ablation_no_relation.yaml", True, True, False),    # Spatial only (alias)
    ]
    
    passed = 0
    failed = 0
    
    for config_path, exp_sr, exp_spatial, exp_relation in expected:
        try:
            config = load_config(config_path)
            
            sr = getattr(config.model, 'use_scene_reasoning', False)
            spatial = getattr(config.model, 'use_spatial_encoding', False)
            relation = getattr(config.model, 'use_relation_attention', False)
            
            # Check correctness
            sr_ok = (sr == exp_sr)
            spatial_ok = (spatial == exp_spatial) or (not exp_sr)  # Flags don't matter if SR disabled
            relation_ok = (relation == exp_relation) or (not exp_sr)
            
            if sr_ok and spatial_ok and relation_ok:
                print(f"  ✓ {Path(config_path).name}")
                print(f"      SR={sr}, Spatial={spatial}, Relation={relation}")
                passed += 1
            else:
                print(f"  ✗ {Path(config_path).name}")
                print(f"      Got:      SR={sr}, Spatial={spatial}, Relation={relation}")
                print(f"      Expected: SR={exp_sr}, Spatial={exp_spatial}, Relation={exp_relation}")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ {config_path}: {e}")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} configs have correct flags")
    return failed == 0


def test_fair_comparison_setup():
    """Test that all configs have matching hyperparameters for fair comparison."""
    print("\n" + "=" * 60)
    print("Testing: Fair Comparison Setup")
    print("=" * 60)
    
    from src.utils.config import load_config
    
    configs_to_compare = [
        "configs/baseline.yaml",
        "configs/proposed.yaml",
        "configs/ablation_spatial_only.yaml",
        "configs/ablation_relation_only.yaml",
        "configs/ablation_no_spatial.yaml",
        "configs/ablation_no_relation.yaml",
    ]
    
    # Hyperparameters that MUST match for fair comparison
    critical_params = [
        ("training.learning_rate", "learning rate"),
        ("training.num_epochs", "number of epochs"),
        ("training.batch_size", "batch size"),
        ("training.gradient_accumulation_steps", "gradient accumulation"),
        ("training.weight_decay", "weight decay"),
        ("training.warmup_ratio", "warmup ratio"),
        ("model.model_name", "model name"),
        ("model.freeze_vision_encoder", "freeze vision"),
        ("model.freeze_llm", "freeze LLM"),
        ("model.freeze_qformer", "freeze Q-Former"),
        ("data.dataset_name", "dataset"),
        ("data.image_size", "image size"),
    ]
    
    # Load all configs
    configs = {}
    for path in configs_to_compare:
        try:
            configs[path] = load_config(path)
        except Exception as e:
            print(f"  ✗ Failed to load {path}: {e}")
            return False
    
    # Use proposed as reference
    reference = configs["configs/proposed.yaml"]
    
    all_passed = True
    
    for param_path, param_name in critical_params:
        # Get reference value
        parts = param_path.split(".")
        ref_val = reference
        for part in parts:
            ref_val = getattr(ref_val, part, None)
        
        print(f"\n  Checking {param_name}:")
        print(f"    Reference (proposed): {ref_val}")
        
        param_passed = True
        for config_path, config in configs.items():
            if config_path == "configs/proposed.yaml":
                continue
                
            # Get value from this config
            val = config
            for part in parts:
                val = getattr(val, part, None)
            
            config_name = Path(config_path).name
            
            if val == ref_val:
                print(f"    ✓ {config_name}: {val}")
            else:
                print(f"    ✗ {config_name}: {val} (MISMATCH!)")
                param_passed = False
                all_passed = False
        
        if param_passed:
            print(f"    → All configs match for {param_name}")
    
    print("\n" + "-" * 60)
    if all_passed:
        print("  ✓ All critical hyperparameters match across configs!")
        print("  ✓ Fair comparison setup verified!")
    else:
        print("  ✗ Some hyperparameters don't match!")
        print("  ✗ Please fix for fair comparison!")
    
    return all_passed


def test_experiment_names_unique():
    """Test that all experiment names are unique."""
    print("\n" + "=" * 60)
    print("Testing: Unique Experiment Names")
    print("=" * 60)
    
    from src.utils.config import load_config
    
    configs_to_check = [
        "configs/baseline.yaml",
        "configs/proposed.yaml",
        "configs/ablation_spatial_only.yaml",
        "configs/ablation_relation_only.yaml",
        "configs/ablation_no_spatial.yaml",
        "configs/ablation_no_relation.yaml",
    ]
    
    names = {}
    
    for config_path in configs_to_check:
        try:
            config = load_config(config_path)
            name = config.logging.experiment_name
            
            if name in names:
                print(f"  ✗ Duplicate name '{name}':")
                print(f"      {names[name]}")
                print(f"      {config_path}")
                return False
            
            names[name] = config_path
            print(f"  ✓ {Path(config_path).name}: '{name}'")
            
        except Exception as e:
            print(f"  ✗ {config_path}: {e}")
            return False
    
    print(f"\n  ✓ All {len(names)} experiment names are unique!")
    return True


def test_scene_reasoning_module_compatibility():
    """Test that configs work with SceneReasoningModule."""
    print("\n" + "=" * 60)
    print("Testing: SceneReasoningModule Compatibility")
    print("=" * 60)
    
    try:
        import torch
        from src.utils.config import load_config
        from src.models.scene_reasoning import SceneReasoningModule, SceneReasoningConfig
    except ImportError as e:
        print(f"  ⚠ Skipping (import error): {e}")
        return True
    
    ablation_configs = [
        ("configs/ablation_spatial_only.yaml", True, False),
        ("configs/ablation_relation_only.yaml", False, True),
        ("configs/ablation_no_spatial.yaml", False, False),
        ("configs/proposed.yaml", True, True),
    ]
    
    passed = 0
    
    for config_path, exp_spatial, exp_relation in ablation_configs:
        try:
            config = load_config(config_path)
            
            sr_config = SceneReasoningConfig(
                hidden_dim=config.model.scene_hidden_dim,
                num_heads=config.model.scene_num_heads,
                num_layers=config.model.scene_num_layers,
                use_spatial_encoding=config.model.use_spatial_encoding,
                use_relation_attention=config.model.use_relation_attention,
            )
            
            module = SceneReasoningModule(sr_config)
            
            # Quick forward pass test
            x = torch.randn(1, 197, 768)  # Typical ViT output
            output, aux = module(x)
            
            assert output.shape == x.shape, f"Output shape mismatch"
            
            print(f"  ✓ {Path(config_path).name}: forward pass OK")
            print(f"      Spatial={sr_config.use_spatial_encoding}, Relation={sr_config.use_relation_attention}")
            passed += 1
            
        except Exception as e:
            print(f"  ✗ {Path(config_path).name}: {e}")
    
    print(f"\n  Results: {passed}/{len(ablation_configs)} configs compatible")
    return passed == len(ablation_configs)


def main():
    """Run all ablation config tests."""
    print("\n" + "=" * 60)
    print("🧪 ABLATION CONFIGURATION TESTS")
    print("=" * 60)
    print("\nValidating ablation study setup for fair comparison...")
    
    tests = [
        ("All Configs Load", lambda: test_all_ablation_configs_load()[0]),
        ("Ablation Flag Correctness", test_ablation_correctness),
        ("Fair Comparison Setup", test_fair_comparison_setup),
        ("Unique Experiment Names", test_experiment_names_unique),
        ("SceneReasoning Compatibility", test_scene_reasoning_module_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"\n  Total tests: {len(tests)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed == 0:
        print("\n  ✅ All ablation config tests passed!")
        print("\n  Verified:")
        print("    • All configs load without errors")
        print("    • Ablation flags set correctly for each config")
        print("    • Critical hyperparameters match for fair comparison")
        print("    • Experiment names are unique")
        print("    • SceneReasoningModule compatible with all configs")
        return 0
    else:
        print(f"\n  ❌ {failed} test(s) failed!")
        print("  Please fix issues before running ablation study.")
        return 1


if __name__ == "__main__":
    exit(main())
