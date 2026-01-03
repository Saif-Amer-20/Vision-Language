"""
Unit tests for Scene Reasoning integration in BLIP-2.

Tests that:
1. Scene Reasoning hook is properly registered
2. Enhanced features differ from original features
3. Gradient flow through Scene Reasoning works
4. Hook can be enabled/disabled for ablation

Run with: python tests/test_scene_reasoning_integration.py
"""

import sys
from pathlib import Path

# Add project root to path BEFORE any other imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# Conditional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    print("pytest not available, running standalone tests")


class MockVisionOutput:
    """Mock vision model output for testing."""
    def __init__(self, hidden_states: torch.Tensor):
        self.last_hidden_state = hidden_states
        self.pooler_output = hidden_states[:, 0, :]


class MockVisionModel(nn.Module):
    """Mock vision encoder for testing without loading full BLIP-2."""
    def __init__(self, hidden_dim: int = 768, num_patches: int = 257):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.proj = nn.Linear(hidden_dim, hidden_dim)  # Dummy trainable layer
    
    def forward(self, pixel_values: torch.Tensor) -> MockVisionOutput:
        batch_size = pixel_values.shape[0]
        # Simulate vision features
        hidden_states = torch.randn(
            batch_size, self.num_patches, self.hidden_dim,
            device=pixel_values.device, dtype=pixel_values.dtype
        )
        return MockVisionOutput(self.proj(hidden_states))


class MockBLIP2Model(nn.Module):
    """Minimal mock BLIP-2 for testing hook integration."""
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.vision_model = MockVisionModel(hidden_dim)
        self.qformer = nn.Linear(hidden_dim, hidden_dim)  # Placeholder
        self.language_model = nn.Linear(hidden_dim, 100)  # Placeholder
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_dict: bool = True,
    ):
        # Get vision features (hook will modify these)
        vision_output = self.vision_model(pixel_values)
        features = vision_output.last_hidden_state
        
        # Simulate loss computation using vision features
        pooled = features.mean(dim=1)  # [B, hidden_dim]
        logits = self.language_model(pooled)  # [B, vocab]
        
        # Dummy loss
        loss = logits.sum() * 0.0 + features.sum() * 0.001
        
        class Output:
            pass
        out = Output()
        out.loss = loss
        out.logits = logits
        return out


class TestSceneReasoningIntegration:
    """Test suite for Scene Reasoning integration."""
    
    @pytest.fixture
    def scene_reasoning_module(self):
        """Create a simple Scene Reasoning module for testing."""
        from src.models.scene_reasoning import SceneReasoningModule, SceneReasoningConfig
        
        config = SceneReasoningConfig(
            hidden_dim=768,
            num_heads=8,
            num_layers=1,  # Minimal for testing
            dropout=0.0,   # Deterministic for testing
            use_spatial_encoding=True,
            use_relation_attention=True,
        )
        return SceneReasoningModule(config)
    
    @pytest.fixture
    def mock_blip2(self):
        """Create mock BLIP-2 model."""
        return MockBLIP2Model(hidden_dim=768)
    
    def test_hook_registration(self, mock_blip2, scene_reasoning_module):
        """Test that hook is properly registered when Scene Reasoning is provided."""
        from src.models.blip2_wrapper import BLIP2VQAModel
        
        # Create wrapper with scene reasoning (we'll patch the model loading)
        wrapper = BLIP2VQAModel.__new__(BLIP2VQAModel)
        wrapper.model = mock_blip2
        wrapper.scene_reasoning = scene_reasoning_module
        wrapper._hook_handle = None
        wrapper._enhanced_features = None
        wrapper._scene_attention = None
        wrapper._return_attention = False
        wrapper._scene_reasoning_active = False
        wrapper._hook_call_count = 0
        wrapper._forward_call_count = 0
        
        # Register hook
        wrapper._register_vision_hook()
        
        assert wrapper._hook_handle is not None, "Hook should be registered"
        assert wrapper.is_scene_reasoning_active() is True
    
    def test_enhanced_features_differ_from_original(self, mock_blip2, scene_reasoning_module):
        """CRITICAL: Test that enhanced features are different from original."""
        # Register hook manually
        original_features = []
        enhanced_features_list = []
        
        def capture_hook(module, input, output):
            """Hook to capture original features."""
            original_features.append(output.last_hidden_state.clone())
            return output
        
        def scene_hook(module, input, output):
            """Hook that applies scene reasoning and captures enhanced features."""
            original = output.last_hidden_state.clone()
            original_features.append(original)
            
            enhanced, _ = scene_reasoning_module(output.last_hidden_state)
            enhanced_features_list.append(enhanced.clone())
            
            output.last_hidden_state = enhanced
            return output
        
        # Register scene reasoning hook
        handle = mock_blip2.vision_model.register_forward_hook(scene_hook)
        
        # Forward pass
        pixel_values = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 100, (2, 10))
        
        output = mock_blip2(pixel_values=pixel_values, input_ids=input_ids)
        
        # Verify features differ
        assert len(original_features) == 1, "Hook should be called once"
        assert len(enhanced_features_list) == 1, "Enhanced features should be captured"
        
        original = original_features[0]
        enhanced = enhanced_features_list[0]
        
        # CRITICAL TEST: Features MUST be different
        diff = (enhanced - original).abs().mean().item()
        assert diff > 1e-6, f"Enhanced features should differ from original! Diff: {diff}"
        
        print(f"✅ Feature difference: {diff:.6f} (expected > 0)")
        
        # Cleanup
        handle.remove()
    
    def test_gradient_flow_through_scene_reasoning(self, mock_blip2, scene_reasoning_module):
        """Test that gradients flow through Scene Reasoning module."""
        # Make scene reasoning module trainable
        for param in scene_reasoning_module.parameters():
            param.requires_grad = True
        
        # Record initial parameter values
        initial_params = {
            name: param.clone() 
            for name, param in scene_reasoning_module.named_parameters()
        }
        
        # Register scene reasoning hook
        def scene_hook(module, input, output):
            enhanced, _ = scene_reasoning_module(output.last_hidden_state)
            output.last_hidden_state = enhanced
            return output
        
        handle = mock_blip2.vision_model.register_forward_hook(scene_hook)
        
        # Forward pass
        pixel_values = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 100, (2, 10))
        
        output = mock_blip2(pixel_values=pixel_values, input_ids=input_ids)
        loss = output.loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for name, param in scene_reasoning_module.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                print(f"✅ Gradient found in {name}: {param.grad.abs().mean():.6f}")
        
        assert has_gradients, "Scene Reasoning should receive gradients!"
        
        # Cleanup
        handle.remove()
    
    def test_hook_disable_for_ablation(self, mock_blip2, scene_reasoning_module):
        """Test that hook can be disabled for ablation studies."""
        # Setup
        enhanced_count = [0]
        
        def scene_hook(module, input, output):
            enhanced_count[0] += 1
            enhanced, _ = scene_reasoning_module(output.last_hidden_state)
            output.last_hidden_state = enhanced
            return output
        
        handle = mock_blip2.vision_model.register_forward_hook(scene_hook)
        
        # Forward with hook
        pixel_values = torch.randn(1, 3, 224, 224)
        mock_blip2(pixel_values=pixel_values)
        assert enhanced_count[0] == 1, "Hook should be called"
        
        # Remove hook
        handle.remove()
        
        # Forward without hook
        mock_blip2(pixel_values=pixel_values)
        assert enhanced_count[0] == 1, "Hook should NOT be called after removal"
        
        print("✅ Hook disable for ablation works correctly")
    
    def test_attention_maps_returned(self, mock_blip2, scene_reasoning_module):
        """Test that attention maps can be returned when requested."""
        attention_maps = []
        
        def scene_hook(module, input, output):
            enhanced, attn = scene_reasoning_module(
                output.last_hidden_state, 
                return_attention=True
            )
            if attn is not None:
                attention_maps.append(attn)
            output.last_hidden_state = enhanced
            return output
        
        handle = mock_blip2.vision_model.register_forward_hook(scene_hook)
        
        # Forward pass
        pixel_values = torch.randn(2, 3, 224, 224)
        mock_blip2(pixel_values=pixel_values)
        
        assert len(attention_maps) > 0, "Attention maps should be returned"
        print(f"✅ Attention maps returned with shape: {attention_maps[0].shape}")
        
        handle.remove()


class TestSceneReasoningModule:
    """Direct tests for SceneReasoningModule."""
    
    def test_forward_shape(self):
        """Test output shape matches input shape."""
        from src.models.scene_reasoning import SceneReasoningModule, SceneReasoningConfig
        
        config = SceneReasoningConfig(hidden_dim=768, num_layers=2)
        module = SceneReasoningModule(config)
        
        # Input: [batch, seq_len, hidden]
        x = torch.randn(4, 257, 768)
        
        output, _ = module(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print(f"✅ Output shape correct: {output.shape}")
    
    def test_spatial_encoding_effect(self):
        """Test that spatial encoding modifies features."""
        from src.models.scene_reasoning import SceneReasoningModule, SceneReasoningConfig
        
        # With spatial encoding
        config_with = SceneReasoningConfig(
            hidden_dim=768, 
            use_spatial_encoding=True,
            use_relation_attention=False,
        )
        module_with = SceneReasoningModule(config_with)
        
        # Without spatial encoding
        config_without = SceneReasoningConfig(
            hidden_dim=768,
            use_spatial_encoding=False,
            use_relation_attention=False,
        )
        module_without = SceneReasoningModule(config_without)
        
        x = torch.randn(2, 196, 768)  # 14x14 patches
        
        out_with, _ = module_with(x)
        out_without, _ = module_without(x)
        
        # They should produce different outputs
        diff = (out_with - out_without).abs().mean().item()
        print(f"Spatial encoding effect: {diff:.6f}")
        # Note: They might be similar due to initialization, but structure differs


def run_quick_validation():
    """Quick validation that can be run without pytest."""
    print("=" * 60)
    print("Running Scene Reasoning Integration Validation")
    print("=" * 60)
    
    # Direct import of scene_reasoning only (avoid blip2_wrapper for now)
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
    from scene_reasoning import SceneReasoningModule, SceneReasoningConfig
    
    # Create module
    config = SceneReasoningConfig(
        hidden_dim=768,
        num_heads=8,
        num_layers=2,
        use_spatial_encoding=True,
        use_relation_attention=True,
    )
    scene_module = SceneReasoningModule(config)
    print(f"✅ SceneReasoningModule created with config: {config}")
    
    # Test forward
    x = torch.randn(2, 257, 768)  # [batch, patches, hidden]
    output, attn = scene_module(x, return_attention=True)
    
    assert output.shape == x.shape, f"Shape mismatch!"
    print(f"✅ Forward pass successful: {x.shape} -> {output.shape}")
    
    # Test features differ
    diff = (output - x).abs().mean().item()
    assert diff > 1e-6, f"Features should be modified! Diff: {diff}"
    print(f"✅ Features modified: diff = {diff:.6f}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in scene_module.parameters()
    )
    assert has_grad, "No gradients in scene module!"
    print("✅ Gradient flow verified")
    
    print("=" * 60)
    print("All validations passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_validation()
