"""
BLIP-2 Model Wrapper for Visual Question Answering.

Provides a clean interface around HuggingFace BLIP-2 with:
- Configurable component freezing
- Training forward pass with loss
- Generation for inference
- Scene Reasoning Module integration via forward hooks

CRITICAL: Uses PyTorch forward hooks to inject Scene Reasoning into
the BLIP-2 pipeline WITHOUT modifying the pretrained model architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
import logging
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

# Setup logging
logger = logging.getLogger(__name__)


class BLIP2VQAModel(nn.Module):
    """
    BLIP-2 wrapper for VQA tasks with Scene Reasoning integration.
    
    Uses forward hooks to intercept and enhance vision features BEFORE
    they reach the Q-Former, ensuring the Scene Reasoning Module actually
    contributes to the model's predictions.
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        freeze_vision_encoder: bool = True,
        freeze_llm: bool = True,
        freeze_qformer: bool = False,
        torch_dtype: str = "float16",
        device_map: str = "auto",
        scene_reasoning_module: Optional[nn.Module] = None,
        max_new_tokens: int = 16,
        num_beams: int = 3,
    ):
        """
        Initialize BLIP-2 VQA model.
        
        Args:
            model_name: HuggingFace model identifier
            freeze_vision_encoder: Freeze vision encoder
            freeze_llm: Freeze language model
            freeze_qformer: Freeze Q-Former
            torch_dtype: Model precision (float16, bfloat16, float32)
            device_map: Device mapping
            scene_reasoning_module: Optional scene reasoning module
            max_new_tokens: Max tokens for generation
            num_beams: Beam search width
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        
        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(torch_dtype, torch.float16)
        
        print(f"🔄 Loading BLIP-2: {model_name}")
        
        # Load model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=self.dtype,
        )
        
        # Load processor
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Apply freezing
        self._freeze_components(freeze_vision_encoder, freeze_llm, freeze_qformer)
        
        # Scene reasoning module
        self.scene_reasoning = scene_reasoning_module
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        
        # State for hook-based injection
        self._enhanced_features: Optional[torch.Tensor] = None
        self._scene_attention: Optional[torch.Tensor] = None
        self._return_attention: bool = False
        self._scene_reasoning_active: bool = False
        
        # Validation counters for debugging
        self._hook_call_count: int = 0
        self._forward_call_count: int = 0
        
        # Register the hook if scene reasoning is available
        if self.scene_reasoning is not None:
            self._register_vision_hook()
            print(f"   🧠 Scene Reasoning Module: ACTIVE (hook registered)")
        else:
            print(f"   🧠 Scene Reasoning Module: DISABLED")
        
        # Log parameter counts
        self._log_params()
    
    def _freeze_components(
        self,
        freeze_vision: bool,
        freeze_llm: bool,
        freeze_qformer: bool
    ) -> None:
        """Freeze model components."""
        if freeze_vision:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            print("   ❄️ Vision encoder frozen")
        
        if freeze_llm:
            for param in self.model.language_model.parameters():
                param.requires_grad = False
            print("   ❄️ Language model frozen")
        
        if freeze_qformer:
            for param in self.model.qformer.parameters():
                param.requires_grad = False
            print("   ❄️ Q-Former frozen")
    
    def _log_params(self) -> None:
        """Log parameter statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count scene reasoning params separately
        scene_params = 0
        if self.scene_reasoning is not None:
            scene_params = sum(p.numel() for p in self.scene_reasoning.parameters())
        
        print(f"   📊 Total: {total/1e6:.1f}M params")
        print(f"   📊 Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")
        if scene_params > 0:
            print(f"   📊 Scene Reasoning: {scene_params/1e6:.1f}M params")
    
    def _register_vision_hook(self) -> None:
        """
        Register forward hook on vision encoder to intercept and enhance features.
        
        The hook intercepts the output of the vision model and applies
        Scene Reasoning BEFORE the features go to Q-Former.
        """
        def vision_hook(
            module: nn.Module,
            input: tuple,
            output: Any
        ) -> Any:
            """
            Forward hook that applies Scene Reasoning to vision features.
            
            This hook:
            1. Intercepts vision encoder output
            2. Applies Scene Reasoning Module
            3. Returns enhanced features (which replace original in forward pass)
            """
            self._hook_call_count += 1
            self._scene_reasoning_active = True
            
            # Extract last_hidden_state from vision output
            if hasattr(output, 'last_hidden_state'):
                vision_features = output.last_hidden_state
            elif isinstance(output, tuple):
                vision_features = output[0]
            else:
                vision_features = output
            
            # Log first call for debugging
            if self._hook_call_count == 1:
                logger.info(
                    f"🎯 Scene Reasoning Hook activated! "
                    f"Input shape: {vision_features.shape}"
                )
            
            # Apply Scene Reasoning Module
            enhanced_features, scene_attention = self.scene_reasoning(
                vision_features,
                return_attention=self._return_attention
            )
            
            # Store for external access (optional)
            self._enhanced_features = enhanced_features
            self._scene_attention = scene_attention
            
            # Validate enhancement (debug mode)
            if self._hook_call_count <= 3:
                with torch.no_grad():
                    diff = (enhanced_features - vision_features).abs().mean().item()
                    logger.debug(
                        f"   Scene Reasoning diff: {diff:.6f} "
                        f"(should be > 0 if module is working)"
                    )
                    if diff < 1e-8:
                        logger.warning(
                            "⚠️ Enhanced features identical to original! "
                            "Scene Reasoning may not be working correctly."
                        )
            
            # Return modified output with enhanced features
            # BLIP-2's vision model returns BaseModelOutputWithPooling
            if hasattr(output, 'last_hidden_state'):
                # Create a new output object with enhanced features
                output.last_hidden_state = enhanced_features
                return output
            elif isinstance(output, tuple):
                # Replace first element (hidden states) with enhanced
                return (enhanced_features,) + output[1:]
            else:
                return enhanced_features
        
        # Register hook on vision model
        self._hook_handle = self.model.vision_model.register_forward_hook(vision_hook)
        logger.info("✅ Vision encoder hook registered for Scene Reasoning")
    
    def remove_hook(self) -> None:
        """Remove the vision hook (useful for ablation or debugging)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self._scene_reasoning_active = False
            logger.info("🔌 Vision encoder hook removed")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_scene_attention: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        If Scene Reasoning Module is active, it automatically enhances
        vision features via the registered forward hook.
        
        Args:
            pixel_values: Images [B, C, H, W]
            input_ids: Input tokens [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            labels: Target labels [B, seq_len]
            return_scene_attention: Whether to return scene attention maps
            
        Returns:
            Dict with loss, logits, and optionally scene_attention
        """
        self._forward_call_count += 1
        self._return_attention = return_scene_attention
        
        # Reset state
        self._enhanced_features = None
        self._scene_attention = None
        
        # Log scene reasoning status periodically
        if self._forward_call_count == 1:
            if self.scene_reasoning is not None:
                logger.info(
                    f"🧠 Forward pass #{self._forward_call_count}: "
                    f"Scene Reasoning ACTIVE"
                )
            else:
                logger.info(
                    f"📷 Forward pass #{self._forward_call_count}: "
                    f"Baseline mode (no Scene Reasoning)"
                )
        
        # Forward through BLIP-2
        # The hook automatically intercepts and enhances vision features
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        result = {
            "loss": outputs.loss,
            "logits": getattr(outputs, 'logits', None),
        }
        
        # Include enhanced features if Scene Reasoning was applied
        if self._enhanced_features is not None:
            result["enhanced_features"] = self._enhanced_features
        
        # Include scene attention if requested
        if return_scene_attention and self._scene_attention is not None:
            result["scene_attention"] = self._scene_attention
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate answers.
        
        Scene Reasoning is automatically applied via hook if enabled.
        
        Args:
            pixel_values: Images [B, C, H, W]
            input_ids: Question tokens [B, seq_len]
            attention_mask: Attention mask
            max_new_tokens: Max tokens to generate
            num_beams: Beam search width
            
        Returns:
            List of generated answers
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        num_beams = num_beams or self.num_beams
        
        # The hook enhances vision features automatically
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            **kwargs
        )
        
        # Decode
        texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Clean up answers
        answers = []
        for text in texts:
            text = text.strip()
            if "Answer:" in text:
                text = text.split("Answer:")[-1].strip()
            answers.append(text)
        
        return answers
    
    def get_processor(self) -> Blip2Processor:
        """Get BLIP-2 processor."""
        return self.processor
    
    def is_scene_reasoning_active(self) -> bool:
        """Check if Scene Reasoning is currently active."""
        return self.scene_reasoning is not None and self._hook_handle is not None
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about the model state."""
        return {
            "scene_reasoning_enabled": self.scene_reasoning is not None,
            "hook_registered": self._hook_handle is not None,
            "hook_call_count": self._hook_call_count,
            "forward_call_count": self._forward_call_count,
            "scene_reasoning_was_active": self._scene_reasoning_active,
            "last_enhanced_features_shape": (
                self._enhanced_features.shape if self._enhanced_features is not None 
                else None
            ),
        }


def create_model(config) -> BLIP2VQAModel:
    """
    Create BLIP-2 model from config.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        Configured BLIP2VQAModel with optional Scene Reasoning
    """
    # Scene reasoning module
    scene_module = None
    if getattr(config.model, 'use_scene_reasoning', False):
        from src.models.scene_reasoning import SceneReasoningModule, SceneReasoningConfig
        
        scene_config = SceneReasoningConfig(
            hidden_dim=getattr(config.model, 'scene_hidden_dim', 768),
            num_heads=getattr(config.model, 'scene_num_heads', 8),
            num_layers=getattr(config.model, 'scene_num_layers', 2),
            mlp_ratio=getattr(config.model, 'scene_mlp_ratio', 4.0),
            dropout=getattr(config.model, 'scene_dropout', 0.1),
            use_spatial_encoding=getattr(config.model, 'use_spatial_encoding', True),
            use_relation_attention=getattr(config.model, 'use_relation_attention', True),
            spatial_dim=getattr(config.model, 'spatial_encoding_dim', 64),
        )
        scene_module = SceneReasoningModule(scene_config)
        print(f"🧠 Scene Reasoning Config: {scene_config}")
    
    model = BLIP2VQAModel(
        model_name=config.model.model_name,
        freeze_vision_encoder=config.model.freeze_vision_encoder,
        freeze_llm=config.model.freeze_llm,
        freeze_qformer=config.model.freeze_qformer,
        torch_dtype=config.model.torch_dtype,
        scene_reasoning_module=scene_module,
        max_new_tokens=config.model.max_new_tokens,
        num_beams=config.model.num_beams,
    )
    
    return model
