"""
BLIP-2 Model Wrapper for Visual Question Answering.

Provides a clean interface around HuggingFace BLIP-2 with:
- Configurable component freezing
- Training forward pass with loss
- Generation for inference
- Scene Reasoning Module integration
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)


class BLIP2VQAModel(nn.Module):
    """
    BLIP-2 wrapper for VQA tasks.
    
    Supports generative VQA with optional Scene Reasoning Module integration.
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
        dtype = dtype_map.get(torch_dtype, torch.float16)
        
        print(f"🔄 Loading BLIP-2: {model_name}")
        
        # Load model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
        
        # Load processor
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Apply freezing
        self._freeze_components(freeze_vision_encoder, freeze_llm, freeze_qformer)
        
        # Scene reasoning module
        self.scene_reasoning = scene_reasoning_module
        
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
        
        print(f"   📊 Total: {total/1e6:.1f}M params")
        print(f"   📊 Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")
    
    def get_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision features."""
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )
        return vision_outputs.last_hidden_state
    
    def apply_scene_reasoning(
        self,
        vision_features: torch.Tensor,
        return_attention: bool = False
    ):
        """Apply scene reasoning if available."""
        if self.scene_reasoning is None:
            return vision_features, None
        
        return self.scene_reasoning(
            vision_features,
            return_attention=return_attention
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            pixel_values: Images [B, C, H, W]
            input_ids: Input tokens [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            labels: Target labels [B, seq_len]
            
        Returns:
            Dict with loss and logits
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return {
            "loss": outputs.loss,
            "logits": getattr(outputs, 'logits', None),
        }
    
    def forward_with_scene_reasoning(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward with scene reasoning features."""
        # Get vision features
        vision_features = self.get_vision_features(pixel_values)
        
        # Apply scene reasoning
        enhanced_features, scene_attention = self.apply_scene_reasoning(
            vision_features, return_attention=return_attention
        )
        
        # Standard forward
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
            "enhanced_features": enhanced_features,
        }
        
        if scene_attention is not None:
            result["scene_attention"] = scene_attention
        
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


def create_model(config) -> BLIP2VQAModel:
    """
    Create BLIP-2 model from config.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured BLIP2VQAModel
    """
    # Scene reasoning module
    scene_module = None
    if config.model.use_scene_reasoning:
        from src.models.scene_reasoning import SceneReasoningModule, SceneReasoningConfig
        
        scene_config = SceneReasoningConfig(
            hidden_dim=config.model.scene_hidden_dim,
            num_heads=config.model.scene_num_heads,
            num_layers=config.model.scene_num_layers,
            mlp_ratio=config.model.scene_mlp_ratio,
            dropout=config.model.scene_dropout,
            use_spatial_encoding=config.model.use_spatial_encoding,
            use_relation_attention=config.model.use_relation_attention,
            spatial_dim=config.model.spatial_encoding_dim,
        )
        scene_module = SceneReasoningModule(scene_config)
    
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
