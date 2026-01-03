"""
Scene Reasoning Module for Enhanced Spatial Understanding.

This module enhances vision features with explicit spatial and relational reasoning:
1. Spatial Position Encodings: 2D relative position encodings for patches
2. Relation-Aware Self-Attention: Models relationships between image regions
3. Interpretability: Exposes attention weights for visualization

Design for Ablation Studies:
- use_spatial_encoding: Enable/disable spatial position encodings
- use_relation_attention: Enable/disable relation-aware attention
- Both can be toggled via config for systematic ablation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


@dataclass
class SceneReasoningConfig:
    """Configuration for Scene Reasoning Module."""
    hidden_dim: int = 768
    num_heads: int = 8
    num_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_spatial_encoding: bool = True
    use_relation_attention: bool = True
    spatial_dim: int = 64
    max_positions: int = 24  # Max patches per dimension
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SceneReasoningConfig':
        """Create config from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


class SpatialPositionEncoding(nn.Module):
    """
    2D Spatial Position Encoding for image patches.
    
    Creates learnable relative position encodings based on 2D spatial relationships.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        spatial_dim: int = 64,
        max_positions: int = 24,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.max_positions = max_positions
        
        # Learnable row and column embeddings
        self.row_embed = nn.Embedding(max_positions, spatial_dim // 2)
        self.col_embed = nn.Embedding(max_positions, spatial_dim // 2)
        
        # Project to hidden dimension
        self.position_proj = nn.Linear(spatial_dim, hidden_dim)
        
        # Relative position bias for attention
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * max_positions - 1, 2 * max_positions - 1)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
    
    def get_absolute_positions(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute absolute 2D position encodings.
        
        Handles CLS token by checking if num_patches is a perfect square.
        If not (e.g., 257 = 256 + 1), assumes first token is CLS and
        assigns it a special position encoding.
        
        Args:
            batch_size: Batch size
            num_patches: Total patches (may include CLS token)
            device: Device
            
        Returns:
            Position encodings [B, num_patches, hidden_dim]
        """
        # Check for CLS token
        has_cls = False
        grid_patches = num_patches
        sqrt_check = int(math.sqrt(num_patches))
        
        if sqrt_check * sqrt_check != num_patches:
            # Not a perfect square - likely has CLS token
            has_cls = True
            grid_patches = num_patches - 1
            sqrt_check = int(math.sqrt(grid_patches))
            
            if sqrt_check * sqrt_check != grid_patches:
                # Still not a perfect square, fall back to nearest
                sqrt_check = int(math.sqrt(grid_patches) + 0.5)
                grid_patches = sqrt_check * sqrt_check
        
        grid_size = sqrt_check
        
        # Create position indices
        rows = torch.arange(min(grid_size, self.max_positions), device=device)
        cols = torch.arange(min(grid_size, self.max_positions), device=device)
        
        # Get embeddings
        row_emb = self.row_embed(rows)  # [grid_size, spatial_dim/2]
        col_emb = self.col_embed(cols)  # [grid_size, spatial_dim/2]
        
        # Create 2D grid of embeddings
        actual_grid = min(grid_size, self.max_positions)
        row_emb = row_emb.unsqueeze(1).expand(-1, actual_grid, -1)  # [H, W, D/2]
        col_emb = col_emb.unsqueeze(0).expand(actual_grid, -1, -1)  # [H, W, D/2]
        
        # Concatenate and flatten
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # [H, W, D]
        pos_emb = pos_emb.view(-1, self.spatial_dim)     # [H*W, D]
        
        # Project to hidden dimension
        pos_emb = self.position_proj(pos_emb)  # [H*W, hidden_dim]
        
        # Handle CLS token - prepend a learned CLS position encoding
        if has_cls:
            # Create CLS position encoding (zeros or learned)
            cls_pos = torch.zeros(1, self.hidden_dim, device=device, dtype=pos_emb.dtype)
            pos_emb = torch.cat([cls_pos, pos_emb], dim=0)  # [1 + H*W, hidden_dim]
        
        # Ensure we have exactly num_patches
        if pos_emb.size(0) < num_patches:
            # Pad with zeros if needed
            pad = torch.zeros(num_patches - pos_emb.size(0), self.hidden_dim, device=device, dtype=pos_emb.dtype)
            pos_emb = torch.cat([pos_emb, pad], dim=0)
        elif pos_emb.size(0) > num_patches:
            # Truncate if needed
            pos_emb = pos_emb[:num_patches]
        
        # Expand for batch
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pos_emb
    
    def get_relative_position_bias(self, num_patches: int) -> torch.Tensor:
        """
        Get relative position bias matrix for attention.
        
        Handles CLS token by computing bias for grid patches and 
        padding for the CLS token.
        
        Args:
            num_patches: Total patches (may include CLS token)
            
        Returns:
            Bias matrix [num_patches, num_patches]
        """
        # Check for CLS token
        has_cls = False
        grid_patches = num_patches
        sqrt_check = int(math.sqrt(num_patches))
        
        if sqrt_check * sqrt_check != num_patches:
            has_cls = True
            grid_patches = num_patches - 1
            sqrt_check = int(math.sqrt(grid_patches))
        
        grid_size = min(sqrt_check, self.max_positions)
        actual_patches = grid_size * grid_size
        
        # Create coordinate grids
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing='ij'
        ))  # [2, H, W]
        coords = coords.flatten(1)  # [2, H*W]
        
        # Compute relative coordinates
        relative_coords = coords[:, :, None] - coords[:, None, :]  # [2, H*W, H*W]
        relative_coords = relative_coords.permute(1, 2, 0)  # [H*W, H*W, 2]
        
        # Shift to positive indices
        relative_coords[:, :, 0] += self.max_positions - 1
        relative_coords[:, :, 1] += self.max_positions - 1
        
        # Compute bias indices
        relative_coords[:, :, 0] *= 2 * self.max_positions - 1
        relative_position_index = relative_coords.sum(-1)  # [H*W, H*W]
        
        # Gather bias values
        bias = self.relative_position_bias.view(-1)[
            relative_position_index.view(-1)
        ].view(actual_patches, actual_patches)
        
        # Handle CLS token - extend bias matrix
        if has_cls or actual_patches < num_patches:
            # Create full bias matrix with zeros for CLS
            full_bias = torch.zeros(
                num_patches, num_patches, 
                device=self.relative_position_bias.device,
                dtype=self.relative_position_bias.dtype
            )
            # Fill in the grid-to-grid bias (skip CLS token at position 0)
            start_idx = 1 if has_cls else 0
            end_idx = start_idx + actual_patches
            if end_idx <= num_patches:
                full_bias[start_idx:end_idx, start_idx:end_idx] = bias
            bias = full_bias
        
        return bias
    
    def forward(
        self,
        x: torch.Tensor,
        return_bias: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Add spatial position encodings.
        
        Args:
            x: Input features [B, N, D]
            return_bias: Return relative position bias
            
        Returns:
            (encoded features, optional bias)
        """
        B, N, D = x.shape
        
        # Add absolute position encodings
        pos_enc = self.get_absolute_positions(B, N, x.device)
        x = x + pos_enc
        
        # Optionally return relative position bias
        bias = None
        if return_bias:
            bias = self.get_relative_position_bias(N)
        
        return x, bias


class RelationAwareAttention(nn.Module):
    """
    Relation-Aware Multi-Head Self-Attention.
    
    Extends standard attention with:
    - Relative position bias for spatial reasoning
    - Interpretable attention weight output
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input [B, N, D]
            position_bias: Relative position bias [N, N]
            return_attention: Return attention weights
            
        Returns:
            (output, optional attention weights)
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Add position bias if provided
        if position_bias is not None:
            attn = attn + position_bias.unsqueeze(0).unsqueeze(0)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn = self.dropout(attn_weights)
        
        # Apply to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


class SceneReasoningLayer(nn.Module):
    """
    Single Scene Reasoning Layer.
    
    Combines relation-aware attention with feedforward network.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_relation_attention: bool = True,
    ):
        super().__init__()
        
        self.use_relation_attention = use_relation_attention
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Attention
        if use_relation_attention:
            self.attention = RelationAwareAttention(hidden_dim, num_heads, dropout)
        else:
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # FFN
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input [B, N, D]
            position_bias: Position bias for attention
            return_attention: Return attention weights
            
        Returns:
            (output, optional attention)
        """
        attn_weights = None
        
        # Attention
        normed = self.norm1(x)
        if self.use_relation_attention:
            attn_out, attn_weights = self.attention(
                normed, position_bias, return_attention
            )
        else:
            attn_out, attn_weights = self.attention(
                normed, normed, normed,
                need_weights=return_attention
            )
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights


class SceneReasoningModule(nn.Module):
    """
    Scene Reasoning Module for spatial and relational understanding.
    
    Enhances vision features with:
    - 2D spatial position encodings
    - Relation-aware self-attention layers
    - Interpretable attention output
    
    Designed for ablation studies with configurable components.
    """
    
    def __init__(self, config: SceneReasoningConfig):
        """
        Initialize Scene Reasoning Module.
        
        Args:
            config: Module configuration
        """
        super().__init__()
        
        self.config = config
        
        # Input projection (match vision encoder output)
        self.input_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Spatial position encoding (optional for ablation)
        self.spatial_encoding = None
        if config.use_spatial_encoding:
            self.spatial_encoding = SpatialPositionEncoding(
                hidden_dim=config.hidden_dim,
                spatial_dim=config.spatial_dim,
                max_positions=config.max_positions,
            )
        
        # Scene reasoning layers
        self.layers = nn.ModuleList([
            SceneReasoningLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                use_relation_attention=config.use_relation_attention,
            )
            for _ in range(config.num_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        
        self._init_weights()
        self._log_config()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_config(self) -> None:
        """Log configuration."""
        params = sum(p.numel() for p in self.parameters())
        print(f"🧩 Scene Reasoning Module:")
        print(f"   Layers: {self.config.num_layers}")
        print(f"   Heads: {self.config.num_heads}")
        print(f"   Spatial Encoding: {'✓' if self.config.use_spatial_encoding else '✗'}")
        print(f"   Relation Attention: {'✓' if self.config.use_relation_attention else '✗'}")
        print(f"   Parameters: {params/1e6:.2f}M")
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            x: Vision features [B, N, D]
            return_attention: Return attention weights
            
        Returns:
            (enhanced features, optional attention dict)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply spatial encoding
        position_bias = None
        if self.spatial_encoding is not None:
            x, position_bias = self.spatial_encoding(x, return_bias=True)
        
        # Collect attention weights
        all_attention = []
        
        # Apply layers
        for layer in self.layers:
            x, attn = layer(x, position_bias, return_attention)
            if attn is not None:
                all_attention.append(attn)
        
        # Output normalization
        x = self.output_norm(x)
        
        # Return attention if requested
        attention_dict = None
        if return_attention and all_attention:
            attention_dict = {
                "layer_attention": torch.stack(all_attention, dim=1),  # [B, L, H, N, N]
                "position_bias": position_bias,
            }
        
        return x, attention_dict
    
    def get_attention_maps(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            x: Vision features [B, N, D]
            
        Returns:
            Dictionary of attention tensors
        """
        _, attention_dict = self.forward(x, return_attention=True)
        return attention_dict or {}
