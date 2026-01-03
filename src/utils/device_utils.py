"""
Device Management Utilities for VLM-VQA Research.

Provides unified device placement strategy to avoid conflicts between
HuggingFace's device_map="auto" and manual .to(device) calls.

Key Features:
- Auto-detect optimal device (CUDA > MPS > CPU)
- Determine optimal dtype per device (fp16 for CUDA, fp32 for MPS/CPU)
- Memory-aware gradient checkpointing decisions
- Unified batch/model device placement
- Mixed precision handling

CRITICAL: This replaces device_map="auto" in model loading to prevent
RuntimeError from conflicting device placements.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about the compute device."""
    device: torch.device
    device_type: str  # "cuda", "mps", "cpu"
    device_index: Optional[int]
    dtype: torch.dtype
    supports_fp16: bool
    supports_bf16: bool
    memory_gb: Optional[float]
    device_name: str


class DeviceManager:
    """
    Unified device management for model training.
    
    Handles:
    - Device detection and selection
    - Optimal dtype selection
    - Model placement (without HuggingFace device_map conflicts)
    - Batch tensor placement
    - Gradient checkpointing decisions
    - Mixed precision context managers
    
    Usage:
        device_manager = DeviceManager("auto")
        model = device_manager.prepare_model(model)
        batch = device_manager.prepare_batch(batch)
        
        with device_manager.autocast():
            outputs = model(**batch)
    """
    
    def __init__(
        self,
        device: str = "auto",
        dtype: Optional[str] = None,
        enable_gradient_checkpointing: Optional[bool] = None,
        memory_threshold_gb: float = 16.0,
    ):
        """
        Initialize DeviceManager.
        
        Args:
            device: Device specification ("auto", "cuda", "cuda:0", "mps", "cpu")
            dtype: Data type ("auto", "float16", "bfloat16", "float32")
            enable_gradient_checkpointing: Force gradient checkpointing (None=auto)
            memory_threshold_gb: GPU memory threshold for auto gradient checkpointing
        """
        self.requested_device = device
        self.requested_dtype = dtype or "auto"
        self.memory_threshold_gb = memory_threshold_gb
        self._force_gradient_checkpointing = enable_gradient_checkpointing
        
        # Resolve device and dtype
        self.device_info = self._detect_device_info()
        
        # Setup mixed precision
        self._setup_mixed_precision()
        
        # Log configuration
        self._log_device_info()
    
    def _detect_device_info(self) -> DeviceInfo:
        """Detect and resolve optimal device configuration."""
        device_type, device_index = self._resolve_device(self.requested_device)
        
        if device_index is not None:
            device = torch.device(f"{device_type}:{device_index}")
        else:
            device = torch.device(device_type)
        
        # Get device capabilities
        supports_fp16 = device_type == "cuda"
        supports_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
        
        # Determine optimal dtype
        dtype = self._get_optimal_dtype(device_type, supports_fp16, supports_bf16)
        
        # Get memory info
        memory_gb = self._get_device_memory(device_type, device_index)
        
        # Get device name
        device_name = self._get_device_name(device_type, device_index)
        
        return DeviceInfo(
            device=device,
            device_type=device_type,
            device_index=device_index,
            dtype=dtype,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            memory_gb=memory_gb,
            device_name=device_name,
        )
    
    def _resolve_device(self, requested: str) -> Tuple[str, Optional[int]]:
        """
        Resolve device string to (device_type, device_index).
        
        Args:
            requested: Device specification
            
        Returns:
            Tuple of (device_type, device_index)
        """
        if requested == "auto":
            # Priority: CUDA > MPS > CPU
            if torch.cuda.is_available():
                return "cuda", 0
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", None
            else:
                return "cpu", None
        
        elif requested.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu", None
            
            if ":" in requested:
                index = int(requested.split(":")[1])
                return "cuda", index
            return "cuda", 0
        
        elif requested == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu", None
            return "mps", None
        
        else:
            return "cpu", None
    
    def _get_optimal_dtype(
        self,
        device_type: str,
        supports_fp16: bool,
        supports_bf16: bool
    ) -> torch.dtype:
        """
        Determine optimal dtype for the device.
        
        CRITICAL:
        - CUDA: fp16 is optimal (or bf16 if preferred)
        - MPS: fp32 only (fp16 has issues on Apple Silicon)
        - CPU: fp32 only
        """
        if self.requested_dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            requested = dtype_map.get(self.requested_dtype, torch.float32)
            
            # Validate requested dtype is supported
            if requested == torch.float16 and not supports_fp16:
                logger.warning(f"float16 not supported on {device_type}, using float32")
                return torch.float32
            if requested == torch.bfloat16 and not supports_bf16:
                logger.warning(f"bfloat16 not supported on {device_type}, using float32")
                return torch.float32
            
            return requested
        
        # Auto-detect
        if device_type == "cuda":
            # Prefer bf16 if available, otherwise fp16
            if supports_bf16:
                return torch.bfloat16
            return torch.float16
        else:
            # MPS and CPU: fp32 only
            return torch.float32
    
    def _get_device_memory(
        self,
        device_type: str,
        device_index: Optional[int]
    ) -> Optional[float]:
        """Get device memory in GB."""
        if device_type == "cuda":
            idx = device_index or 0
            try:
                props = torch.cuda.get_device_properties(idx)
                return props.total_memory / (1024 ** 3)
            except Exception:
                return None
        elif device_type == "mps":
            # MPS shares system memory, can't query directly
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True
                )
                return int(result.stdout.strip()) / (1024 ** 3)
            except Exception:
                return None
        return None
    
    def _get_device_name(
        self,
        device_type: str,
        device_index: Optional[int]
    ) -> str:
        """Get human-readable device name."""
        if device_type == "cuda":
            idx = device_index or 0
            try:
                return torch.cuda.get_device_name(idx)
            except Exception:
                return f"CUDA:{idx}"
        elif device_type == "mps":
            return "Apple Silicon (MPS)"
        else:
            return "CPU"
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training context."""
        self._use_amp = (
            self.device_info.device_type == "cuda" and
            self.device_info.dtype in (torch.float16, torch.bfloat16)
        )
        
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler(
                enabled=self.device_info.dtype == torch.float16
            )
        else:
            self._scaler = None
    
    def _log_device_info(self) -> None:
        """Log device configuration."""
        info = self.device_info
        logger.info(f"DeviceManager initialized:")
        logger.info(f"  Device: {info.device_name} ({info.device})")
        logger.info(f"  Dtype: {info.dtype}")
        if info.memory_gb:
            logger.info(f"  Memory: {info.memory_gb:.1f} GB")
        logger.info(f"  FP16 supported: {info.supports_fp16}")
        logger.info(f"  BF16 supported: {info.supports_bf16}")
        logger.info(f"  Mixed precision (AMP): {self._use_amp}")
    
    @property
    def device(self) -> torch.device:
        """Get the compute device."""
        return self.device_info.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the compute dtype."""
        return self.device_info.dtype
    
    @property
    def use_amp(self) -> bool:
        """Whether AMP is enabled."""
        return self._use_amp
    
    @property
    def scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Get the gradient scaler for mixed precision."""
        return self._scaler
    
    def should_use_gradient_checkpointing(self) -> bool:
        """
        Determine if gradient checkpointing should be enabled.
        
        Automatically enables for GPUs with less than threshold memory.
        """
        if self._force_gradient_checkpointing is not None:
            return self._force_gradient_checkpointing
        
        # Auto-detect based on memory
        if self.device_info.memory_gb is not None:
            return self.device_info.memory_gb < self.memory_threshold_gb
        
        # Default to True for safety on CUDA
        return self.device_info.device_type == "cuda"
    
    def prepare_model(
        self,
        model: nn.Module,
        enable_gradient_checkpointing: Optional[bool] = None,
    ) -> nn.Module:
        """
        Prepare model for training on the target device.
        
        Args:
            model: PyTorch model
            enable_gradient_checkpointing: Override gradient checkpointing setting
            
        Returns:
            Model moved to device with appropriate settings
        """
        # Move to device
        model = model.to(self.device)
        
        # Convert dtype if needed (only for parameters that are floating point)
        if self.dtype != torch.float32:
            model = model.to(self.dtype)
        
        # Enable gradient checkpointing if needed
        use_checkpointing = (
            enable_gradient_checkpointing 
            if enable_gradient_checkpointing is not None
            else self.should_use_gradient_checkpointing()
        )
        
        if use_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("✓ Gradient checkpointing enabled")
            elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
                model.model.gradient_checkpointing_enable()
                logger.info("✓ Gradient checkpointing enabled (inner model)")
        
        return model
    
    def prepare_batch(
        self,
        batch: Dict[str, Any],
        non_blocking: bool = True,
    ) -> Dict[str, Any]:
        """
        Move batch tensors to the target device.
        
        Args:
            batch: Dictionary of batch data
            non_blocking: Use non-blocking transfer
            
        Returns:
            Batch with tensors on target device
        """
        result = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Move tensor to device
                result[key] = value.to(
                    self.device,
                    non_blocking=non_blocking
                )
            elif isinstance(value, list):
                # Keep lists as-is (strings, etc.)
                result[key] = value
            else:
                result[key] = value
        
        return result
    
    def autocast(self):
        """
        Get autocast context manager for mixed precision.
        
        Usage:
            with device_manager.autocast():
                outputs = model(**batch)
        """
        if self._use_amp:
            return torch.cuda.amp.autocast(
                dtype=self.dtype,
                enabled=True
            )
        else:
            # No-op context manager for non-CUDA
            return NullContext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss
    
    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        """
        Step optimizer with proper gradient handling for mixed precision.
        
        Args:
            optimizer: Optimizer to step
            model: Model (for gradient clipping)
            max_grad_norm: Max gradient norm for clipping
        """
        if self._scaler is not None:
            # Unscale gradients
            self._scaler.unscale_(optimizer)
            
            # Clip gradients
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm
                )
            
            # Step optimizer
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            # No AMP - regular step
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm
                )
            optimizer.step()
    
    def estimate_memory_usage(
        self,
        model: nn.Module,
        batch_size: int = 1,
        seq_length: int = 128,
        include_gradients: bool = True,
        include_optimizer: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate memory usage for training.
        
        Args:
            model: Model to estimate for
            batch_size: Batch size
            seq_length: Sequence length
            include_gradients: Include gradient memory
            include_optimizer: Include optimizer states (AdamW = 2x params)
            
        Returns:
            Dict with memory estimates in GB
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Bytes per parameter (depends on dtype)
        if self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            bytes_per_param = 2
        else:
            bytes_per_param = 4
        
        # Model parameters
        model_memory = total_params * bytes_per_param / (1024 ** 3)
        
        # Gradients (same size as trainable params)
        gradient_memory = 0
        if include_gradients:
            gradient_memory = trainable_params * bytes_per_param / (1024 ** 3)
        
        # Optimizer states (AdamW: m and v, both fp32)
        optimizer_memory = 0
        if include_optimizer:
            # AdamW stores m and v in fp32
            optimizer_memory = trainable_params * 4 * 2 / (1024 ** 3)
        
        # Activation memory (rough estimate)
        # This is very approximate - actual depends on model architecture
        activation_memory = (
            batch_size * seq_length * 768 * 4  # hidden states
            * bytes_per_param / (1024 ** 3)
        )
        
        total = model_memory + gradient_memory + optimizer_memory + activation_memory
        
        return {
            "model_gb": model_memory,
            "gradient_gb": gradient_memory,
            "optimizer_gb": optimizer_memory,
            "activation_gb": activation_memory,
            "total_gb": total,
            "available_gb": self.device_info.memory_gb or 0,
            "utilization_pct": (
                100 * total / self.device_info.memory_gb 
                if self.device_info.memory_gb else 0
            ),
        }
    
    def print_memory_estimate(
        self,
        model: nn.Module,
        batch_size: int = 1,
    ) -> None:
        """Print formatted memory usage estimate."""
        estimate = self.estimate_memory_usage(model, batch_size)
        
        print(f"\n{'='*50}")
        print("📊 Memory Usage Estimate")
        print(f"{'='*50}")
        print(f"  Model parameters:  {estimate['model_gb']:.2f} GB")
        print(f"  Gradients:         {estimate['gradient_gb']:.2f} GB")
        print(f"  Optimizer states:  {estimate['optimizer_gb']:.2f} GB")
        print(f"  Activations:       {estimate['activation_gb']:.2f} GB")
        print(f"  {'─'*46}")
        print(f"  Total estimated:   {estimate['total_gb']:.2f} GB")
        
        if estimate['available_gb'] > 0:
            print(f"  Available memory:  {estimate['available_gb']:.1f} GB")
            print(f"  Utilization:       {estimate['utilization_pct']:.1f}%")
            
            if estimate['utilization_pct'] > 90:
                print("  ⚠️  WARNING: High memory usage, consider gradient checkpointing")
            elif estimate['utilization_pct'] > 100:
                print("  ❌ ERROR: Estimated memory exceeds available!")
        
        print(f"{'='*50}\n")
    
    def get_info_dict(self) -> Dict[str, Any]:
        """Get device info as dictionary."""
        return {
            "device": str(self.device_info.device),
            "device_type": self.device_info.device_type,
            "device_name": self.device_info.device_name,
            "dtype": str(self.dtype),
            "memory_gb": self.device_info.memory_gb,
            "supports_fp16": self.device_info.supports_fp16,
            "supports_bf16": self.device_info.supports_bf16,
            "use_amp": self._use_amp,
            "gradient_checkpointing": self.should_use_gradient_checkpointing(),
        }


class NullContext:
    """No-op context manager for non-AMP training."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def get_device_manager(
    config,
    log_info: bool = True,
) -> DeviceManager:
    """
    Create DeviceManager from configuration.
    
    Args:
        config: ExperimentConfig or TrainingConfig
        log_info: Whether to log device info
        
    Returns:
        Configured DeviceManager
    """
    # Handle both ExperimentConfig and TrainingConfig
    if hasattr(config, 'training'):
        training = config.training
    else:
        training = config
    
    device = getattr(training, 'device', 'auto')
    
    # Determine dtype from training config
    dtype = "auto"
    if getattr(training, 'fp16', False):
        dtype = "float16"
    elif getattr(training, 'bf16', False):
        dtype = "bfloat16"
    
    gradient_checkpointing = getattr(training, 'gradient_checkpointing', None)
    
    manager = DeviceManager(
        device=device,
        dtype=dtype,
        enable_gradient_checkpointing=gradient_checkpointing,
    )
    
    if log_info:
        print(f"\n💻 Device: {manager.device_info.device_name}")
        print(f"   Type: {manager.device_info.device_type}")
        print(f"   Dtype: {manager.dtype}")
        if manager.device_info.memory_gb:
            print(f"   Memory: {manager.device_info.memory_gb:.1f} GB")
        print(f"   AMP: {'Enabled' if manager.use_amp else 'Disabled'}")
    
    return manager


# Convenience exports
__all__ = [
    'DeviceManager',
    'DeviceInfo',
    'NullContext',
    'get_device_manager',
]
