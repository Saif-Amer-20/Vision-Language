# Vision-Language Project - AI Coding Instructions

<!-- 
This file guides AI coding agents working in this codebase.
Last updated: Mac-first + Colab-train workflow added.
-->

## Project Overview

This is a Vision-Language research project implementing a **Scene Reasoning Module** 
that enhances BLIP-2's visual question answering capabilities through explicit 
spatial relationship modeling.

**Research Focus**: VQA with spatial reasoning using BLIP-2 + custom Scene Reasoning Module

**Development Model**: Mac-first + Colab-train (Git-based sync)

## Execution Profiles

The project supports three execution profiles:

| Profile | Environment | Purpose | Training Limits |
|---------|-------------|---------|-----------------|
| `colab_train` | Colab GPU | Full training | Unlimited (default) |
| `mac_dev` | Mac/Local | Development | ≤10 steps, ≤50 samples |
| `eval_only` | Any | Evaluation | No training |

**CRITICAL**: Never reduce BLIP-2 model size/architecture for any profile. Only runtime settings differ.

## Architecture

```
VLM_Thesis/
├── configs/                    # YAML experiment configurations
│   ├── baseline.yaml          # BLIP-2 baseline (auto-detect env)
│   ├── proposed.yaml          # BLIP-2 + Scene Reasoning (auto-detect)
│   ├── baseline_mac.yaml      # Mac development overlay
│   ├── proposed_mac.yaml      # Mac development overlay
│   ├── baseline_colab.yaml    # Colab with Drive sync option
│   ├── proposed_colab.yaml    # Colab with Drive sync option
│   └── ablation_*.yaml        # Ablation study configs
├── src/
│   ├── datasets/              # VQA dataset loaders
│   ├── models/                # BLIP-2 wrapper, Scene Reasoning Module
│   ├── training/              # Trainer, losses, metrics, schedulers
│   ├── evaluation/            # Evaluation pipeline, error analysis
│   └── utils/                 # Config (with ExecutionProfile), seed, io, logging
├── scripts/
│   ├── train.py               # Training with --execution_profile, --sync_to_drive
│   ├── eval.py                # Evaluation entry point
│   └── make_report.py         # Report generation
├── docs/
│   └── WORKFLOW.md            # Mac-Colab workflow documentation
├── outputs/                   # Experiment outputs (gitignored)
└── thesis_assets/             # Thesis-ready materials
```

## Mac-First + Colab-Train Workflow

### Development Cycle

```
┌─────────────────┐      git push       ┌─────────────────┐
│   Mac / VS Code │ ──────────────────▶ │     GitHub      │
│   (Development) │ ◀────────────────── │   (Repository)  │
└─────────────────┘      git pull       └─────────────────┘
                                               │ git clone/pull
                                               ▼
                                        ┌─────────────────┐
                                        │   Google Colab  │
                                        │   (Training)    │
                                        └─────────────────┘
```

### On Mac (Development)
```bash
# Test locally (auto-limited to 10 steps)
python scripts/train.py --config configs/baseline_mac.yaml

# Or explicitly set profile
python scripts/train.py --config configs/proposed.yaml --execution_profile mac_dev

# Commit and push
git add . && git commit -m "message" && git push
```

### On Colab (Training)
```python
# Pull latest code
!git pull origin main

# Full training
!python scripts/train.py --config configs/proposed_colab.yaml

# With Drive sync
!python scripts/train.py --config configs/proposed_colab.yaml --sync_to_drive
```

## Key Commands

```bash
# Mac Development (smoke test)
python scripts/train.py --config configs/baseline_mac.yaml
python scripts/train.py --config configs/proposed_mac.yaml

# Colab Full Training
python scripts/train.py --config configs/baseline_colab.yaml --sync_to_drive
python scripts/train.py --config configs/proposed_colab.yaml --sync_to_drive

# Evaluation (any environment)
python scripts/eval.py --checkpoint outputs/proposed/best_model.pt --config configs/proposed.yaml

# Generate reports
python scripts/make_report.py --results_dir outputs/
```

## CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config (required) | - |
| `--execution_profile` | `colab_train`, `mac_dev`, `eval_only` | Auto-detect |
| `--sync_to_drive` | Sync outputs to Google Drive | `false` |
| `--smoke_test` | Quick test mode | `false` |

## Safety Guards (mac_dev profile)

The `mac_dev` profile enforces strict limits to prevent accidental long training:

```python
# These limits are ENFORCED (not just warnings):
max_steps: 10          # Training stops after 10 steps
max_samples: 50        # Maximum dataset samples
fp16: false            # Disabled (MPS limitation)
save_checkpoints: false # No checkpoint saving
device: mps/cpu        # Force local device
```

Attempting to exceed limits raises an error with instructions to use Colab.

## Git Rules

### What to Commit ✅
- `src/**/*.py` - All source code
- `scripts/*.py` - Training/eval scripts
- `configs/*.yaml` - All config files
- `README.md`, `requirements.txt`
- `.github/copilot-instructions.md`

### What NOT to Commit ❌
- `outputs/` - Training outputs
- `*.pt`, `*.pth` - Checkpoints
- `__pycache__/`, `.cache/`
- `.ipynb_checkpoints/`

### Never Use Drive as Code Source
- Git is the ONLY source of truth for code
- Drive is for OUTPUT sync only

## Important Conventions

### Config Pattern
```python
@dataclass
class RuntimeConfig:
    execution_profile: str = "colab_train"  # colab_train, mac_dev, eval_only
    sync_to_drive: bool = False
    mac_dev_max_steps: int = 10
    mac_dev_max_samples: int = 50
```

### Module Patterns
```python
class ModuleName(nn.Module):
    def __init__(self, config: ConfigClass):
        super().__init__()
        # Initialize from config
    
    def forward(self, x: Tensor, ...) -> Tuple[Tensor, Optional[Dict]]:
        # Return output and optional auxiliary info
```

### Device Handling
```python
# In config
device: "auto"  # Auto-detect: cuda > mps > cpu

# Get effective device
device = config.get_effective_device()
```

## Testing

```bash
# Mac smoke test
python scripts/train.py --config configs/baseline_mac.yaml --smoke_test true

# Verify imports
python -c "from src.models.scene_reasoning import SceneReasoningModule; print('OK')"
```

## Colab-Specific Notes

- Working directory: `/content/VLM_Thesis`
- Outputs saved to: `/content/VLM_Thesis/outputs`
- Drive sync path: `/content/drive/MyDrive/VLM_Thesis_Outputs`
- Always use fp16 on Colab
- Freeze BLIP-2 backbone (memory constraint)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SomeConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

## Testing

```bash
# Smoke test (verify imports)
python -c "from src.models.scene_reasoning import SceneReasoningModule; print('OK')"

# Run with small dataset subset
python scripts/train.py --config configs/baseline.yaml --data.train_split "train[:100]"
```

## Colab-Specific Notes

- Working directory: `/content/VLM_Thesis`
- Batch size: 1 (memory constraints)
- Gradient accumulation: 8 steps
- Always use fp16
- Freeze BLIP-2 backbone (vision + language)
- Use `device_map='auto'` for model loading
