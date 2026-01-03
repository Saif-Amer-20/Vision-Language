# Mac-First + Colab-Train Workflow

This document describes the development workflow for the VLM-VQA research project.

## Philosophy

**Git is the single source of truth for code. Google Drive is for outputs only.**

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
                                               │ sync outputs
                                               ▼
                                        ┌─────────────────┐
                                        │  Google Drive   │
                                        │    (Outputs)    │
                                        └─────────────────┘
```

## Execution Profiles

### `mac_dev` - Local Development

**Purpose**: Quick iteration, debugging, testing changes

**Automatic Safety Limits**:
- Maximum 10 training steps
- Maximum 50 samples
- fp16 disabled (MPS limitation)
- No checkpoint saving

```bash
# Automatically detects Mac and applies limits
python scripts/train.py --config configs/mac_dev.yaml

# Or force mac_dev profile
python scripts/train.py --config configs/proposed.yaml --execution_profile mac_dev
```

### `colab_train` - Full Training

**Purpose**: Full experiments with GPU acceleration

**Settings**:
- Unlimited steps/samples
- fp16 enabled
- Full checkpointing
- TensorBoard logging

```python
# In Colab notebook
!git clone https://github.com/yourusername/VLM_Thesis.git
%cd VLM_Thesis
!pip install -r requirements.txt

# Full training
!python scripts/train.py --config configs/proposed.yaml
```

### `eval_only` - Evaluation Mode

**Purpose**: Evaluate trained models without training

```bash
python scripts/eval.py \
    --checkpoint outputs/proposed/best_model.pt \
    --config configs/proposed.yaml
```

## Development Cycle

### 1. Develop on Mac

```bash
# Make changes to code
vim src/models/scene_reasoning.py

# Test locally (auto-limited)
python scripts/train.py --config configs/mac_dev.yaml

# Commit and push
git add .
git commit -m "Improved scene attention mechanism"
git push origin main
```

### 2. Train on Colab

```python
# Cell 1: Setup
!git clone https://github.com/yourusername/VLM_Thesis.git
%cd VLM_Thesis
!pip install -r requirements.txt

# Cell 2: Train (takes hours)
!python scripts/train.py --config configs/proposed.yaml --sync_to_drive
```

### 3. Analyze Results

Results are automatically synced to Drive if `--sync_to_drive` is used:

```
/content/drive/MyDrive/VLM_Thesis_Outputs/
└── proposed/
    └── 20240115_143022/
        ├── checkpoints/
        ├── logs/
        └── results/
```

## Colab Notebook Template

```python
# ============================================================================
# VLM-VQA Research - Training Notebook
# ============================================================================

# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone/Update Repository
import os
if os.path.exists('/content/VLM_Thesis'):
    %cd /content/VLM_Thesis
    !git pull origin main
else:
    !git clone https://github.com/yourusername/VLM_Thesis.git
    %cd /content/VLM_Thesis

# Cell 3: Install Dependencies
!pip install -q -r requirements.txt

# Cell 4: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 5: Smoke Test (optional)
!python scripts/train.py --config configs/smoke_test.yaml

# Cell 6: Full Training
!python scripts/train.py \
    --config configs/proposed.yaml \
    --sync_to_drive

# Cell 7: Evaluation
!python scripts/eval.py \
    --checkpoint outputs/proposed/*/checkpoints/best_model.pt \
    --config configs/proposed.yaml \
    --error_analysis

# Cell 8: Generate Report
!python scripts/make_report.py --results_dir outputs/ --latex
```

## Git Workflow

### What to Commit ✅

```
src/**/*.py          # All source code
scripts/*.py         # Entry scripts
configs/*.yaml       # All configurations
docs/*.md            # Documentation
README.md            # Project README
requirements.txt     # Dependencies
.github/             # GitHub configs
```

### What NOT to Commit ❌

```
outputs/             # Training outputs
*.pt, *.pth          # Checkpoints
__pycache__/         # Python cache
.cache/              # HuggingFace cache
*.ipynb_checkpoints/ # Notebook checkpoints
.env                 # Environment variables
```

### .gitignore

```gitignore
# Outputs
outputs/
thesis_assets/generated/

# Checkpoints
*.pt
*.pth
*.bin

# Cache
__pycache__/
.cache/
*.pyc

# Environment
.env
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

## Troubleshooting

### "RuntimeError: Expected all tensors to be on the same device"

**Cause**: MPS/CPU tensor mixing on Mac

**Solution**: 
```python
# In config
training:
  device: "cpu"  # Force CPU on Mac
```

### "OutOfMemoryError: CUDA out of memory"

**Cause**: Batch size too large for GPU

**Solution**:
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16  # Increase this instead
```

### "Training too slow on Mac"

**Expected**: Mac is for development only!

**Solution**: Use Colab for full training:
```python
!python scripts/train.py --config configs/proposed_colab.yaml
```

## Best Practices

1. **Always test on Mac first** before pushing to Colab
2. **Use smoke_test.yaml** for quick validation
3. **Commit frequently** with descriptive messages
4. **Never edit code on Colab** - always pull from Git
5. **Use --sync_to_drive** for long training runs
6. **Monitor with TensorBoard** during training

## Resources

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [VQAv2 Dataset](https://visualqa.org/)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)
- [Google Colab Pro](https://colab.research.google.com/signup)
