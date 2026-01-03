# Configuration Files Documentation

This directory contains all experiment configurations for the VLM-VQA Research project,
including baseline, proposed model, and ablation study configurations.

---

## A. Configuration Files Overview

| Config File | Scene Reasoning | Spatial Enc | Relation Att | Purpose |
|-------------|:---------------:|:-----------:|:------------:|---------|
| `baseline.yaml` | ✗ | - | - | BLIP-2 without modifications |
| `proposed.yaml` | ✓ | ✓ | ✓ | Full Scene Reasoning Module |
| `ablation_spatial_only.yaml` | ✓ | ✓ | ✗ | Isolate spatial encoding |
| `ablation_relation_only.yaml` | ✓ | ✗ | ✓ | Isolate relation attention |
| `ablation_no_spatial.yaml` | ✓ | ✗ | ✗ | Architecture overhead test |
| `ablation_no_relation.yaml` | ✓ | ✓ | ✗ | Same as spatial_only |
| `mac_dev.yaml` | ✓ | ✓ | ✓ | Local development (10 steps max) |
| `smoke_test.yaml` | ✓ | ✓ | ✓ | Quick validation |

### Config Purposes

- **baseline.yaml**: Pure BLIP-2 model without Scene Reasoning Module. Establishes performance baseline.
- **proposed.yaml**: Full model with both Spatial Position Encoding and Relation-Aware Attention.
- **ablation_spatial_only.yaml**: Tests contribution of 2D position embeddings alone.
- **ablation_relation_only.yaml**: Tests relation attention architecture without position bias.
- **ablation_no_spatial.yaml**: Tests Scene Reasoning architecture with both components disabled.
- **ablation_no_relation.yaml**: Equivalent to spatial_only (kept for naming symmetry).
- **mac_dev.yaml**: Safe local testing with enforced limits (10 steps, 50 samples).
- **smoke_test.yaml**: Quick validation that code runs without errors.

---

## B. Ablation Study Design

### Research Questions

| ID | Research Question | Config Comparison |
|----|-------------------|-------------------|
| **RQ1** | Does Scene Reasoning improve VQA performance? | `proposed` vs `baseline` |
| **RQ2** | What is the contribution of Spatial Position Encoding? | `spatial_only` vs `baseline` |
| **RQ3** | What is the contribution of Relation-Aware Attention? | `proposed` vs `spatial_only` |
| **RQ4** | Is there synergy between components? | `proposed` vs `spatial_only` + `relation_only` |

### Expected Performance Hierarchy

```
                            Higher Performance
                                    ↑
    ┌──────────────────────────────────────────────────────────┐
    │                      proposed                             │  ← Best (both components)
    │               (Spatial + Relation)                        │
    ├──────────────────────────────────────────────────────────┤
    │            ablation_spatial_only                          │  ← Spatial contributes more
    │           (Spatial, no Relation)                          │
    ├──────────────────────────────────────────────────────────┤
    │           ablation_relation_only                          │  ← Relation alone (no bias)
    │           (Relation, no Spatial)                          │
    ├──────────────────────────────────────────────────────────┤
    │            ablation_no_spatial                            │  ← Architecture overhead
    │           (Neither component)                             │
    ├──────────────────────────────────────────────────────────┤
    │                     baseline                              │  ← Pure BLIP-2
    │               (No Scene Reasoning)                        │
    └──────────────────────────────────────────────────────────┘
                                    ↓
                            Lower Performance
```

### Component Matrix (All Combinations)

| Configuration | Scene Module | Spatial Enc | Relation Att | Position Bias | Expected |
|---------------|:------------:|:-----------:|:------------:|:-------------:|----------|
| baseline | ✗ | - | - | - | Baseline |
| ablation_no_spatial | ✓ | ✗ | ✗ | ✗ | ≈ Baseline |
| ablation_relation_only | ✓ | ✗ | ✓ | ✗ | > Baseline |
| ablation_spatial_only | ✓ | ✓ | ✗ | ✗ | > Baseline |
| proposed | ✓ | ✓ | ✓ | ✓ | Best |

---

## C. Running Experiments

### Full Training Commands

```bash
# Baseline (BLIP-2 only)
python scripts/train.py --config configs/baseline.yaml

# Proposed (Full Scene Reasoning)
python scripts/train.py --config configs/proposed.yaml

# Ablation: Spatial Encoding Only
python scripts/train.py --config configs/ablation_spatial_only.yaml

# Ablation: Relation Attention Only  
python scripts/train.py --config configs/ablation_relation_only.yaml

# Ablation: Neither Component (Architecture Only)
python scripts/train.py --config configs/ablation_no_spatial.yaml
```

### Quick Validation (Smoke Test)

```bash
# Any config with smoke_test flag
python scripts/train.py --config configs/proposed.yaml --smoke_test

# Or use dedicated smoke test config
python scripts/train.py --config configs/smoke_test.yaml
```

### Local Development (Mac)

```bash
# Safe local testing (auto-limits to 10 steps)
python scripts/train.py --config configs/mac_dev.yaml

# Or override execution profile
python scripts/train.py --config configs/proposed.yaml --execution_profile mac_dev
```

### Override Examples

```bash
# Quick test with fewer epochs
python scripts/train.py --config configs/proposed.yaml \
    --training.num_epochs 1 \
    --data.max_train_samples 1000

# Different learning rate
python scripts/train.py --config configs/proposed.yaml \
    --training.learning_rate 5e-6

# Sync to Google Drive (Colab)
python scripts/train.py --config configs/proposed.yaml --sync_to_drive
```

---

## D. Evaluation Instructions

### Evaluate a Checkpoint

```bash
# Evaluate best checkpoint
python scripts/eval.py \
    --checkpoint outputs/proposed/best_model.pt \
    --config configs/proposed.yaml

# With error analysis
python scripts/eval.py \
    --checkpoint outputs/proposed/best_model.pt \
    --config configs/proposed.yaml \
    --error_analysis
```

### Generate Comparison Report

```bash
# Run make_report.py after all experiments complete
python scripts/make_report.py --results_dir outputs/
```

### Compare Metrics Across Experiments

```bash
# Manual comparison
cat outputs/baseline/metrics.json
cat outputs/proposed/metrics.json
cat outputs/ablation_spatial_only/metrics.json
cat outputs/ablation_relation_only/metrics.json
```

### Key Metrics to Compare

1. **VQA Accuracy** (Primary): Official VQAv2 metric using `min(count/3, 1)` formula
2. **Exact Match**: Strict string equality
3. **Normalized Match**: After text normalization
4. **Per-Question-Type Accuracy**: Breakdown by yes/no, counting, spatial, etc.

---

## E. Interpreting Results

### Primary Metrics Explanation

| Metric | Description | Good Value |
|--------|-------------|------------|
| VQA Accuracy | Official VQAv2 metric (multi-annotator agreement) | > 60% |
| Exact Match | Prediction == Ground Truth (strict) | > 50% |
| Normalized Match | After lowercasing, article removal, etc. | > 55% |

### Per-Question-Type Breakdown

| Question Type | What it Tests | Scene Reasoning Should Help |
|---------------|---------------|:---------------------------:|
| Yes/No | Binary classification | Slightly |
| Counting | Numerical reasoning | ✓ Moderately |
| Color | Attribute recognition | Slightly |
| Spatial | Position/location | ✓ Significantly |
| What/Who | Object/person identification | Slightly |

### Good Signs ✅

- `proposed > baseline` on overall VQA accuracy
- Largest improvement on **spatial** and **counting** questions
- `spatial_only > baseline` (spatial encoding contributes)
- `proposed > spatial_only` (relation attention adds value)

### Red Flags 🚩

- `baseline > proposed` (something is wrong)
- `ablation_no_spatial ≈ proposed` (components not contributing)
- Large gap between train and val accuracy (overfitting)
- Training loss not decreasing (learning rate issue)

---

## F. Expected Training Times

### Colab T4 GPU (16GB)

| Configuration | ~Time/Epoch | Total (3 epochs) | Memory |
|---------------|-------------|------------------|--------|
| baseline | 2.5 hours | 7.5 hours | ~12 GB |
| proposed | 3.0 hours | 9.0 hours | ~14 GB |
| ablation_spatial_only | 2.8 hours | 8.4 hours | ~13 GB |
| ablation_relation_only | 2.8 hours | 8.4 hours | ~13 GB |
| ablation_no_spatial | 2.6 hours | 7.8 hours | ~12 GB |

### Notes

- Times are approximate for VQAv2 full training set (~440K samples)
- Use `max_train_samples` to reduce for faster iteration
- Enable `gradient_checkpointing` if OOM occurs

---

## G. Troubleshooting

### OOM (Out of Memory) Errors

```yaml
# In config, add:
training:
  gradient_checkpointing: true
  batch_size: 1
  gradient_accumulation_steps: 16  # Increase if needed
```

### Slow Training

```bash
# Check if using GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reduce num_workers if I/O bound
python scripts/train.py --config configs/proposed.yaml \
    --data.num_workers 0
```

### Config Validation

```bash
# Test that config loads correctly
python -c "
from src.utils.config import load_config
config = load_config('configs/proposed.yaml')
print(f'Scene Reasoning: {config.model.use_scene_reasoning}')
print(f'Spatial: {config.model.use_spatial_encoding}')
print(f'Relation: {config.model.use_relation_attention}')
"
```

### Check Training Hyperparameters Match

```bash
# Run ablation config tests
python tests/test_ablation_configs.py
```

---

## H. For Thesis Writing

### Experiment Results Table (Markdown)

```markdown
| Configuration | Spatial | Relation | VQA Acc (%) | Exact (%) | Δ Baseline |
|---------------|:-------:|:--------:|:-----------:|:---------:|:----------:|
| Baseline | - | - | XX.XX | XX.XX | - |
| Spatial Only | ✓ | ✗ | XX.XX | XX.XX | +X.XX |
| Relation Only | ✗ | ✓ | XX.XX | XX.XX | +X.XX |
| Proposed | ✓ | ✓ | XX.XX | XX.XX | +X.XX |
```

### LaTeX Table Code

```latex
\begin{table}[h]
\centering
\caption{Component-wise Ablation Analysis on VQAv2}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Configuration & Spatial & Relation & VQA Acc (\%) & Exact (\%) & $\Delta$ Baseline \\
\midrule
Baseline & - & - & XX.XX & XX.XX & - \\
Spatial Only & \checkmark & - & XX.XX & XX.XX & +X.XX \\
Relation Only & - & \checkmark & XX.XX & XX.XX & +X.XX \\
\textbf{Proposed} & \checkmark & \checkmark & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{+X.XX} \\
\bottomrule
\end{tabular}
\end{table}
```

### How to Fill In Results

1. Run all experiments to completion
2. Extract metrics from `outputs/<experiment_name>/metrics.json`
3. Calculate Δ Baseline = (Experiment VQA Acc) - (Baseline VQA Acc)
4. Bold the best values

---

## References

- **VQAv2**: Goyal et al., "Making the V in VQA Matter", CVPR 2017
- **BLIP-2**: Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training", ICML 2023
- **Official VQA Evaluation**: https://visualqa.org/evaluation

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│                    ABLATION STUDY QUICK REF                     │
├────────────────────────────────────────────────────────────────┤
│ baseline.yaml         → BLIP-2 only (no scene reasoning)       │
│ proposed.yaml         → Full model (spatial + relation)        │
│ ablation_spatial_only → Spatial encoding only                  │
│ ablation_relation_only→ Relation attention only (no bias)      │
│ ablation_no_spatial   → Architecture only (neither component)  │
├────────────────────────────────────────────────────────────────┤
│ Expected: proposed > spatial_only > relation_only ≈ baseline   │
├────────────────────────────────────────────────────────────────┤
│ Quick Test: python scripts/train.py --config configs/X --smoke_test │
│ Mac Dev:    python scripts/train.py --config configs/mac_dev.yaml   │
│ Full Run:   python scripts/train.py --config configs/proposed.yaml  │
└────────────────────────────────────────────────────────────────┘
```
