# VLM-VQA Research Project

> **Vision-Language Model for Visual Question Answering with Scene Reasoning**  
> Master's Thesis Research Implementation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This project implements a **Scene Reasoning Module** that enhances BLIP-2's visual question answering capabilities through explicit spatial relationship modeling. The research investigates how explicit scene understanding can improve VQA performance, particularly for spatial and relational questions.

### Key Features

- **BLIP-2 Integration**: Leverages Salesforce's BLIP-2 (OPT-2.7B) as the base VLM
- **Scene Reasoning Module**: Custom module with:
  - 2D Spatial Position Encodings
  - Relation-Aware Self-Attention
  - Configurable for ablation studies
- **Mac-First + Colab-Train Workflow**: Develop locally, train on cloud
- **Comprehensive Evaluation**: VQA accuracy, exact/normalized match, error analysis
- **Thesis-Ready Reporting**: LaTeX tables, comparison reports, visualizations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BLIP-2 VQA Model                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Vision    │    │   Q-Former  │    │       LLM       │  │
│  │   Encoder   │───▶│  (Querying  │───▶│   (OPT-2.7B)    │  │
│  │  (ViT-G)    │    │ Transformer)│    │                 │  │
│  └─────────────┘    └──────┬──────┘    └─────────────────┘  │
│                            │                                │
│                     ┌──────▼──────┐                         │
│                     │   Scene     │  ◀── Proposed Module    │
│                     │  Reasoning  │      (Plug-in)          │
│                     │   Module    │                         │
│                     └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
VLM_Thesis/
├── configs/                    # YAML experiment configurations
│   ├── baseline.yaml          # BLIP-2 baseline
│   ├── proposed.yaml          # BLIP-2 + Scene Reasoning
│   ├── ablation_*.yaml        # Ablation study configs
│   ├── mac_dev.yaml           # Mac development config
│   └── smoke_test.yaml        # Quick validation
├── src/
│   ├── data/                  # VQA dataset loaders
│   │   ├── vqa_dataset.py     # HuggingFace VQAv2 loader
│   │   └── answer_utils.py    # Answer normalization
│   ├── models/                # Model implementations
│   │   ├── blip2_wrapper.py   # BLIP-2 wrapper with freeze controls
│   │   └── scene_reasoning.py # Scene Reasoning Module
│   ├── training/              # Training pipeline
│   │   ├── trainer.py         # VQATrainer with Accelerate
│   │   ├── losses.py          # VQA loss functions
│   │   └── schedulers.py      # LR schedulers
│   ├── evaluation/            # Evaluation pipeline
│   │   ├── evaluator.py       # VQA evaluator
│   │   ├── metrics.py         # VQA metrics
│   │   └── error_analysis.py  # Error categorization
│   └── utils/                 # Utilities
│       ├── config.py          # Config system with ExecutionProfile
│       ├── seed.py            # Reproducibility
│       ├── io_utils.py        # Checkpoint/JSON handling
│       └── logging_utils.py   # TensorBoard/W&B logging
├── scripts/
│   ├── train.py               # Training entry point
│   ├── eval.py                # Evaluation entry point
│   └── make_report.py         # Report generation
├── outputs/                   # Experiment outputs (gitignored)
├── thesis_assets/             # Thesis-ready materials
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VLM_Thesis.git
cd VLM_Thesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Smoke Test

```bash
# Verify everything works
python scripts/train.py --config configs/smoke_test.yaml
```

### Training

```bash
# Mac development (auto-limited to 10 steps)
python scripts/train.py --config configs/mac_dev.yaml

# Full training (on Colab/GPU)
python scripts/train.py --config configs/proposed.yaml
```

### Evaluation

```bash
python scripts/eval.py \
    --checkpoint outputs/proposed/best_model.pt \
    --config configs/proposed.yaml \
    --error_analysis
```

### Report Generation

```bash
python scripts/make_report.py --results_dir outputs/ --latex
```

## 💻 Execution Profiles

| Profile | Environment | Purpose | Limits |
|---------|-------------|---------|--------|
| `colab_train` | Colab GPU | Full training | Unlimited |
| `mac_dev` | Mac/Local | Development | ≤10 steps, ≤50 samples |
| `eval_only` | Any | Evaluation | No training |

**CRITICAL**: Never reduce BLIP-2 model size/architecture. Only runtime settings differ.

## 🔧 Configuration

All experiments are configured via YAML files. Key configuration sections:

```yaml
# configs/proposed.yaml
model:
  model_name: "Salesforce/blip2-opt-2.7b"
  use_scene_reasoning: true  # Enable/disable our module
  scene_num_layers: 2
  use_spatial_encoding: true
  use_relation_attention: true

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  num_epochs: 3
  fp16: true

runtime:
  execution_profile: "colab_train"
```

Override from CLI:

```bash
python scripts/train.py --config configs/baseline.yaml \
    --training.learning_rate 2e-5 \
    --model.scene_num_layers 4
```

## 📊 Experiments

### Baseline vs Proposed

```bash
# Train baseline (no Scene Reasoning)
python scripts/train.py --config configs/baseline.yaml

# Train proposed model
python scripts/train.py --config configs/proposed.yaml
```

### Ablation Studies

```bash
# Without spatial encoding
python scripts/train.py --config configs/ablation_no_spatial.yaml

# Without relation attention
python scripts/train.py --config configs/ablation_no_relation.yaml
```

## 📈 Results

Results are saved to `outputs/<experiment_name>/<timestamp>/`:

```
outputs/proposed/20240115_143022/
├── checkpoints/           # Model checkpoints
├── logs/                  # TensorBoard logs
├── results/
│   ├── final_results.json # Metrics summary
│   ├── predictions.json   # All predictions
│   └── error_analysis.md  # Error report
└── config.json            # Run configuration
```

View TensorBoard logs:

```bash
tensorboard --logdir outputs/
```

## 🔬 Research Notes

### Scene Reasoning Module

The Scene Reasoning Module enhances visual features with:

1. **Spatial Position Encoding**: Learnable 2D position embeddings that encode row/column positions
2. **Relation-Aware Attention**: Self-attention with relative position bias for modeling object relationships

### Ablation Hypotheses

- **H1**: Spatial encoding improves performance on "where" questions
- **H2**: Relation attention improves performance on comparative questions
- **H3**: Combined approach provides complementary benefits

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{yourname2024vlmvqa,
  title={Enhancing Visual Question Answering with Scene Reasoning},
  author={Your Name},
  school={Your University},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Salesforce BLIP-2](https://github.com/salesforce/LAVIS) for the base VLM
- [HuggingFace](https://huggingface.co/) for transformers and datasets
- [VQAv2 Dataset](https://visualqa.org/) for evaluation data
