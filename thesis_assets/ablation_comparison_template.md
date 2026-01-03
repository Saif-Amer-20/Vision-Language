# Ablation Study Results Template

This document provides ready-to-use templates for presenting ablation study
results in a thesis. Fill in the XX.XX placeholders with actual values.

---

## Table 1: Component-wise Ablation Analysis

**Caption**: Ablation study results on VQAv2 validation set. We evaluate the
contribution of each Scene Reasoning Module component: Spatial Position
Encoding and Relation-Aware Self-Attention.

| Configuration | Spatial Enc | Relation Att | VQA Acc (%) | Exact Match (%) | Normalized Match (%) | Δ vs Baseline |
|---------------|:-----------:|:------------:|:-----------:|:---------------:|:--------------------:|:-------------:|
| Baseline (BLIP-2) | - | - | XX.XX | XX.XX | XX.XX | - |
| Spatial Only | ✓ | ✗ | XX.XX | XX.XX | XX.XX | +X.XX |
| Relation Only | ✗ | ✓ | XX.XX | XX.XX | XX.XX | +X.XX |
| **Proposed (Full)** | ✓ | ✓ | **XX.XX** | **XX.XX** | **XX.XX** | **+X.XX** |

**Notes**:
- VQA Accuracy uses the official VQAv2 formula: `min(count/3, 1)` where count is the number of annotators who provided the predicted answer
- Baseline: Standard BLIP-2 without Scene Reasoning Module
- All models trained for 3 epochs with identical hyperparameters

---

## Table 2: Per-Question-Type Performance

**Caption**: Performance breakdown by question type. Scene Reasoning Module
shows largest improvements on spatial reasoning and counting questions.

| Question Type | Baseline | Spatial Only | Relation Only | Proposed | Best Δ |
|---------------|:--------:|:------------:|:-------------:|:--------:|:------:|
| What | XX.XX% | XX.XX% | XX.XX% | XX.XX% | +X.XX |
| Yes/No | XX.XX% | XX.XX% | XX.XX% | XX.XX% | +X.XX |
| Counting | XX.XX% | XX.XX% | XX.XX% | XX.XX% | +X.XX |
| **Spatial** | XX.XX% | XX.XX% | XX.XX% | **XX.XX%** | **+X.XX** |
| Color | XX.XX% | XX.XX% | XX.XX% | XX.XX% | +X.XX |
| Who | XX.XX% | XX.XX% | XX.XX% | XX.XX% | +X.XX |
| Other | XX.XX% | XX.XX% | XX.XX% | XX.XX% | +X.XX |

**Expected Pattern**: Largest improvements on Spatial and Counting questions,
moderate improvements on What/Who, minimal change on Yes/No.

---

## Key Findings

### Overall Performance Summary

> The proposed Scene Reasoning Module achieves **XX.XX% VQA accuracy**,
> representing a **+X.XX percentage point improvement** over the BLIP-2
> baseline (XX.XX%). This confirms our hypothesis that explicit spatial
> reasoning enhances visual question answering.

### Component Contribution Analysis

| Component | Isolated Contribution | With Other Component | Synergy |
|-----------|:---------------------:|:--------------------:|:-------:|
| Spatial Position Encoding | +X.XX% | +X.XX% (in Proposed) | [Additive/Synergistic] |
| Relation-Aware Attention | +X.XX% | +X.XX% (in Proposed) | [Additive/Synergistic] |

**Interpretation**:
- Spatial Encoding alone contributes +X.XX% improvement
- Relation Attention alone contributes +X.XX% improvement  
- Combined improvement is +X.XX%, indicating [additive/synergistic] effect

### Per-Question-Type Analysis

> As hypothesized, the Scene Reasoning Module provides the largest
> improvements on questions requiring spatial understanding:
>
> - **Spatial questions**: +X.XX% improvement (XX.XX% → XX.XX%)
> - **Counting questions**: +X.XX% improvement (XX.XX% → XX.XX%)
> - **What questions**: +X.XX% improvement (XX.XX% → XX.XX%)
>
> These results support our thesis that explicit 2D spatial encoding
> and relation-aware attention enhance the model's spatial reasoning
> capabilities.

### Hypothesis Validation

| Hypothesis | Status | Evidence |
|------------|:------:|----------|
| H1: Scene Reasoning improves overall VQA accuracy | [✓/✗] | +X.XX% improvement |
| H2: Spatial encoding contributes to spatial questions | [✓/✗] | +X.XX% on spatial type |
| H3: Relation attention enhances object relationships | [✓/✗] | [Evidence description] |
| H4: Components have synergistic effects | [✓/✗] | Combined > sum of parts: [Yes/No] |

---

## Statistical Significance

### t-Test Results

| Comparison | t-statistic | p-value | Significant (α=0.05) |
|------------|:-----------:|:-------:|:--------------------:|
| Proposed vs Baseline | X.XXX | 0.XXX | [Yes/No] |
| Spatial Only vs Baseline | X.XXX | 0.XXX | [Yes/No] |
| Relation Only vs Baseline | X.XXX | 0.XXX | [Yes/No] |
| Proposed vs Spatial Only | X.XXX | 0.XXX | [Yes/No] |

### Effect Size (Cohen's d)

| Comparison | Cohen's d | Interpretation |
|------------|:---------:|----------------|
| Proposed vs Baseline | X.XX | [Small/Medium/Large] |
| Spatial Only vs Baseline | X.XX | [Small/Medium/Large] |
| Relation Only vs Baseline | X.XX | [Small/Medium/Large] |

### Confidence Intervals (95%)

| Model | VQA Accuracy | 95% CI |
|-------|:------------:|:------:|
| Baseline | XX.XX% | [XX.XX, XX.XX] |
| Spatial Only | XX.XX% | [XX.XX, XX.XX] |
| Relation Only | XX.XX% | [XX.XX, XX.XX] |
| Proposed | XX.XX% | [XX.XX, XX.XX] |

---

## Training Metrics

### Training Summary

| Configuration | Epochs | Final Train Loss | Final Val Loss | Training Time | GPU Memory |
|---------------|:------:|:----------------:|:--------------:|:-------------:|:----------:|
| Baseline | 3 | X.XXX | X.XXX | X.X hours | XX GB |
| Spatial Only | 3 | X.XXX | X.XXX | X.X hours | XX GB |
| Relation Only | 3 | X.XXX | X.XXX | X.X hours | XX GB |
| Proposed | 3 | X.XXX | X.XXX | X.X hours | XX GB |

### Trainable Parameters

| Configuration | Total Params | Trainable Params | % Trainable |
|---------------|:------------:|:----------------:|:-----------:|
| Baseline | ~3.8B | ~XXM | X.X% |
| Proposed | ~3.8B | ~XXM | X.X% |
| Scene Module Only | N/A | ~XXM | 100% |

---

## Error Analysis Highlights

### Common Error Patterns

| Error Type | Baseline Count | Proposed Count | Δ Count | % Reduction |
|------------|:--------------:|:--------------:|:-------:|:-----------:|
| Spatial Confusion | XXX | XXX | -XXX | XX% |
| Counting Errors | XXX | XXX | -XXX | XX% |
| Color Mistakes | XXX | XXX | -XXX | XX% |
| Yes/No Flips | XXX | XXX | -XXX | XX% |

### Example Improvements

#### Example 1: Spatial Reasoning
- **Question**: "Where is the dog sitting?"
- **Image**: [Description]
- **Baseline**: "floor" (incorrect)
- **Proposed**: "couch" (correct)
- **Analysis**: Scene Reasoning Module correctly identifies spatial relationship

#### Example 2: Counting
- **Question**: "How many people are in the picture?"
- **Image**: [Description]
- **Baseline**: "2" (incorrect)
- **Proposed**: "4" (correct)
- **Analysis**: 2D position encoding helps distinguish overlapping objects

#### Example 3: Relation Understanding
- **Question**: "What is to the left of the car?"
- **Image**: [Description]
- **Baseline**: "building" (incorrect)
- **Proposed**: "tree" (correct)
- **Analysis**: Relation-aware attention captures directional relationships

### Failure Case Analysis

| Failure Type | Count | % of Errors | Possible Cause |
|--------------|:-----:|:-----------:|----------------|
| [Type 1] | XXX | XX% | [Analysis] |
| [Type 2] | XXX | XX% | [Analysis] |
| [Type 3] | XXX | XX% | [Analysis] |

---

## Computational Cost Analysis

### Training Cost Comparison

| Configuration | Time (hours) | GPU Hours | Estimated Cost* |
|---------------|:------------:|:---------:|:---------------:|
| Baseline | X.X | X.X | $X.XX |
| Proposed | X.X | X.X | $X.XX |
| Total Ablation Study | X.X | X.X | $X.XX |

*Based on Colab Pro rates or cloud GPU pricing

### Inference Speed

| Configuration | Samples/sec | Latency (ms/sample) | Overhead vs Baseline |
|---------------|:-----------:|:-------------------:|:--------------------:|
| Baseline | XX.X | XXX | - |
| Proposed | XX.X | XXX | +XX% |

---

## Discussion Templates

### Why Does the Scene Reasoning Module Work?

> The Scene Reasoning Module improves VQA performance through two
> complementary mechanisms:
>
> 1. **Spatial Position Encoding**: By injecting explicit 2D spatial
>    coordinates into patch features, the model gains awareness of
>    where objects are located in the image grid. This is particularly
>    beneficial for questions about relative positions ("left of",
>    "above", "next to").
>
> 2. **Relation-Aware Attention**: The relative position bias in
>    attention weights allows the model to weight nearby patches
>    differently from distant ones, enabling better modeling of
>    spatial relationships without explicit object detection.
>
> The synergistic effect of combining both components suggests that
> [spatial awareness + relational reasoning > sum of parts].

### Limitations

> While the Scene Reasoning Module demonstrates consistent improvements,
> several limitations should be noted:
>
> 1. **[Limitation 1]**: [Description and impact]
> 2. **[Limitation 2]**: [Description and impact]
> 3. **Computational overhead**: The proposed model requires approximately
>    X% more training time and X% more memory than baseline.
> 4. **Dataset specificity**: Results are validated on VQAv2; generalization
>    to other VQA datasets requires further investigation.

### Future Work

> Based on our ablation analysis, several directions merit further exploration:
>
> 1. **Dynamic spatial encoding**: Adapting position encoding based on
>    question type or image content
> 2. **Cross-attention integration**: Injecting spatial information into
>    the question-image interaction
> 3. **Scale to larger models**: Testing with BLIP-2 variants (OPT-6.7B)
> 4. **Multi-dataset evaluation**: Generalizing to GQA, VizWiz, OK-VQA

---

## LaTeX Code for Main Results Table

```latex
\begin{table*}[t]
\centering
\caption{Ablation study on VQAv2 validation set. We systematically evaluate 
the contribution of Spatial Position Encoding and Relation-Aware Attention 
in our Scene Reasoning Module. Best results in \textbf{bold}.}
\label{tab:ablation-results}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Configuration} & \textbf{Spatial} & \textbf{Relation} & 
\textbf{VQA Acc} & \textbf{Exact} & \textbf{Norm} & 
$\boldsymbol{\Delta}$ \textbf{Base} \\
\midrule
Baseline (BLIP-2) & - & - & XX.XX & XX.XX & XX.XX & - \\
+ Spatial Only & \checkmark & - & XX.XX & XX.XX & XX.XX & +X.XX \\
+ Relation Only & - & \checkmark & XX.XX & XX.XX & XX.XX & +X.XX \\
\textbf{Proposed (Full)} & \checkmark & \checkmark & 
\textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{+X.XX} \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## LaTeX Code for Question-Type Table

```latex
\begin{table}[t]
\centering
\caption{Per-question-type VQA accuracy (\%) on VQAv2 validation set.
Scene Reasoning Module shows largest improvements on spatial and counting
questions, supporting our hypothesis about enhanced spatial understanding.}
\label{tab:question-type-results}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Q-Type} & \textbf{Baseline} & \textbf{Spatial} & 
\textbf{Proposed} & $\boldsymbol{\Delta}$ \\
\midrule
What & XX.XX & XX.XX & XX.XX & +X.XX \\
Yes/No & XX.XX & XX.XX & XX.XX & +X.XX \\
Counting & XX.XX & XX.XX & \textbf{XX.XX} & +X.XX \\
\textbf{Spatial} & XX.XX & XX.XX & \textbf{XX.XX} & \textbf{+X.XX} \\
Color & XX.XX & XX.XX & XX.XX & +X.XX \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Checklist Before Submission

- [ ] All XX.XX placeholders replaced with actual values
- [ ] Statistical significance computed and reported
- [ ] Confidence intervals calculated
- [ ] Error analysis completed with examples
- [ ] Training curves plotted (separate figures)
- [ ] Attention visualizations included (separate figures)
- [ ] Computational costs accurately measured
- [ ] All claims supported by evidence in tables
- [ ] Limitations acknowledged
- [ ] Future work section completed
