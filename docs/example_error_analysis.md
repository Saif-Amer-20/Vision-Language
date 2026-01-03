# VQA Error Analysis Report

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1000 |
| Correct | 653 (65.3%) |
| Incorrect | 298 |
| Close Misses | 49 |
| Overall Accuracy | 65.30% |
| VQA Accuracy | 71.24% |

## 🔍 Error Type Breakdown

| Error Type | Count | Percentage |
|------------|-------|------------|
| exact_wrong | 187 | 53.9% |
| close_miss | 49 | 14.1% |
| type_mismatch | 78 | 22.5% |
| partial_correct | 33 | 9.5% |

## 📝 Performance by Question Type

| Question Type | Count | Accuracy | VQA Accuracy |
|---------------|-------|----------|--------------|
| yes/no | 412 | 82.5% | 85.2% |
| what | 298 | 54.7% | 61.3% |
| counting | 112 | 48.2% | 52.8% |
| color | 89 | 71.9% | 76.4% |
| spatial | 54 | 42.6% | 48.1% |
| who | 21 | 61.9% | 67.2% |
| other | 14 | 57.1% | 63.5% |

## 📏 Performance by Answer Length

| Answer Length | Count | Accuracy |
|---------------|-------|----------|
| 1_word | 782 | 68.5% |
| 2_words | 156 | 55.1% |
| 3+_words | 62 | 41.9% |

## 🔄 Common Confusions

| Prediction | Ground Truth | Count |
|------------|--------------|-------|
| yes | no | 23 |
| no | yes | 19 |
| 2 | 3 | 12 |
| 1 | 2 | 11 |
| white | black | 8 |
| red | blue | 7 |
| man | woman | 6 |
| left | right | 6 |
| cat | dog | 5 |
| tennis | baseball | 5 |

## ❌ Most Frequent Wrong Predictions

| Prediction | Count |
|------------|-------|
| yes | 45 |
| no | 38 |
| 2 | 21 |
| white | 18 |
| 1 | 15 |
| red | 14 |
| black | 12 |
| blue | 11 |
| 3 | 10 |
| man | 9 |

## 🎯 Most Commonly Missed Answers

| Answer | Count |
|--------|-------|
| no | 42 |
| yes | 35 |
| 3 | 19 |
| 2 | 17 |
| black | 15 |
| white | 13 |
| blue | 12 |
| woman | 11 |
| right | 10 |
| 4 | 9 |

## 🔬 Top 10 Error Examples

Sorted by VQA accuracy (lowest first):

### Example 1
- **Question ID**: 458293
- **Question**: How many people are standing on the beach?
- **Prediction**: `2`
- **Ground Truths**: `5`, `5`, `5`, `4`, `5`
- **VQA Accuracy**: 0.0%
- **Error Type**: exact_wrong
- **Question Type**: counting

### Example 2
- **Question ID**: 129847
- **Question**: What color is the car in the foreground?
- **Prediction**: `white`
- **Ground Truths**: `silver`, `gray`, `grey`, `silver`, `gray`
- **VQA Accuracy**: 0.0%
- **Error Type**: close_miss
- **Question Type**: color

### Example 3
- **Question ID**: 384756
- **Question**: Is this a professional kitchen?
- **Prediction**: `yes`
- **Ground Truths**: `no`, `no`, `no`, `no`, `no`
- **VQA Accuracy**: 0.0%
- **Error Type**: type_mismatch
- **Question Type**: yes/no

### Example 4
- **Question ID**: 927183
- **Question**: What sport is being played?
- **Prediction**: `soccer`
- **Ground Truths**: `football`, `football`, `soccer`, `football`, `football`
- **VQA Accuracy**: 33.3%
- **Error Type**: partial_correct
- **Question Type**: what

### Example 5
- **Question ID**: 573921
- **Question**: Where is the dog sitting?
- **Prediction**: `couch`
- **Ground Truths**: `sofa`, `couch`, `on the sofa`, `sofa`, `couch`
- **VQA Accuracy**: 66.7%
- **Error Type**: partial_correct
- **Question Type**: spatial

### Example 6
- **Question ID**: 847362
- **Question**: What time of day is it?
- **Prediction**: `afternoon`
- **Ground Truths**: `daytime`, `day`, `afternoon`, `day`, `daytime`
- **VQA Accuracy**: 33.3%
- **Error Type**: partial_correct
- **Question Type**: what

### Example 7
- **Question ID**: 638291
- **Question**: How many birds are in the sky?
- **Prediction**: `3`
- **Ground Truths**: `4`, `4`, `5`, `4`, `4`
- **VQA Accuracy**: 0.0%
- **Error Type**: exact_wrong
- **Question Type**: counting

### Example 8
- **Question ID**: 192847
- **Question**: What is the man wearing on his head?
- **Prediction**: `cap`
- **Ground Truths**: `hat`, `baseball cap`, `cap`, `hat`, `cap`
- **VQA Accuracy**: 66.7%
- **Error Type**: partial_correct
- **Question Type**: what

### Example 9
- **Question ID**: 746382
- **Question**: Is the woman smiling?
- **Prediction**: `no`
- **Ground Truths**: `yes`, `yes`, `no`, `yes`, `yes`
- **VQA Accuracy**: 33.3%
- **Error Type**: type_mismatch
- **Question Type**: yes/no

### Example 10
- **Question ID**: 283746
- **Question**: What animal is in the picture?
- **Prediction**: `cat`
- **Ground Truths**: `dog`, `puppy`, `dog`, `dog`, `dog`
- **VQA Accuracy**: 0.0%
- **Error Type**: exact_wrong
- **Question Type**: what

---

## Generated Files

After running error analysis, you will have:

| File | Description |
|------|-------------|
| `error_analysis.md` | This markdown report |
| `error_analysis.json` | Full analysis data in JSON format |
| `top_errors.csv` | Top error examples in CSV format |
| `error_types.png` | Bar chart of error type distribution |
| `question_type_accuracy.png` | Accuracy by question type |
| `answer_length_accuracy.png` | Accuracy by answer length |

## Usage

```bash
# Run evaluation with error analysis
python scripts/eval.py \
    --checkpoint outputs/proposed/best_model.pt \
    --config configs/proposed.yaml \
    --error_analysis

# Or analyze existing predictions file
python -c "
from src.evaluation.error_analysis import analyze_predictions_file
analyze_predictions_file('outputs/predictions.json', 'outputs/analysis/')
"
```
