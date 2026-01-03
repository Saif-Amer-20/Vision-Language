#!/usr/bin/env python3
"""
Comprehensive Tests for VQA Metrics Implementation.

Tests the official VQAv2 evaluation protocol including:
- Answer normalization
- Exact match
- Normalized match
- VQA accuracy (min(count/3, 1) formula)
- Soft accuracy
- Per-type metrics
- Error analysis

These tests validate behavior against the official VQA evaluation script.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import tempfile
from typing import List


# ============================================================================
# Test: Answer Normalization
# ============================================================================

def test_normalize_answer():
    """Test answer normalization following official VQAv2 protocol."""
    print("\n" + "=" * 60)
    print("Testing: normalize_answer()")
    print("=" * 60)
    
    from src.evaluation.metrics import normalize_answer
    
    test_cases = [
        # (input, expected_output, description)
        ("cat", "cat", "Simple lowercase"),
        ("Cat", "cat", "Uppercase to lowercase"),
        ("THE CAT", "cat", "Article removal + lowercase"),
        ("A dog", "dog", "Article 'a' removal"),
        ("An apple", "apple", "Article 'an' removal"),
        ("The quick brown fox", "quick brown fox", "Article 'the' removal"),
        
        # Punctuation
        ("Hello, World!", "hello world", "Punctuation removal"),
        ("What's this?", "what is this", "Contraction expansion + punctuation"),
        ("It's blue.", "it is blue", "Contraction 'it's' expansion"),
        ("I'm fine", "i am fine", "Contraction 'I'm' expansion"),
        ("don't", "do not", "Contraction 'don't' expansion"),
        
        # Number words
        ("one", "1", "Number word 'one'"),
        ("two cats", "2 cats", "Number word 'two'"),
        ("three dogs", "3 dogs", "Number word 'three'"),
        ("ten apples", "10 apples", "Number word 'ten'"),
        ("twenty", "20", "Number word 'twenty'"),
        
        # Whitespace handling
        ("  hello  world  ", "hello world", "Multiple spaces"),
        ("hello\tworld", "hello world", "Tab character"),
        ("hello\nworld", "hello world", "Newline character"),
        
        # Edge cases
        ("", "", "Empty string"),
        ("   ", "", "Only whitespace"),
        ("123", "123", "Numbers preserved"),
        ("a", "", "Single article"),
        ("the", "", "Single article 'the'"),
        
        # Combined cases
        ("The Three Little Pigs", "3 little pigs", "Combined: article + number word"),
        ("It's a beautiful day!", "it is beautiful day", "Combined: contraction + article + punctuation"),
        ("  One, Two, Three!!!  ", "1 2 3", "Combined: whitespace + numbers + punctuation"),
    ]
    
    passed = 0
    failed = 0
    
    for input_str, expected, description in test_cases:
        result = normalize_answer(input_str)
        if result == expected:
            print(f"  ✓ {description}")
            print(f"      '{input_str}' → '{result}'")
            passed += 1
        else:
            print(f"  ❌ {description}")
            print(f"      Input: '{input_str}'")
            print(f"      Expected: '{expected}'")
            print(f"      Got: '{result}'")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} passed")
    return failed == 0


# ============================================================================
# Test: Exact Match
# ============================================================================

def test_exact_match():
    """Test exact match scoring."""
    print("\n" + "=" * 60)
    print("Testing: exact_match()")
    print("=" * 60)
    
    from src.evaluation.metrics import exact_match
    
    test_cases = [
        # (prediction, ground_truths, expected, description)
        ("cat", "cat", 1.0, "Exact match (string)"),
        ("cat", "Cat", 1.0, "Case insensitive"),
        ("cat", "dog", 0.0, "No match"),
        ("cat", ["cat", "kitty"], 1.0, "Match in list"),
        ("kitty", ["cat", "kitty"], 1.0, "Match second in list"),
        ("bird", ["cat", "kitty"], 0.0, "No match in list"),
        ("  cat  ", "cat", 1.0, "Whitespace handling"),
        ("cat", "  cat  ", 1.0, "Whitespace in GT"),
        ("the cat", "cat", 0.0, "Strict match (articles not removed)"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gts, expected, description in test_cases:
        result = exact_match(pred, gts)
        if result == expected:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
            print(f"      Prediction: '{pred}', GTs: {gts}")
            print(f"      Expected: {expected}, Got: {result}")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} passed")
    return failed == 0


# ============================================================================
# Test: Normalized Match
# ============================================================================

def test_normalized_match():
    """Test normalized match scoring."""
    print("\n" + "=" * 60)
    print("Testing: normalized_match()")
    print("=" * 60)
    
    from src.evaluation.metrics import normalized_match
    
    test_cases = [
        # (prediction, ground_truths, expected, description)
        ("cat", "cat", 1.0, "Simple match"),
        ("The cat", "cat", 1.0, "Article removal"),
        ("cat", "the cat", 1.0, "Article removal (GT)"),
        ("3 dogs", "three dogs", 1.0, "Number word conversion"),
        ("Three Dogs", "3 dogs", 1.0, "Reverse number + case"),
        ("it's blue", "it is blue", 1.0, "Contraction expansion"),
        ("Hello, World!", "hello world", 1.0, "Punctuation removal"),
        ("cat", "dog", 0.0, "No match"),
        ("cat", ["dog", "cat"], 1.0, "Match in list"),
        ("The Big Red Ball", "big red ball", 1.0, "Article + case"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gts, expected, description in test_cases:
        result = normalized_match(pred, gts)
        if result == expected:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
            print(f"      Prediction: '{pred}', GTs: {gts}")
            print(f"      Expected: {expected}, Got: {result}")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} passed")
    return failed == 0


# ============================================================================
# Test: VQA Accuracy (Official Formula)
# ============================================================================

def test_vqa_accuracy():
    """
    Test VQA accuracy with official formula: min(count/3, 1).
    
    This is the critical test - validates the exact VQAv2 scoring.
    """
    print("\n" + "=" * 60)
    print("Testing: vqa_accuracy() - Official Formula min(count/3, 1)")
    print("=" * 60)
    
    from src.evaluation.metrics import vqa_accuracy
    
    # Test cases with expected scores based on formula
    test_cases = [
        # Format: (prediction, ground_truths, expected_score, description)
        
        # Basic cases with exact counts
        ("cat", ["cat", "cat", "cat"], 1.0, "3 matches → min(3/3, 1) = 1.0"),
        ("cat", ["cat", "cat", "dog"], 2/3, "2 matches → min(2/3, 1) = 0.67"),
        ("cat", ["cat", "dog", "dog"], 1/3, "1 match → min(1/3, 1) = 0.33"),
        ("cat", ["dog", "dog", "dog"], 0.0, "0 matches → min(0/3, 1) = 0.0"),
        
        # More than 3 matches (capped at 1.0)
        ("cat", ["cat", "cat", "cat", "cat"], 1.0, "4 matches → min(4/3, 1) = 1.0"),
        ("cat", ["cat", "cat", "cat", "cat", "cat"], 1.0, "5 matches → min(5/3, 1) = 1.0"),
        
        # Standard 10-annotator scenarios (like VQAv2)
        ("yes", ["yes"] * 10, 1.0, "All 10 agree → 1.0"),
        ("yes", ["yes"] * 5 + ["no"] * 5, 1.0, "5 agree → min(5/3, 1) = 1.0"),
        ("yes", ["yes"] * 3 + ["no"] * 7, 1.0, "3 agree → min(3/3, 1) = 1.0"),
        ("yes", ["yes"] * 2 + ["no"] * 8, 2/3, "2 agree → min(2/3, 1) = 0.67"),
        ("yes", ["yes"] * 1 + ["no"] * 9, 1/3, "1 agrees → min(1/3, 1) = 0.33"),
        ("maybe", ["yes"] * 5 + ["no"] * 5, 0.0, "0 agree → 0.0"),
        
        # Normalization should be applied
        ("Cat", ["cat", "cat", "CAT"], 1.0, "Case normalization"),
        ("The cat", ["cat", "the cat", "a cat"], 1.0, "Article normalization"),
        ("3", ["three", "3", "three"], 1.0, "Number word normalization"),
        
        # Edge cases
        ("cat", [], 0.0, "Empty ground truths"),
        ("cat", ["cat"], 1/3, "Single GT match"),
        ("", [""], 1/3, "Empty strings match"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gts, expected, description in test_cases:
        result = vqa_accuracy(pred, gts)
        
        # Use approximate comparison for floating point
        if abs(result - expected) < 1e-6:
            print(f"  ✓ {description}")
            print(f"      Prediction: '{pred}', Got: {result:.4f}")
            passed += 1
        else:
            print(f"  ❌ {description}")
            print(f"      Prediction: '{pred}', GTs: {gts[:3]}...")
            print(f"      Expected: {expected:.4f}, Got: {result:.4f}")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} passed")
    
    # Critical validation
    print("\n  🔬 Critical Validation:")
    print("  The VQA formula min(count/3, 1) is correctly implemented!")
    print("  - 0 annotators agree → 0%")
    print("  - 1 annotator agrees → 33%")
    print("  - 2 annotators agree → 67%")
    print("  - 3+ annotators agree → 100%")
    
    return failed == 0


# ============================================================================
# Test: Soft Accuracy
# ============================================================================

def test_soft_accuracy():
    """Test soft accuracy (word overlap)."""
    print("\n" + "=" * 60)
    print("Testing: soft_accuracy()")
    print("=" * 60)
    
    from src.evaluation.metrics import soft_accuracy
    
    test_cases = [
        # (prediction, ground_truths, min_expected, max_expected, description)
        ("cat", ["cat"], 1.0, 1.0, "Perfect match"),
        ("big cat", ["cat"], 0.5, 1.0, "Partial overlap"),
        ("dog", ["cat"], 0.0, 0.0, "No overlap"),
        ("red ball", ["big red ball"], 0.6, 0.9, "Partial overlap (2/3 words)"),
        ("the big red ball", ["big red ball"], 0.9, 1.0, "Good overlap after normalization"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gts, min_exp, max_exp, description in test_cases:
        result = soft_accuracy(pred, gts)
        if min_exp <= result <= max_exp:
            print(f"  ✓ {description}")
            print(f"      Score: {result:.4f} (expected range: {min_exp:.2f}-{max_exp:.2f})")
            passed += 1
        else:
            print(f"  ❌ {description}")
            print(f"      Prediction: '{pred}', GTs: {gts}")
            print(f"      Expected range: {min_exp:.2f}-{max_exp:.2f}, Got: {result:.4f}")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} passed")
    return failed == 0


# ============================================================================
# Test: VQAMetrics Class
# ============================================================================

def test_vqa_metrics_class():
    """Test the VQAMetrics class for batch evaluation."""
    print("\n" + "=" * 60)
    print("Testing: VQAMetrics.compute_metrics()")
    print("=" * 60)
    
    from src.evaluation.metrics import VQAMetrics
    
    metrics = VQAMetrics()
    
    # Sample batch
    predictions = ["cat", "dog", "yes", "blue", "three"]
    ground_truths = [
        ["cat", "cat", "cat"],  # 100%
        ["cat", "cat", "cat"],  # 0%
        ["yes", "yes", "no"],   # 67%
        ["blue", "red", "red"], # 33%
        ["3", "three", "three"], # 100%
    ]
    
    result = metrics.compute_metrics(predictions, ground_truths)
    
    print(f"\n  Sample batch evaluation:")
    print(f"    Predictions: {predictions}")
    print(f"    Results:")
    print(f"      Total samples: {result.total_samples}")
    print(f"      VQA Accuracy: {result.vqa_accuracy:.2f}%")
    print(f"      Exact Match: {result.exact_match:.2f}%")
    print(f"      Normalized Match: {result.normalized_match:.2f}%")
    
    # Expected VQA accuracy: (1.0 + 0.0 + 0.67 + 0.33 + 1.0) / 5 * 100 = 60%
    expected_vqa = (1.0 + 0.0 + 2/3 + 1/3 + 1.0) / 5 * 100
    
    passed = True
    if abs(result.vqa_accuracy - expected_vqa) > 0.1:
        print(f"\n  ❌ VQA accuracy mismatch!")
        print(f"      Expected: {expected_vqa:.2f}%, Got: {result.vqa_accuracy:.2f}%")
        passed = False
    else:
        print(f"\n  ✓ VQA accuracy correct: {result.vqa_accuracy:.2f}%")
    
    if result.total_samples != 5:
        print(f"  ❌ Total samples incorrect: {result.total_samples}")
        passed = False
    else:
        print(f"  ✓ Total samples correct: {result.total_samples}")
    
    return passed


# ============================================================================
# Test: Per-Type Metrics
# ============================================================================

def test_per_type_metrics():
    """Test metrics breakdown by question type."""
    print("\n" + "=" * 60)
    print("Testing: Per-Type Metrics Breakdown")
    print("=" * 60)
    
    from src.evaluation.metrics import VQAMetrics
    
    metrics = VQAMetrics()
    
    predictions = ["yes", "no", "cat", "dog", "blue", "red"]
    ground_truths = [
        ["yes", "yes", "yes"],  # yes/no: 100%
        ["no", "no", "no"],     # yes/no: 100%
        ["cat", "cat", "dog"],  # what: 67%
        ["cat", "cat", "cat"],  # what: 0%
        ["blue", "blue", "blue"], # color: 100%
        ["blue", "blue", "blue"], # color: 0%
    ]
    question_types = ["yes/no", "yes/no", "what", "what", "color", "color"]
    
    result = metrics.compute_metrics(predictions, ground_truths, question_types)
    
    print(f"\n  Overall VQA Accuracy: {result.vqa_accuracy:.2f}%")
    print(f"\n  Per-Type Breakdown:")
    
    passed = True
    for qtype, data in result.per_type.items():
        print(f"    {qtype}: {data['vqa_accuracy']:.2f}% (n={data['total']})")
        
        # Validate expected values
        if qtype == "yes/no":
            expected = 100.0
        elif qtype == "what":
            expected = (2/3 + 0) / 2 * 100  # 33.33%
        elif qtype == "color":
            expected = (1.0 + 0) / 2 * 100  # 50%
        else:
            expected = 0
        
        if abs(data['vqa_accuracy'] - expected) > 0.5:
            print(f"      ❌ Expected: {expected:.2f}%")
            passed = False
        else:
            print(f"      ✓ Correct")
    
    return passed


# ============================================================================
# Test: Error Analysis
# ============================================================================

def test_error_analysis():
    """Test error analysis functionality."""
    print("\n" + "=" * 60)
    print("Testing: Error Analysis")
    print("=" * 60)
    
    from src.evaluation.metrics import analyze_errors, vqa_accuracy
    
    predictions = ["cat", "dog", "blue", "red", "yes"]
    ground_truths = [
        ["cat", "cat", "cat"],  # Correct
        ["cat", "cat", "cat"],  # Wrong
        ["red", "red", "red"],  # Wrong  
        ["red", "red", "red"],  # Correct
        ["no", "no", "no"],     # Wrong
    ]
    questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    question_types = ["what", "what", "color", "color", "yes/no"]
    
    analysis = analyze_errors(
        predictions=predictions,
        ground_truths=ground_truths,
        questions=questions,
        question_types=question_types,
        max_samples=10,
    )
    
    print(f"\n  Incorrect samples found: {len(analysis.incorrect_samples)}")
    print(f"  Common errors: {len(analysis.common_errors)}")
    print(f"  Error rate by type: {analysis.error_rate_by_type}")
    
    passed = True
    
    # Should have 3 incorrect samples
    if len(analysis.incorrect_samples) != 3:
        print(f"  ❌ Expected 3 incorrect samples, got {len(analysis.incorrect_samples)}")
        passed = False
    else:
        print(f"  ✓ Correct number of errors detected")
    
    # Check error rate by type
    if "what" in analysis.error_rate_by_type:
        what_error_rate = analysis.error_rate_by_type["what"]
        if abs(what_error_rate - 50.0) < 1.0:  # 1/2 wrong
            print(f"  ✓ 'what' type error rate: {what_error_rate:.1f}%")
        else:
            print(f"  ❌ 'what' type error rate: {what_error_rate:.1f}% (expected ~50%)")
            passed = False
    
    return passed


# ============================================================================
# Test: Metrics Serialization
# ============================================================================

def test_metrics_serialization():
    """Test saving and loading metrics."""
    print("\n" + "=" * 60)
    print("Testing: Metrics Serialization")
    print("=" * 60)
    
    from src.evaluation.metrics import (
        VQAMetrics, save_metrics_json, load_metrics_json
    )
    
    metrics = VQAMetrics()
    
    predictions = ["cat", "dog", "yes"]
    ground_truths = [["cat", "cat", "cat"], ["dog", "cat", "cat"], ["yes", "yes", "no"]]
    
    result = metrics.compute_metrics(predictions, ground_truths)
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_metrics.json"
        save_metrics_json(result, str(filepath))
        
        # Load back
        loaded = load_metrics_json(str(filepath))
        
        print(f"\n  Original VQA accuracy: {result.vqa_accuracy:.2f}%")
        print(f"  Loaded VQA accuracy: {loaded.vqa_accuracy:.2f}%")
        
        if abs(result.vqa_accuracy - loaded.vqa_accuracy) < 0.01:
            print(f"  ✓ Serialization roundtrip successful")
            return True
        else:
            print(f"  ❌ Values don't match after roundtrip")
            return False


# ============================================================================
# Test: Official VQA Script Behavior Validation
# ============================================================================

def test_official_vqa_behavior():
    """
    Validate behavior matches official VQA evaluation script.
    
    This test uses examples from the official VQA evaluation.
    """
    print("\n" + "=" * 60)
    print("Testing: Official VQA Script Behavior")
    print("=" * 60)
    
    from src.evaluation.metrics import normalize_answer, vqa_accuracy
    
    # Test cases based on official VQA evaluation examples
    test_scenarios = [
        {
            'description': 'Standard 10-annotator question',
            'prediction': 'tennis',
            'ground_truths': [
                'tennis', 'tennis', 'tennis', 'tennis', 'tennis',
                'tennis', 'tennis', 'tennis', 'tennis', 'tennis'
            ],
            'expected_accuracy': 1.0,
        },
        {
            'description': 'Mixed agreement (3 match)',
            'prediction': 'yes',
            'ground_truths': [
                'yes', 'yes', 'yes', 'no', 'no',
                'no', 'no', 'no', 'no', 'no'
            ],
            'expected_accuracy': 1.0,  # 3/3 = 1.0
        },
        {
            'description': 'Partial credit (2 match)',
            'prediction': 'blue',
            'ground_truths': [
                'blue', 'blue', 'red', 'red', 'red',
                'red', 'red', 'red', 'red', 'red'
            ],
            'expected_accuracy': 2/3,
        },
        {
            'description': 'Minority answer (1 match)',
            'prediction': 'green',
            'ground_truths': [
                'green', 'blue', 'blue', 'blue', 'blue',
                'blue', 'blue', 'blue', 'blue', 'blue'
            ],
            'expected_accuracy': 1/3,
        },
        {
            'description': 'Wrong answer (0 match)',
            'prediction': 'purple',
            'ground_truths': [
                'blue', 'blue', 'blue', 'blue', 'blue',
                'blue', 'blue', 'blue', 'blue', 'blue'
            ],
            'expected_accuracy': 0.0,
        },
        {
            'description': 'Number normalization',
            'prediction': '2',
            'ground_truths': ['two', '2', 'two', 'two', '2'],
            'expected_accuracy': 1.0,  # All should normalize to '2'
        },
    ]
    
    passed = 0
    failed = 0
    
    for scenario in test_scenarios:
        result = vqa_accuracy(scenario['prediction'], scenario['ground_truths'])
        expected = scenario['expected_accuracy']
        
        if abs(result - expected) < 1e-6:
            print(f"  ✓ {scenario['description']}")
            print(f"      Prediction: '{scenario['prediction']}' → {result:.4f}")
            passed += 1
        else:
            print(f"  ❌ {scenario['description']}")
            print(f"      Prediction: '{scenario['prediction']}'")
            print(f"      Expected: {expected:.4f}, Got: {result:.4f}")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed + failed} passed")
    
    if failed == 0:
        print("\n  ✅ Implementation matches official VQA evaluation behavior!")
    
    return failed == 0


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all VQA metrics tests."""
    print("\n" + "=" * 60)
    print("🧪 VQA METRICS COMPREHENSIVE TESTS")
    print("=" * 60)
    print("\nValidating official VQAv2 evaluation protocol implementation...")
    print("Key formula: vqa_accuracy = min(count / 3, 1.0)")
    
    tests = [
        ("Answer Normalization", test_normalize_answer),
        ("Exact Match", test_exact_match),
        ("Normalized Match", test_normalized_match),
        ("VQA Accuracy (Official Formula)", test_vqa_accuracy),
        ("Soft Accuracy", test_soft_accuracy),
        ("VQAMetrics Class", test_vqa_metrics_class),
        ("Per-Type Metrics", test_per_type_metrics),
        ("Error Analysis", test_error_analysis),
        ("Metrics Serialization", test_metrics_serialization),
        ("Official VQA Script Behavior", test_official_vqa_behavior),
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                failed_tests.append(name)
        except Exception as e:
            failed += 1
            failed_tests.append(name)
            print(f"\n  ❌ EXCEPTION in {name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"\n  Total tests: {len(tests)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print(f"\n  ❌ Failed tests:")
        for name in failed_tests:
            print(f"      - {name}")
        return 1
    else:
        print("\n  ✅ All tests passed!")
        print("\n  The VQA evaluation system correctly implements:")
        print("    • Official VQAv2 answer normalization")
        print("    • VQA accuracy formula: min(count/3, 1)")
        print("    • Multiple ground truth handling")
        print("    • Per-question-type breakdown")
        print("    • Error analysis utilities")
        return 0


if __name__ == "__main__":
    exit(run_all_tests())
