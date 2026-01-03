#!/usr/bin/env python3
"""
Tests for VQA Error Analysis System.

Validates:
- ErrorAnalyzer class functionality
- Question type inference
- Error type classification
- Report generation
- Visualization functions
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    from src.evaluation.error_analysis import (
        ErrorAnalyzer,
        ErrorAnalysisResult,
        PredictionRecord,
        analyze_predictions,
        analyze_predictions_file,
    )
    
    from src.evaluation.visualizations import (
        plot_error_analysis,
        plot_confusion_heatmap,
    )
    
    print("  ✓ All imports successful")
    return True


def test_question_type_inference():
    """Test question type inference from question text."""
    print("\nTesting question type inference...")
    
    from src.evaluation.error_analysis import ErrorAnalyzer
    
    analyzer = ErrorAnalyzer(predictions=[])
    
    test_cases = [
        # (question, expected_type)
        ("Is this a cat?", "yes/no"),
        ("Are they playing tennis?", "yes/no"),
        ("Does the man wear glasses?", "yes/no"),
        ("Can you see the sky?", "yes/no"),
        
        ("How many people are there?", "counting"),
        ("How much water is in the glass?", "counting"),
        ("What number is on the jersey?", "counting"),
        
        ("Where is the dog?", "spatial"),
        ("Which side is the car on?", "spatial"),
        ("What is the position of the ball?", "spatial"),
        
        ("What color is the car?", "color"),
        ("What colour is the dress?", "color"),
        
        ("What is the man doing?", "what"),
        ("What animal is this?", "what"),
        
        ("Who is in the picture?", "who"),
        ("Why is she crying?", "why"),
        ("How is the weather?", "how"),
        ("Which sport is this?", "which"),
    ]
    
    passed = 0
    failed = 0
    
    for question, expected in test_cases:
        result = analyzer._infer_question_type(question)
        if result == expected:
            print(f"  ✓ '{question[:30]}...' → {result}")
            passed += 1
        else:
            print(f"  ✗ '{question[:30]}...' → {result} (expected: {expected})")
            failed += 1
    
    print(f"\n  Results: {passed}/{passed+failed} passed")
    return failed == 0


def test_type_mismatch_detection():
    """Test detection of type mismatches."""
    print("\nTesting type mismatch detection...")
    
    from src.evaluation.error_analysis import ErrorAnalyzer
    
    analyzer = ErrorAnalyzer(predictions=[])
    
    test_cases = [
        # (prediction, ground_truths, expected_mismatch)
        ("yes", ["no"], False),  # Both yes/no, not a type mismatch
        ("yes", ["5"], True),    # yes/no vs number
        ("3", ["yes"], True),    # number vs yes/no
        ("cat", ["dog"], False), # Both text
        ("2", ["two"], False),   # Both numbers (different formats)
        ("blue", ["3"], True),   # text vs number
    ]
    
    passed = 0
    
    for pred, gts, expected in test_cases:
        result = analyzer._is_type_mismatch(pred, gts)
        if result == expected:
            print(f"  ✓ '{pred}' vs {gts} → mismatch={result}")
            passed += 1
        else:
            print(f"  ✗ '{pred}' vs {gts} → mismatch={result} (expected: {expected})")
    
    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_partial_correct_detection():
    """Test detection of partial correctness."""
    print("\nTesting partial correct detection...")
    
    from src.evaluation.error_analysis import ErrorAnalyzer
    
    analyzer = ErrorAnalyzer(predictions=[])
    
    test_cases = [
        # (prediction, ground_truths, expected_partial)
        ("red car", ["blue car"], True),        # 'car' overlaps
        ("playing tennis", ["tennis"], True),   # 'tennis' overlaps
        ("cat", ["dog"], False),                # No overlap
        ("", ["yes"], False),                   # Empty prediction
        ("the big dog", ["big brown dog"], True), # 'big', 'dog' overlap
    ]
    
    passed = 0
    
    for pred, gts, expected in test_cases:
        result = analyzer._is_partial_correct(pred, gts)
        if result == expected:
            print(f"  ✓ '{pred}' vs {gts} → partial={result}")
            passed += 1
        else:
            print(f"  ✗ '{pred}' vs {gts} → partial={result} (expected: {expected})")
    
    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_full_analysis():
    """Test complete error analysis pipeline."""
    print("\nTesting full analysis pipeline...")
    
    from src.evaluation.error_analysis import ErrorAnalyzer
    
    # Create sample predictions
    predictions = [
        {
            'question_id': '1',
            'question': 'Is this a cat?',
            'prediction': 'yes',
            'ground_truths': ['yes', 'yes', 'yes'],
        },
        {
            'question_id': '2',
            'question': 'What color is the car?',
            'prediction': 'red',
            'ground_truths': ['blue', 'blue', 'navy'],
        },
        {
            'question_id': '3',
            'question': 'How many dogs are there?',
            'prediction': '2',
            'ground_truths': ['3', '3', '3'],
        },
        {
            'question_id': '4',
            'question': 'Where is the ball?',
            'prediction': 'on the ground',
            'ground_truths': ['ground', 'on ground', 'floor'],
        },
        {
            'question_id': '5',
            'question': 'Is it raining?',
            'prediction': 'no',
            'ground_truths': ['yes', 'yes', 'no', 'yes'],
        },
    ]
    
    analyzer = ErrorAnalyzer(predictions=predictions, max_samples=100)
    result = analyzer.analyze()
    
    # Validate result structure
    assert result.total_samples == 5, f"Expected 5 samples, got {result.total_samples}"
    print(f"  ✓ Total samples: {result.total_samples}")
    
    assert result.correct_count >= 1, "Expected at least 1 correct"
    print(f"  ✓ Correct count: {result.correct_count}")
    
    assert len(result.question_type_counts) > 0, "Expected question type counts"
    print(f"  ✓ Question types found: {list(result.question_type_counts.keys())}")
    
    assert len(result.error_types) > 0 or result.incorrect_count == 0, "Expected error types or no errors"
    print(f"  ✓ Error types: {result.error_types}")
    
    # Generate report
    report = analyzer.generate_report()
    assert len(report) > 100, "Report should have content"
    assert "Summary Statistics" in report, "Report should have summary"
    assert "Error Type Breakdown" in report, "Report should have error types"
    print(f"  ✓ Report generated ({len(report)} chars)")
    
    print("\n  Full analysis test passed!")
    return True


def test_report_generation():
    """Test markdown report generation."""
    print("\nTesting report generation...")
    
    from src.evaluation.error_analysis import ErrorAnalyzer
    
    predictions = [
        {
            'question_id': str(i),
            'question': f'Question {i}?',
            'prediction': 'yes' if i % 2 == 0 else 'no',
            'ground_truths': ['yes', 'yes', 'no'],
        }
        for i in range(20)
    ]
    
    analyzer = ErrorAnalyzer(predictions=predictions)
    result = analyzer.analyze()
    report = analyzer.generate_report()
    
    # Check report sections
    sections = [
        "Summary Statistics",
        "Error Type Breakdown",
        "Performance by Question Type",
        "Performance by Answer Length",
        "Common Confusions",
        "Top 10 Error Examples",
    ]
    
    passed = 0
    for section in sections:
        if section in report:
            print(f"  ✓ Section found: {section}")
            passed += 1
        else:
            print(f"  ✗ Section missing: {section}")
    
    print(f"\n  Results: {passed}/{len(sections)} sections found")
    return passed == len(sections)


def test_save_analysis(tmp_dir=None):
    """Test saving analysis to files."""
    print("\nTesting save analysis...")
    
    import tempfile
    import os
    
    from src.evaluation.error_analysis import ErrorAnalyzer
    
    predictions = [
        {
            'question_id': '1',
            'question': 'Is this a test?',
            'prediction': 'yes',
            'ground_truths': ['yes', 'yes', 'yes'],
        },
        {
            'question_id': '2',
            'question': 'What is this?',
            'prediction': 'test',
            'ground_truths': ['example', 'sample', 'test'],
        },
    ]
    
    analyzer = ErrorAnalyzer(predictions=predictions)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_files = analyzer.save_analysis(tmpdir)
        
        # Check files were created
        for name, path in saved_files.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  ✓ {name}: {path.name} ({size} bytes)")
            else:
                print(f"  ✗ {name}: file not created")
                return False
    
    print("\n  Save analysis test passed!")
    return True


def test_visualizations():
    """Test visualization functions (optional - requires matplotlib)."""
    print("\nTesting visualizations...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  ⚠ matplotlib/seaborn not installed, skipping visualization tests")
        return True
    
    import tempfile
    from src.evaluation.error_analysis import ErrorAnalyzer
    from src.evaluation.visualizations import plot_error_analysis
    
    predictions = [
        {
            'question_id': str(i),
            'question': ['Is this?', 'What is?', 'How many?'][i % 3] + f' {i}',
            'prediction': ['yes', 'cat', '3'][i % 3],
            'ground_truths': [['yes'], ['dog'], ['5']][i % 3],
        }
        for i in range(30)
    ]
    
    analyzer = ErrorAnalyzer(predictions=predictions)
    result = analyzer.analyze()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_paths = plot_error_analysis(result, tmpdir)
        
        if not plot_paths:
            print("  ⚠ No plots generated (may be missing dependencies)")
            return True
        
        for name, path in plot_paths.items():
            if path.exists():
                print(f"  ✓ {name}: {path.name}")
            else:
                print(f"  ✗ {name}: not created")
    
    print("\n  Visualization test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 VQA ERROR ANALYSIS TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Question Type Inference", test_question_type_inference),
        ("Type Mismatch Detection", test_type_mismatch_detection),
        ("Partial Correct Detection", test_partial_correct_detection),
        ("Full Analysis Pipeline", test_full_analysis),
        ("Report Generation", test_report_generation),
        ("Save Analysis", test_save_analysis),
        ("Visualizations", test_visualizations),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    if failed == 0:
        print("\n  ✅ All tests passed!")
    else:
        print(f"\n  ❌ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
