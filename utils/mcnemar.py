"""
McNemar's test for comparing model predictions.
Statistical test to determine if architectural changes significantly improve performance.
"""

import numpy as np
from scipy.stats import chi2


def mcnemar_test(predictions_before, predictions_after, true_labels, 
                 correction=True, alpha=0.05):
    """
    Perform McNemar's test to compare two models on the same test set.
    
    Tests null hypothesis: P(correct before, wrong after) = P(wrong before, correct after)
    
    Args:
        predictions_before: Predictions from model before change
        predictions_after: Predictions from model after change
        true_labels: True labels
        correction: Whether to apply continuity correction
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with test results
    """
    correct_before = (predictions_before == true_labels)
    correct_after = (predictions_after == true_labels)
    
    # Contingency table entries
    n_both_correct = np.sum(correct_before & correct_after)
    n_both_wrong = np.sum(~correct_before & ~correct_after)
    n_improved = np.sum(~correct_before & correct_after)  # wrong -> correct
    n_degraded = np.sum(correct_before & ~correct_after)  # correct -> wrong
    
    # McNemar's test focuses on discordant pairs
    n_discordant = n_improved + n_degraded
    
    if n_discordant == 0:
        # No disagreement between models
        chi2_stat = 0.0
        p_value = 1.0
    else:
        # Chi-squared test statistic with continuity correction
        if correction:
            chi2_stat = (abs(n_improved - n_degraded) - 1) ** 2 / n_discordant
        else:
            chi2_stat = (n_improved - n_degraded) ** 2 / n_discordant
        
        # p-value from chi-squared distribution (df=1)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Calculate improvement ratio
    if n_degraded > 0:
        improvement_ratio = n_improved / n_degraded
    else:
        improvement_ratio = float('inf') if n_improved > 0 else 1.0
    
    return {
        'n_improved': int(n_improved),
        'n_degraded': int(n_degraded),
        'n_both_correct': int(n_both_correct),
        'n_both_wrong': int(n_both_wrong),
        'chi2_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'is_significant': bool(is_significant),
        'improvement_ratio': float(improvement_ratio),
        'alpha': alpha
    }


def print_mcnemar_results(results, name_before, name_after):
    """
    Print formatted McNemar's test results.
    
    Args:
        results: Dictionary from mcnemar_test()
        name_before: Name of model before change
        name_after: Name of model after change
    """
    print("\n" + "=" * 70)
    print(f"McNemar's Test: {name_before} vs {name_after}")
    print("=" * 70)
    
    print(f"Samples improved (wrong → correct):    {results['n_improved']:4d}")
    print(f"Samples degraded (correct → wrong):    {results['n_degraded']:4d}")
    print(f"Both models correct:                   {results['n_both_correct']:4d}")
    print(f"Both models wrong:                     {results['n_both_wrong']:4d}")
    print("-" * 70)
    print(f"Improvement ratio:                     {results['improvement_ratio']:.2f}:1")
    print(f"Chi-squared statistic:                 {results['chi2_statistic']:.3f}")
    print(f"p-value:                               {results['p_value']:.4f}")
    print(f"Significant at α={results['alpha']}:               {'Yes ✓' if results['is_significant'] else 'No ✗'}")
    print("=" * 70)
    
    # Interpretation
    if results['is_significant']:
        if results['n_improved'] > results['n_degraded']:
            print(f"→ {name_after} is SIGNIFICANTLY BETTER than {name_before}")
        else:
            print(f"→ {name_after} is SIGNIFICANTLY WORSE than {name_before}")
    else:
        print(f"→ No significant difference between {name_before} and {name_after}")
    print()


def compare_multiple_models(models_dict, true_labels, alpha=0.05):
    """
    Compare multiple model pairs using McNemar's test.
    
    Args:
        models_dict: Dict with keys as (name_before, name_after) tuples
                    and values as (predictions_before, predictions_after) tuples
        true_labels: True labels
        alpha: Significance level
        
    Returns:
        Dictionary of results for each comparison
    """
    all_results = {}
    
    for (name_before, name_after), (preds_before, preds_after) in models_dict.items():
        results = mcnemar_test(preds_before, preds_after, true_labels, alpha=alpha)
        all_results[(name_before, name_after)] = results
        print_mcnemar_results(results, name_before, name_after)
    
    return all_results


if __name__ == "__main__":
    print("Testing McNemar's test implementation...")
    
    # Generate dummy predictions
    n_samples = 1000
    true_labels = np.random.randint(0, 10, n_samples)
    
    # Model 1: 85% accuracy
    preds_before = true_labels.copy()
    wrong_idx = np.random.choice(n_samples, size=150, replace=False)
    preds_before[wrong_idx] = (preds_before[wrong_idx] + 1) % 10
    
    # Model 2: 88% accuracy (improves 40, degrades 10)
    preds_after = preds_before.copy()
    # Correct 40 mistakes
    correct_idx = np.random.choice(wrong_idx, size=40, replace=False)
    preds_after[correct_idx] = true_labels[correct_idx]
    # Make 10 new mistakes
    correct_before = np.where(preds_before == true_labels)[0]
    new_wrong_idx = np.random.choice(correct_before, size=10, replace=False)
    preds_after[new_wrong_idx] = (preds_after[new_wrong_idx] + 1) % 10
    
    # Run test
    results = mcnemar_test(preds_before, preds_after, true_labels)
    print_mcnemar_results(results, "Model Before", "Model After")
    
    print("✓ McNemar's test implementation tested successfully")
