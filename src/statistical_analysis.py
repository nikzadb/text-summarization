#!/usr/bin/env python3
"""
Statistical Analysis Module for Text Summarization Benchmarking

This module provides comprehensive statistical analysis including:
- Bootstrap confidence intervals for all metrics
- Paired bootstrap significance tests
- Holm-Bonferroni multiple comparison correction
- Per-dataset statistical reporting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
from scipy import stats


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis."""
    mean: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_samples: int


@dataclass 
class SignificanceTestResult:
    """Results from paired bootstrap significance test."""
    method_a: str
    method_b: str
    metric: str
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    corrected_p_value: Optional[float] = None


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for benchmarking results."""
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95, random_seed: int = 42):
        """
        Initialize statistical analyzer.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (default: 0.95)
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def bootstrap_statistic(self, data: np.ndarray, statistic_func=np.mean) -> BootstrapResult:
        """
        Calculate bootstrap confidence intervals for a statistic.
        
        Args:
            data: Array of data points
            statistic_func: Function to calculate statistic (default: mean)
            
        Returns:
            BootstrapResult with mean, CI, and standard error
        """
        data = np.array(data)
        n = len(data)
        
        if n == 0:
            return BootstrapResult(0.0, 0.0, 0.0, 0.0, 0)
        
        # Original statistic
        original_stat = statistic_func(data)
        
        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha_2 = self.alpha / 2
        ci_lower = np.percentile(bootstrap_stats, alpha_2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha_2) * 100)
        
        # Standard error
        std_error = np.std(bootstrap_stats)
        
        return BootstrapResult(
            mean=original_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_error,
            n_samples=n
        )
    
    def paired_bootstrap_test(self, data_a: np.ndarray, data_b: np.ndarray, 
                            method_a: str, method_b: str, metric: str) -> SignificanceTestResult:
        """
        Perform paired bootstrap significance test.
        
        Args:
            data_a: Data for method A
            data_b: Data for method B (paired with data_a)
            method_a: Name of method A
            method_b: Name of method B
            metric: Name of metric being tested
            
        Returns:
            SignificanceTestResult
        """
        data_a = np.array(data_a)
        data_b = np.array(data_b)
        
        if len(data_a) != len(data_b):
            raise ValueError("Data arrays must have the same length for paired test")
        
        n = len(data_a)
        if n == 0:
            return SignificanceTestResult(
                method_a, method_b, metric, 1.0, 0.0, 0.0, 0.0, False
            )
        
        # Calculate observed difference
        observed_diff = np.mean(data_a) - np.mean(data_b)
        
        # Bootstrap test under null hypothesis (no difference)
        # Center the differences around zero
        differences = data_a - data_b
        centered_diffs = differences - np.mean(differences)
        
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            # Sample with replacement from centered differences
            boot_centered_diffs = np.random.choice(centered_diffs, size=n, replace=True)
            boot_diff = np.mean(boot_centered_diffs)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Effect size (Cohen's d for paired data)
        effect_size = observed_diff / np.std(differences) if np.std(differences) > 0 else 0.0
        
        # Confidence interval for the difference
        alpha_2 = self.alpha / 2
        actual_diffs = []
        for _ in range(self.n_bootstrap):
            boot_indices = np.random.choice(n, size=n, replace=True)
            boot_diff = np.mean(data_a[boot_indices] - data_b[boot_indices])
            actual_diffs.append(boot_diff)
        
        actual_diffs = np.array(actual_diffs)
        ci_lower = np.percentile(actual_diffs, alpha_2 * 100)
        ci_upper = np.percentile(actual_diffs, (1 - alpha_2) * 100)
        
        return SignificanceTestResult(
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            p_value=p_value,
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=p_value < self.alpha
        )
    
    def holm_bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Holm-Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of uncorrected p-values
            
        Returns:
            List of corrected p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if n == 0:
            return []
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Apply Holm correction
        corrected_p = np.zeros(n)
        for i, p in enumerate(sorted_p):
            # Holm correction: multiply by (n - i)
            corrected_p[sorted_indices[i]] = min(1.0, p * (n - i))
        
        # Ensure monotonicity (later p-values can't be smaller than earlier ones)
        for i in range(1, n):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i-1]
            corrected_p[idx] = max(corrected_p[idx], corrected_p[prev_idx])
        
        return corrected_p.tolist()
    
    def analyze_dataset_results(self, detailed_results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze results for a single dataset with full statistical analysis.
        
        Args:
            detailed_results_df: DataFrame with columns: method, rouge1_f1, rouge2_f1, 
                                rougeL_f1, bert_f1, processing_time, cost
        
        Returns:
            Dictionary with statistical analysis results
        """
        if detailed_results_df.empty:
            return {}
        
        methods = detailed_results_df['method'].unique()
        metrics = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bert_f1', 'processing_time', 'cost']
        
        # Calculate combined ROUGE score
        detailed_results_df = detailed_results_df.copy()
        detailed_results_df['combined_rouge'] = (
            detailed_results_df['rouge1_f1'] + 
            detailed_results_df['rouge2_f1'] + 
            detailed_results_df['rougeL_f1']
        ) / 3.0
        
        metrics.append('combined_rouge')
        
        # Bootstrap analysis for each method and metric
        bootstrap_results = {}
        for method in methods:
            method_data = detailed_results_df[detailed_results_df['method'] == method]
            bootstrap_results[method] = {}
            
            for metric in metrics:
                if metric in method_data.columns:
                    data = method_data[metric].values
                    bootstrap_results[method][metric] = self.bootstrap_statistic(data)
        
        # Find best method based on Combined_ROUGE
        best_method = None
        best_combined_rouge = -np.inf
        
        for method in methods:
            if 'combined_rouge' in bootstrap_results[method]:
                rouge_mean = bootstrap_results[method]['combined_rouge'].mean
                if rouge_mean > best_combined_rouge:
                    best_combined_rouge = rouge_mean
                    best_method = method
        
        # Paired bootstrap tests: best vs all others
        significance_tests = []
        if best_method:
            best_data = detailed_results_df[detailed_results_df['method'] == best_method]
            
            p_values_for_correction = []
            test_results = []
            
            for other_method in methods:
                if other_method != best_method:
                    other_data = detailed_results_df[detailed_results_df['method'] == other_method]
                    
                    # Ensure we have paired data (same sample indices)
                    if len(best_data) == len(other_data):
                        # Test Combined_ROUGE
                        test_result = self.paired_bootstrap_test(
                            best_data['combined_rouge'].values,
                            other_data['combined_rouge'].values,
                            best_method,
                            other_method,
                            'combined_rouge'
                        )
                        test_results.append(test_result)
                        p_values_for_correction.append(test_result.p_value)
            
            # Apply Holm-Bonferroni correction
            if p_values_for_correction:
                corrected_p_values = self.holm_bonferroni_correction(p_values_for_correction)
                
                for i, test_result in enumerate(test_results):
                    test_result.corrected_p_value = corrected_p_values[i]
                    test_result.is_significant = corrected_p_values[i] < self.alpha
                
                significance_tests = test_results
        
        return {
            'bootstrap_results': bootstrap_results,
            'best_method': best_method,
            'best_combined_rouge': best_combined_rouge,
            'significance_tests': significance_tests,
            'n_methods': len(methods),
            'confidence_level': self.confidence_level
        }
    
    def generate_statistical_report(self, analysis_results: Dict[str, Any], 
                                  dataset_name: str) -> str:
        """
        Generate a formatted statistical report for a dataset.
        
        Args:
            analysis_results: Results from analyze_dataset_results
            dataset_name: Name of the dataset
            
        Returns:
            Formatted report string
        """
        if not analysis_results:
            return f"No results available for {dataset_name}"
        
        report = [f"\n{'='*60}"]
        report.append(f"STATISTICAL ANALYSIS: {dataset_name.upper()}")
        report.append(f"{'='*60}")
        report.append(f"Confidence Level: {self.confidence_level*100:.1f}%")
        report.append(f"Bootstrap Samples: {self.n_bootstrap:,}")
        
        if analysis_results.get('best_method'):
            report.append(f"Best Method (Combined_ROUGE): {analysis_results['best_method']}")
        
        report.append(f"\n{'='*60}")
        report.append("PER-METHOD RESULTS (Mean ± 95% Bootstrap CI)")
        report.append(f"{'='*60}")
        
        bootstrap_results = analysis_results.get('bootstrap_results', {})
        
        # Format results table
        for method, method_results in bootstrap_results.items():
            report.append(f"\n{method.upper()}:")
            report.append("-" * 40)
            
            # Combined ROUGE
            if 'combined_rouge' in method_results:
                result = method_results['combined_rouge']
                report.append(f"Combined_ROUGE: {result.mean:.4f} ± ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
            
            # BERTScore
            if 'bert_f1' in method_results:
                result = method_results['bert_f1']
                report.append(f"BERTScore:      {result.mean:.4f} ± ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
            
            # Latency
            if 'processing_time' in method_results:
                result = method_results['processing_time']
                report.append(f"Latency (s):    {result.mean:.4f} ± ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
            
            # Cost
            if 'cost' in method_results:
                result = method_results['cost']
                report.append(f"Cost ($):       {result.mean:.6f} ± ({result.ci_lower:.6f}, {result.ci_upper:.6f})")
        
        # Significance tests
        significance_tests = analysis_results.get('significance_tests', [])
        if significance_tests:
            report.append(f"\n{'='*60}")
            report.append("SIGNIFICANCE TESTS (Paired Bootstrap)")
            report.append("Holm-Bonferroni Corrected p-values")
            report.append(f"{'='*60}")
            
            for test in significance_tests:
                sig_marker = "***" if test.corrected_p_value < 0.001 else "**" if test.corrected_p_value < 0.01 else "*" if test.corrected_p_value < 0.05 else ""
                
                report.append(f"{test.method_a} vs {test.method_b}:")
                report.append(f"  Effect size (Cohen's d): {test.effect_size:.4f}")
                report.append(f"  Raw p-value: {test.p_value:.6f}")
                report.append(f"  Corrected p-value: {test.corrected_p_value:.6f} {sig_marker}")
                report.append(f"  Difference CI: ({test.ci_lower:.4f}, {test.ci_upper:.4f})")
                report.append("")
        
        report.append(f"{'='*60}")
        report.append("* p < 0.05, ** p < 0.01, *** p < 0.001")
        
        return "\n".join(report)


def load_detailed_results(dataset_name: str, method_name: str) -> Optional[pd.DataFrame]:
    """
    Load detailed results CSV file for a dataset and method.
    
    Args:
        dataset_name: Name of the dataset
        method_name: Name of the method
        
    Returns:
        DataFrame with detailed results or None if file doesn't exist
    """
    filename = f'detailed_results_{dataset_name}_{method_name}.csv'
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return None


def combine_detailed_results(dataset_name: str, methods: List[str]) -> pd.DataFrame:
    """
    Combine detailed results from multiple methods for statistical analysis.
    
    Args:
        dataset_name: Name of the dataset
        methods: List of method names
        
    Returns:
        Combined DataFrame with all method results
    """
    combined_df = pd.DataFrame()
    
    for method in methods:
        method_df = load_detailed_results(dataset_name, method)
        if method_df is not None:
            combined_df = pd.concat([combined_df, method_df], ignore_index=True)
    
    return combined_df