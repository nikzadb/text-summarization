from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import warnings
from transformers import logging
import torch
from tqdm import tqdm
from scipy import stats
import pandas as pd
from bleurt_pytorch import BleurtTokenizer, BleurtForSequenceClassification

# Suppress transformers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class EvaluationMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Set device for BERT scoring (CUDA or MPS)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # Initialize BLEURT model lazily
        self.bleurt_tokenizer = None
        self.bleurt_model = None
        self._bleurt_initialized = False
    
    def _initialize_bleurt(self):
        """Initialize BLEURT model and tokenizer lazily."""
        if not self._bleurt_initialized:
            try:
                print("Loading BLEURT model...")
                self.bleurt_tokenizer = BleurtTokenizer.from_pretrained('Elron/bleurt-base-512')
                self.bleurt_model = BleurtForSequenceClassification.from_pretrained('Elron/bleurt-base-512')
                self.bleurt_model.to(self.device)
                self.bleurt_model.eval()
                self._bleurt_initialized = True
                print(f"BLEURT model loaded successfully on device: {self.device}")
            except Exception as e:
                print(f"Warning: Failed to load BLEURT model: {e}")
                self._bleurt_initialized = False
                
    def compute_bleurt_scores(self, references: List[str], candidates: List[str], batch_size: int = 32) -> Dict[str, List[float]]:
        """Compute BLEURT scores for a list of reference-candidate pairs."""
        self._initialize_bleurt()
        
        if not self._bleurt_initialized:
            print("BLEURT model not available, returning zero scores")
            return {'bleurt_score': [0.0] * len(references)}
        
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")
        
        bleurt_scores = []
        
        # Process in batches
        for i in tqdm(range(0, len(references), batch_size), desc="BLEURT"):
            batch_refs = references[i:i+batch_size]
            batch_cands = candidates[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.bleurt_tokenizer(
                    batch_refs, 
                    batch_cands, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                outputs = self.bleurt_model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
                
                # Handle single item case
                if scores.ndim == 0:
                    scores = [float(scores)]
                else:
                    scores = scores.tolist()
                    
                bleurt_scores.extend(scores)
        
        return {'bleurt_score': bleurt_scores}
    
    def compute_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure
        }
    
    def compute_bert_scores(self, references: List[str], candidates: List[str], lang: str = 'en', batch_size: int = 64) -> Dict[str, List[float]]:
        P, R, F1 = bert_score(candidates, references, lang=lang, verbose=False, device=self.device, batch_size=batch_size)
        
        return {
            'bert_precision': P.tolist(),
            'bert_recall': R.tolist(),
            'bert_f1': F1.tolist()
        }
    
    def evaluate_single_summary(self, reference: str, candidate: str) -> Dict[str, float]:
        rouge_scores = self.compute_rouge_scores(reference, candidate)
        
        bert_scores = self.compute_bert_scores([reference], [candidate])
        bert_metrics = {
            'bert_precision': bert_scores['bert_precision'][0],
            'bert_recall': bert_scores['bert_recall'][0],
            'bert_f1': bert_scores['bert_f1'][0]
        }
        
        bleurt_scores = self.compute_bleurt_scores([reference], [candidate])
        bleurt_metrics = {
            'bleurt_score': bleurt_scores['bleurt_score'][0]
        }
        
        return {**rouge_scores, **bert_metrics, **bleurt_metrics}
    
    def evaluate_batch_summaries(self, references: List[str], candidates: List[str], batch_size: int = 64) -> Dict[str, Any]:
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")
        
        print(f"Evaluating {len(references)} summaries using device: {self.device}")
        
        rouge_scores = {
            'rouge1_precision': [], 'rouge1_recall': [], 'rouge1_f1': [],
            'rouge2_precision': [], 'rouge2_recall': [], 'rouge2_f1': [],
            'rougeL_precision': [], 'rougeL_recall': [], 'rougeL_f1': []
        }
        
        # Process ROUGE scores with progress bar
        print("Computing ROUGE scores...")
        for ref, cand in tqdm(zip(references, candidates), total=len(references), desc="ROUGE"):
            rouge_result = self.compute_rouge_scores(ref, cand)
            for key, value in rouge_result.items():
                rouge_scores[key].append(value)
        
        # Process BERT scores in optimized batch mode
        print("Computing BERT scores...")
        bert_scores = self.compute_bert_scores(references, candidates, batch_size=batch_size)
        
        # Process BLEURT scores in optimized batch mode
        print("Computing BLEURT scores...")
        bleurt_scores = self.compute_bleurt_scores(references, candidates, batch_size=batch_size//2)  # Smaller batch size for BLEURT
        
        all_scores = {**rouge_scores, **bert_scores, **bleurt_scores}
        
        avg_scores = {}
        for key, values in all_scores.items():
            avg_scores[f"{key}_mean"] = np.mean(values)
            avg_scores[f"{key}_std"] = np.std(values)
        
        return {
            'individual_scores': all_scores,
            'average_scores': avg_scores,
            'sample_count': len(references)
        }
    
    def get_summary_statistics(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        avg_scores = evaluation_results['average_scores']
        
        return {
            'rouge1_f1': avg_scores.get('rouge1_f1_mean', 0.0),
            'rouge2_f1': avg_scores.get('rouge2_f1_mean', 0.0),
            'rougeL_f1': avg_scores.get('rougeL_f1_mean', 0.0),
            'bert_f1': avg_scores.get('bert_f1_mean', 0.0),
            'bleurt_score': avg_scores.get('bleurt_score_mean', 0.0),
            'combined_score': (
                avg_scores.get('rouge1_f1_mean', 0.0) +
                avg_scores.get('rouge2_f1_mean', 0.0) +
                avg_scores.get('rougeL_f1_mean', 0.0)
            ) / 3.0,
            'combined_score_with_bleurt': (
                avg_scores.get('rouge1_f1_mean', 0.0) +
                avg_scores.get('rouge2_f1_mean', 0.0) +
                avg_scores.get('rougeL_f1_mean', 0.0) +
                avg_scores.get('bleurt_score_mean', 0.0)
            ) / 4.0
        }
    
    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        comparison = {}
        
        for method_name, evaluation_result in results.items():
            comparison[method_name] = self.get_summary_statistics(evaluation_result)
        
        return comparison
    
    def bootstrap_confidence_interval(self, 
                                    scores: List[float], 
                                    n_bootstrap: int = 5000, 
                                    confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate bootstrap confidence interval for a metric.
        
        Process:
        1. Take original scores (sample data)
        2. Generate n_bootstrap resampled datasets by sampling with replacement
        3. Calculate mean for each resampled dataset
        4. Use distribution of bootstrap means to estimate confidence interval
        
        Args:
            scores: List of metric scores for individual samples
            n_bootstrap: Number of bootstrap samples (default: 1000)
            confidence_level: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            Dictionary with mean, lower bound, upper bound, and std error
        """
        scores = np.array(scores)
        n_samples = len(scores)
        
        # Generate bootstrap samples
        bootstrap_means = []
        np.random.seed(42)  # For reproducible results
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'mean': np.mean(scores),
            'ci_lower': lower_bound,
            'ci_upper': upper_bound,
            'std_error': np.std(bootstrap_means),
            'n_samples': n_samples,
            'confidence_level': confidence_level
        }
    
    def statistical_comparison(self, 
                             method_results: Dict[str, Dict[str, Any]], 
                             dataset_name: str,
                             metric: str = 'combined_score',
                             n_bootstrap: int = 1000,
                             confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform statistical comparison between methods using bootstrap confidence intervals.
        
        Process:
        1. Calculate combined_score for each sample for each method
        2. Find the best performing method (highest mean combined_score)
        3. Calculate bootstrap confidence intervals for all methods
        4. Perform pairwise statistical tests against the best method
        5. Determine statistical significance based on CI overlap and permutation tests
        
        Args:
            method_results: Dictionary mapping method names to evaluation results
            dataset_name: Name of the dataset being analyzed
            metric: Metric to use for comparison ('combined_score', 'rouge1_f1', etc.)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
        
        Returns:
            Statistical analysis results including best method and comparisons
        """
        
        # Calculate individual combined scores for each method
        method_scores = {}
        for method_name, results in method_results.items():
            individual_scores = results['individual_scores']
            
            if metric == 'combined_score':
                # Calculate combined score for each sample
                rouge1_scores = individual_scores['rouge1_f1']
                rouge2_scores = individual_scores['rouge2_f1']
                rougeL_scores = individual_scores['rougeL_f1']
                
                combined_scores = [(r1 + r2 + rl) / 3.0 
                                 for r1, r2, rl in zip(rouge1_scores, rouge2_scores, rougeL_scores)]
                method_scores[method_name] = combined_scores
            elif metric == 'combined_score_with_bleurt':
                # Calculate combined score including BLEURT for each sample
                rouge1_scores = individual_scores['rouge1_f1']
                rouge2_scores = individual_scores['rouge2_f1']
                rougeL_scores = individual_scores['rougeL_f1']
                bleurt_scores = individual_scores.get('bleurt_score', [0.0] * len(rouge1_scores))
                
                combined_scores = [(r1 + r2 + rl + b) / 4.0 
                                 for r1, r2, rl, b in zip(rouge1_scores, rouge2_scores, rougeL_scores, bleurt_scores)]
                method_scores[method_name] = combined_scores
            else:
                method_scores[method_name] = individual_scores[metric]
        
        # Find best performing method
        method_means = {method: np.mean(scores) for method, scores in method_scores.items()}
        best_method = max(method_means.keys(), key=lambda k: method_means[k])
        best_scores = method_scores[best_method]
        
        # Calculate bootstrap CIs for all methods
        bootstrap_results = {}
        for method_name, scores in method_scores.items():
            bootstrap_results[method_name] = self.bootstrap_confidence_interval(
                scores, n_bootstrap, confidence_level
            )
        
        # Perform statistical tests against best method
        statistical_tests = {}
        for method_name, scores in method_scores.items():
            if method_name == best_method:
                statistical_tests[method_name] = {
                    'is_best': True,
                    'p_value': None,
                    'effect_size': 0.0,
                    'significantly_different': False,
                    'interpretation': 'Best performing method'
                }
            else:
                # Permutation test for statistical significance
                p_value = self._permutation_test(best_scores, scores)
                effect_size = (method_means[best_method] - method_means[method_name]) / np.std(best_scores)
                
                # Check CI overlap
                best_ci = bootstrap_results[best_method]
                method_ci = bootstrap_results[method_name]
                ci_overlap = not (method_ci['ci_upper'] < best_ci['ci_lower'] or 
                                method_ci['ci_lower'] > best_ci['ci_upper'])
                
                significantly_different = p_value < (1 - confidence_level) or not ci_overlap
                
                if significantly_different:
                    if method_means[method_name] < method_means[best_method]:
                        interpretation = f"Significantly worse than {best_method}"
                    else:
                        interpretation = f"Significantly better than {best_method}"
                else:
                    interpretation = f"No significant difference from {best_method}"
                
                statistical_tests[method_name] = {
                    'is_best': False,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significantly_different': significantly_different,
                    'ci_overlap': ci_overlap,
                    'interpretation': interpretation
                }
        
        return {
            'dataset': dataset_name,
            'metric': metric,
            'best_method': best_method,
            'best_method_score': method_means[best_method],
            'method_scores': method_means,
            'bootstrap_results': bootstrap_results,
            'statistical_tests': statistical_tests,
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence_level
        }
    
    def _permutation_test(self, 
                         group1: List[float], 
                         group2: List[float], 
                         n_permutations: int = 1000) -> float:
        """
        Perform permutation test to determine if two groups have significantly different means.
        
        Process:
        1. Calculate observed difference in means
        2. Pool all samples together
        3. Randomly permute samples between groups many times
        4. Calculate difference in means for each permutation
        5. P-value = proportion of permutations with difference >= observed difference
        """
        observed_diff = np.mean(group1) - np.mean(group2)
        combined = np.array(group1 + group2)
        n1, n2 = len(group1), len(group2)
        
        np.random.seed(42)  # For reproducible results
        permutation_diffs = []
        
        for _ in range(n_permutations):
            # Randomly shuffle combined data
            shuffled = np.random.permutation(combined)
            # Split into two groups of original sizes
            perm_group1 = shuffled[:n1]
            perm_group2 = shuffled[n1:n1+n2]
            # Calculate difference in means
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            permutation_diffs.append(perm_diff)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        return p_value
    
    def generate_statistical_report(self, 
                                  statistical_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable statistical analysis report.
        """
        report = []
        report.append("=" * 80)
        report.append(f"STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Dataset: {statistical_results['dataset']}")
        report.append(f"Metric: {statistical_results['metric']}")
        report.append(f"Confidence Level: {statistical_results['confidence_level']*100:.0f}%")
        report.append(f"Bootstrap Samples: {statistical_results['n_bootstrap']}")
        report.append("")
        
        # Best method summary
        best_method = statistical_results['best_method']
        best_score = statistical_results['best_method_score']
        report.append(f"🏆 BEST PERFORMING METHOD: {best_method}")
        report.append(f"Score: {best_score:.4f}")
        report.append("")
        
        # Detailed results for each method
        report.append("📊 DETAILED RESULTS:")
        report.append("-" * 80)
        
        bootstrap_results = statistical_results['bootstrap_results']
        statistical_tests = statistical_results['statistical_tests']
        
        for method_name in sorted(bootstrap_results.keys()):
            bootstrap_info = bootstrap_results[method_name]
            test_info = statistical_tests[method_name]
            
            report.append(f"\n{method_name}:")
            report.append(f"  Mean: {bootstrap_info['mean']:.4f}")
            report.append(f"  95% CI: [{bootstrap_info['ci_lower']:.4f}, {bootstrap_info['ci_upper']:.4f}]")
            report.append(f"  Std Error: {bootstrap_info['std_error']:.4f}")
            report.append(f"  Sample Size: {bootstrap_info['n_samples']}")
            
            if test_info['is_best']:
                report.append("  Status: ✓ BEST METHOD")
            else:
                report.append(f"  vs {best_method}:")
                report.append(f"    P-value: {test_info['p_value']:.4f}")
                report.append(f"    Effect Size: {test_info['effect_size']:.4f}")
                report.append(f"    CI Overlap: {'Yes' if test_info['ci_overlap'] else 'No'}")
                status = "❌ Significantly Different" if test_info['significantly_different'] else "✓ Not Significant"
                report.append(f"    Status: {status}")
                report.append(f"    Interpretation: {test_info['interpretation']}")
        
        report.append("")
        report.append("=" * 80)
        report.append("METHODOLOGY:")
        report.append("• Bootstrap confidence intervals with 1000 resamples")
        report.append("• Permutation tests for statistical significance")
        report.append("• Combined score = average of ROUGE-1, ROUGE-2, ROUGE-L F1 scores")
        report.append("• Statistical significance determined by CI overlap and p-values")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def comprehensive_statistical_analysis(self, 
                                         method_results: Dict[str, Dict[str, Any]], 
                                         dataset_name: str,
                                         n_bootstrap: int = 5000,
                                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis for ALL metrics with confidence intervals.
        
        Calculates bootstrap confidence intervals for:
        - ROUGE-1 F1
        - ROUGE-2 F1  
        - ROUGE-L F1
        - BERT F1
        - Combined Score
        - Average Time
        - Average Cost
        
        Args:
            method_results: Dictionary mapping method names to evaluation results
            dataset_name: Name of the dataset being analyzed
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
        
        Returns:
            Comprehensive statistical analysis for all metrics
        """
        
        # Define all metrics to analyze (including performance metrics)
        metrics = {
            'rouge1_f1': 'ROUGE-1 F1',
            'rouge2_f1': 'ROUGE-2 F1', 
            'rougeL_f1': 'ROUGE-L F1',
            'bert_f1': 'BERT F1',
            'bleurt_score': 'BLEURT Score',
            'combined_score': 'Combined Score',
            'combined_score_with_bleurt': 'Combined Score with BLEURT',
            'processing_time': 'Processing Time (s)',
            'cost': 'Cost ($)'
        }
        
        comprehensive_results = {
            'dataset': dataset_name,
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap,
            'metrics_analysis': {}
        }
        
        # Calculate scores for each metric
        all_method_scores = {}
        for metric_key, metric_name in metrics.items():
            method_scores = {}
            
            for method_name, results in method_results.items():
                individual_scores = results['individual_scores']
                
                if metric_key == 'combined_score':
                    # Calculate combined score for each sample
                    rouge1_scores = individual_scores['rouge1_f1']
                    rouge2_scores = individual_scores['rouge2_f1']
                    rougeL_scores = individual_scores['rougeL_f1']
                    
                    combined_scores = [(r1 + r2 + rl) / 3.0 
                                     for r1, r2, rl in zip(rouge1_scores, rouge2_scores, rougeL_scores)]
                    method_scores[method_name] = combined_scores
                elif metric_key == 'combined_score_with_bleurt':
                    # Calculate combined score including BLEURT for each sample
                    rouge1_scores = individual_scores['rouge1_f1']
                    rouge2_scores = individual_scores['rouge2_f1']
                    rougeL_scores = individual_scores['rougeL_f1']
                    bleurt_scores = individual_scores.get('bleurt_score', [0.0] * len(rouge1_scores))
                    
                    combined_scores = [(r1 + r2 + rl + b) / 4.0 
                                     for r1, r2, rl, b in zip(rouge1_scores, rouge2_scores, rougeL_scores, bleurt_scores)]
                    method_scores[method_name] = combined_scores
                elif metric_key in ['processing_time', 'cost']:
                    # Use performance metrics from detailed results
                    method_scores[method_name] = individual_scores[metric_key]
                else:
                    # Use existing quality metric scores
                    method_scores[method_name] = individual_scores[metric_key]
            
            all_method_scores[metric_key] = method_scores
        
        # Analyze each metric
        for metric_key, metric_name in metrics.items():
            method_scores = all_method_scores[metric_key]
            
            # Find best performing method for this metric
            method_means = {method: np.mean(scores) for method, scores in method_scores.items()}
            
            # For time and cost, lower is better; for quality metrics, higher is better
            if metric_key in ['processing_time', 'cost']:
                best_method = min(method_means.keys(), key=lambda k: method_means[k])
            else:
                best_method = max(method_means.keys(), key=lambda k: method_means[k])
            
            best_scores = method_scores[best_method]
            
            # Calculate bootstrap CIs for all methods
            bootstrap_results = {}
            for method_name, scores in method_scores.items():
                bootstrap_results[method_name] = self.bootstrap_confidence_interval(
                    scores, n_bootstrap, confidence_level
                )
            
            # Perform statistical tests against best method
            statistical_tests = {}
            for method_name, scores in method_scores.items():
                if method_name == best_method:
                    statistical_tests[method_name] = {
                        'is_best': True,
                        'p_value': None,
                        'effect_size': 0.0,
                        'significantly_different': False,
                        'interpretation': 'Best performing method'
                    }
                else:
                    # Permutation test for statistical significance
                    p_value = self._permutation_test(best_scores, scores)
                    effect_size = (method_means[best_method] - method_means[method_name]) / np.std(best_scores)
                    
                    # Check CI overlap
                    best_ci = bootstrap_results[best_method]
                    method_ci = bootstrap_results[method_name]
                    ci_overlap = not (method_ci['ci_upper'] < best_ci['ci_lower'] or 
                                    method_ci['ci_lower'] > best_ci['ci_upper'])
                    
                    significantly_different = p_value < (1 - confidence_level) or not ci_overlap
                    
                    if significantly_different:
                        # Interpretation depends on whether lower or higher is better
                        if metric_key in ['processing_time', 'cost']:
                            # For time/cost, lower is better
                            if method_means[method_name] > method_means[best_method]:
                                interpretation = f"Significantly worse than {best_method} (higher {metric_key.replace('_', ' ')})"
                            else:
                                interpretation = f"Significantly better than {best_method} (lower {metric_key.replace('_', ' ')})"
                        else:
                            # For quality metrics, higher is better  
                            if method_means[method_name] < method_means[best_method]:
                                interpretation = f"Significantly worse than {best_method}"
                            else:
                                interpretation = f"Significantly better than {best_method}"
                    else:
                        interpretation = f"No significant difference from {best_method}"
                    
                    statistical_tests[method_name] = {
                        'is_best': False,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'significantly_different': significantly_different,
                        'ci_overlap': ci_overlap,
                        'interpretation': interpretation
                    }
            
            # Store results for this metric
            comprehensive_results['metrics_analysis'][metric_key] = {
                'metric_name': metric_name,
                'best_method': best_method,
                'best_method_score': method_means[best_method],
                'method_scores': method_means,
                'bootstrap_results': bootstrap_results,
                'statistical_tests': statistical_tests
            }
        
        return comprehensive_results
    
    def generate_comprehensive_report(self, 
                                    comprehensive_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive statistical analysis report for all metrics.
        """
        report = []
        report.append("=" * 100)
        report.append(f"COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append("=" * 100)
        report.append(f"Dataset: {comprehensive_results['dataset']}")
        report.append(f"Confidence Level: {comprehensive_results['confidence_level']*100:.0f}%")
        report.append(f"Bootstrap Samples: {comprehensive_results['n_bootstrap']}")
        report.append("")
        
        # Summary of best methods per metric
        report.append("🏆 BEST METHODS BY METRIC:")
        report.append("-" * 50)
        for metric_key, analysis in comprehensive_results['metrics_analysis'].items():
            metric_name = analysis['metric_name']
            best_method = analysis['best_method']
            best_score = analysis['best_method_score']
            report.append(f"{metric_name:15}: {best_method:20} ({best_score:.4f})")
        report.append("")
        
        # Detailed analysis for each metric
        for metric_key, analysis in comprehensive_results['metrics_analysis'].items():
            metric_name = analysis['metric_name']
            best_method = analysis['best_method']
            bootstrap_results = analysis['bootstrap_results']
            statistical_tests = analysis['statistical_tests']
            
            report.append("=" * 80)
            report.append(f"METRIC: {metric_name}")
            report.append("=" * 80)
            report.append(f"🏆 Best Method: {best_method} ({analysis['best_method_score']:.4f})")
            report.append("")
            
            # Results for each method
            for method_name in sorted(bootstrap_results.keys()):
                bootstrap_info = bootstrap_results[method_name]
                test_info = statistical_tests[method_name]
                
                report.append(f"{method_name}:")
                report.append(f"  Mean: {bootstrap_info['mean']:.4f}")
                report.append(f"  95% CI: [{bootstrap_info['ci_lower']:.4f}, {bootstrap_info['ci_upper']:.4f}]")
                report.append(f"  Std Error: {bootstrap_info['std_error']:.4f}")
                
                if test_info['is_best']:
                    report.append("  Status: ✓ BEST METHOD")
                else:
                    report.append(f"  vs {best_method}:")
                    report.append(f"    P-value: {test_info['p_value']:.4f}")
                    report.append(f"    Effect Size: {test_info['effect_size']:.4f}")
                    status = "❌ Significantly Different" if test_info['significantly_different'] else "✓ Not Significant"
                    report.append(f"    Status: {status}")
                
                report.append("")
        
        report.append("=" * 100)
        report.append("METHODOLOGY:")
        report.append("• Bootstrap confidence intervals with 5000 resamples")
        report.append("• Permutation tests for statistical significance")
        report.append("• Combined score = average of ROUGE-1, ROUGE-2, ROUGE-L F1 scores")
        report.append("• Combined score with BLEURT = average of ROUGE-1, ROUGE-2, ROUGE-L F1, and BLEURT scores")
        report.append("• BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)")
        report.append("• Statistical significance determined by CI overlap and p-values")
        report.append("• Analysis performed separately for each metric")
        report.append("=" * 100)
        
        return "\n".join(report)