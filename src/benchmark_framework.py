import time
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .summarizers.traditional import TextRankSummarizer, TFIDFRankSummarizer
from .summarizers.llm import OpenSourceLLMSummarizer, T5Summarizer, DistilBARTSummarizer, PegasusXSummarizer, RetrievalAugmentedSummarizer, LongformerEncoderDecoderSummarizer
from .summarizers.gemini import GeminiSummarizer, HybridTFIDFRankGeminiSummarizer, HybridTextRankGeminiSummarizer
from .dataset_loader import DatasetLoader
from .evaluation_metrics import EvaluationMetrics
from .lambda_simulation import LambdaSimulator
from .statistical_analysis import StatisticalAnalyzer, combine_detailed_results


@dataclass
class BenchmarkResult:
    method: str
    dataset: str
    avg_time: float
    avg_cost: float
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    bert_f1: float
    combined_score: float
    sample_count: int
    total_time: float
    total_cost: float


class BenchmarkFramework:
    def __init__(self, use_lambda_simulation: bool = True, enable_statistical_analysis: bool = True):
        self.use_lambda_simulation = use_lambda_simulation
        self.enable_statistical_analysis = enable_statistical_analysis
        self.summarizers = {}
        self.evaluator = EvaluationMetrics()
        self.results = []
        self.statistical_analyzer = StatisticalAnalyzer() if enable_statistical_analysis else None
        
        self._initialize_summarizers()
    
    def _initialize_summarizers(self):
        self.summarizers = {
            'textrank': TextRankSummarizer(),
            'tfidfrank': TFIDFRankSummarizer(),
            't5': T5Summarizer(),
            'distilbart': DistilBARTSummarizer(),
            'bart': OpenSourceLLMSummarizer('facebook/bart-large-cnn'),
            'Pegasus-X': PegasusXSummarizer(),
            'LongformerEncoderDecoder': LongformerEncoderDecoderSummarizer(),
            'Retrieval-Augmented-Summarizer': RetrievalAugmentedSummarizer()
        }
        
        try:
            self.summarizers['gemini'] = GeminiSummarizer()
            self.summarizers['hybrid_textrank_gemini'] = HybridTextRankGeminiSummarizer()
            self.summarizers['hybrid_tfidfrank_gemini'] = HybridTFIDFRankGeminiSummarizer()
        except Exception as e:
            print(f"Warning: Could not initialize Gemini-based summarizers: {e}")
    
    def benchmark_method(self, 
                        method_name: str, 
                        dataset_samples: List[Dict[str, str]], 
                        dataset_name: str,
                        max_sentences: int = 3) -> BenchmarkResult:
        
        if method_name not in self.summarizers:
            raise ValueError(f"Unknown method: {method_name}")
        
        summarizer = self.summarizers[method_name]
        # Configure summarizer for dataset-specific regime (e.g., Hybrid extraction sentence count)
        if hasattr(summarizer, "set_dataset") and callable(getattr(summarizer, "set_dataset")):
            try:
                summarizer.set_dataset(dataset_name)
            except Exception as e:
                print(f"Warning: could not set dataset on summarizer {method_name}: {e}")
        summaries = []
        references = []
        times = []
        costs = []
        
        print(f"Benchmarking {method_name} on {len(dataset_samples)} samples...")
        
        for sample in tqdm(dataset_samples, desc=f"Processing {method_name}"):
            if self.use_lambda_simulation and method_name in ['textrank', 'tfidfrank']:
                with LambdaSimulator() as lambda_sim:
                    result = lambda_sim.benchmark_summarizer_in_lambda(
                        sample['article'], method_name, max_sentences
                    )
                    summary = f"Lambda-processed summary for {method_name}"
                    time_taken = result['total_time']
                    cost = 0.001  # Lambda cost simulation
            else:
                benchmark_result = summarizer.benchmark_summarize(
                    sample['article'], max_sentences
                )
                summary = benchmark_result['summary']
                time_taken = benchmark_result['time_taken']
                cost = benchmark_result.get('cost', 0.0)
            
            summaries.append(summary)
            references.append(sample['reference_summary'])
            times.append(time_taken)
            costs.append(cost)
        
        evaluation_result = self.evaluator.evaluate_batch_summaries(references, summaries)
        summary_stats = self.evaluator.get_summary_statistics(evaluation_result)

        # Save detailed sample-level results
        print(f"Saving detailed results for {method_name}...")
        individual_scores = evaluation_result['individual_scores']
        
        detailed_data = []
        for i, (ref, summ) in enumerate(zip(references, summaries)):
            sample_data = {
                'sample_id': i,
                'method': method_name,
                'dataset': dataset_name,
                'reference_summary': ref,
                'generated_summary': summ,
                'processing_time': times[i],
                'cost': costs[i],
                'rouge1_precision': individual_scores['rouge1_precision'][i],
                'rouge1_recall': individual_scores['rouge1_recall'][i], 
                'rouge1_f1': individual_scores['rouge1_f1'][i],
                'rouge2_precision': individual_scores['rouge2_precision'][i],
                'rouge2_recall': individual_scores['rouge2_recall'][i],
                'rouge2_f1': individual_scores['rouge2_f1'][i],
                'rougeL_precision': individual_scores['rougeL_precision'][i],
                'rougeL_recall': individual_scores['rougeL_recall'][i],
                'rougeL_f1': individual_scores['rougeL_f1'][i],
                'bert_precision': individual_scores['bert_precision'][i],
                'bert_recall': individual_scores['bert_recall'][i],
                'bert_f1': individual_scores['bert_f1'][i]
            }
            detailed_data.append(sample_data)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(f'detailed_results_{dataset_name}_{method_name}.csv', index=False)

        # free up the dataframe memory
        detailed_df = None 
        
        result = BenchmarkResult(
            method=method_name,
            dataset=dataset_name,
            avg_time=sum(times) / len(times),
            avg_cost=sum(costs) / len(costs),
            rouge1_f1=summary_stats['rouge1_f1'],
            rouge2_f1=summary_stats['rouge2_f1'],
            rougeL_f1=summary_stats['rougeL_f1'],
            bert_f1=summary_stats['bert_f1'],
            combined_score=summary_stats['combined_score'],
            sample_count=len(dataset_samples),
            total_time=sum(times),
            total_cost=sum(costs)
        )
        
        self.results.append(result)
        
        # Save intermediate results after each method
        self._save_intermediate_results()
        
        return result
    
    def run_comprehensive_benchmark(self, 
                                  datasets: List[str] = ['cnn_dailymail', 'arxiv'],
                                  methods: Optional[List[str]] = None,
                                  max_samples: int = 50,
                                  max_sentences: int = 3) -> List[BenchmarkResult]:
        
        if methods is None:
            methods = list(self.summarizers.keys())
        
        loader = DatasetLoader('benchmark')
        all_results = []
        
        for dataset_name in datasets:
            print(f"\n=== Loading {dataset_name} dataset ===")
            try:
                dataset_samples = loader.load_dataset(dataset_name, max_samples=max_samples)
                dataset_stats = loader.get_data_statistics()
                print(f"Dataset stats: {dataset_stats}")
                
                for method in methods:
                    if method in self.summarizers:
                        try:
                            result = self.benchmark_method(
                                method, dataset_samples, dataset_name, max_sentences
                            )
                            all_results.append(result)
                            print(f"✓ {method}: ROUGE-1 F1: {result.rouge1_f1:.3f}, "
                                  f"Time: {result.avg_time:.3f}s, Cost: ${result.avg_cost:.6f}")
                        except Exception as e:
                            print(f"✗ Failed to benchmark {method}: {e}")
                    else:
                        print(f"✗ Unknown method: {method}")
                        
            except Exception as e:
                print(f"✗ Failed to load {dataset_name}: {e}")
        
        return all_results
    
    def _save_intermediate_results(self):
        """Save intermediate results as CSV after each method completion"""
        if self.results:
            df = self.get_results_dataframe()
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_intermediate_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Intermediate results saved to {filename}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Method': result.method,
                'Dataset': result.dataset,
                'ROUGE-1 F1': result.rouge1_f1,
                'ROUGE-2 F1': result.rouge2_f1,
                'ROUGE-L F1': result.rougeL_f1,
                'BERT F1': result.bert_f1,
                'Combined Score': result.combined_score,
                'Avg Time (s)': result.avg_time,
                'Avg Cost ($)': result.avg_cost,
                'Total Time (s)': result.total_time,
                'Total Cost ($)': result.total_cost,
                'Sample Count': result.sample_count
            })
        
        return pd.DataFrame(data)
    
    def save_results(self, filename: str):
        df = self.get_results_dataframe()
        
        if filename.endswith('.csv'):
            df.to_csv(filename, index=False)
        elif filename.endswith('.json'):
            results_dict = [vars(result) for result in self.results]
            with open(filename, 'w') as f:
                json.dump(results_dict, f, indent=2)
        else:
            raise ValueError("Filename must end with .csv or .json")
        
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        if not self.results:
            print("No results to display.")
            return
        
        df = self.get_results_dataframe()
        print("\n=== BENCHMARK RESULTS SUMMARY ===")
        print(df.to_string(index=False, float_format='{:.4f}'.format))
        
        print("\n=== TOP PERFORMERS BY METRIC ===")
        for metric in ['Combined Score', 'ROUGE-1 F1', 'BERT F1', 'Avg Time (s)', 'Avg Cost ($)']:
            if metric in ['Avg Time (s)', 'Avg Cost ($)']:
                best = df.loc[df[metric].idxmin()]
                print(f"Best {metric}: {best['Method']} ({best[metric]:.6f})")
            else:
                best = df.loc[df[metric].idxmax()]
                print(f"Best {metric}: {best['Method']} ({best[metric]:.4f})")
    
    def run_statistical_analysis(self, datasets: List[str], methods: List[str]) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis on benchmark results.
        
        Args:
            datasets: List of dataset names
            methods: List of method names
            
        Returns:
            Dictionary with statistical analysis results per dataset
        """
        if not self.statistical_analyzer:
            print("Statistical analysis is disabled.")
            return {}
        
        print("\n🔬 Running Statistical Analysis...")
        print("=" * 60)
        
        analysis_results = {}
        
        for dataset_name in datasets:
            print(f"\nAnalyzing {dataset_name}...")
            
            # Combine detailed results from all methods
            combined_df = combine_detailed_results(dataset_name, methods)
            
            if combined_df.empty:
                print(f"⚠️  No detailed results found for {dataset_name}")
                continue
            
            # Run statistical analysis
            dataset_analysis = self.statistical_analyzer.analyze_dataset_results(combined_df)
            analysis_results[dataset_name] = dataset_analysis
            
            # Generate and display report
            report = self.statistical_analyzer.generate_statistical_report(
                dataset_analysis, dataset_name
            )
            print(report)
        
        return analysis_results
    
    def save_statistical_results(self, analysis_results: Dict[str, Any]):
        """
        Save statistical analysis results to file.
        
        Args:
            analysis_results: Results from run_statistical_analysis
            filename: Output filename
        """
        if not analysis_results:
            print("No statistical analysis results to save.")
            return
        
        # Convert results to serializable format
        serializable_results = {}
        
        for dataset_name, dataset_results in analysis_results.items():
            dataset_dict = {
                'best_method': dataset_results.get('best_method'),
                'best_combined_rouge': dataset_results.get('best_combined_rouge'),
                'n_methods': dataset_results.get('n_methods'),
                'confidence_level': dataset_results.get('confidence_level'),
                'bootstrap_results': {},
                'significance_tests': []
            }
            
            # Convert bootstrap results
            bootstrap_results = dataset_results.get('bootstrap_results', {})
            for method, method_results in bootstrap_results.items():
                dataset_dict['bootstrap_results'][method] = {}
                for metric, result in method_results.items():
                    dataset_dict['bootstrap_results'][method][metric] = {
                        'mean': result.mean,
                        'ci_lower': result.ci_lower,
                        'ci_upper': result.ci_upper,
                        'std_error': result.std_error,
                        'n_samples': result.n_samples
                    }
            
            # Convert significance test results
            significance_tests = dataset_results.get('significance_tests', [])
            for test in significance_tests:
                dataset_dict['significance_tests'].append({
                    'method_a': test.method_a,
                    'method_b': test.method_b,
                    'metric': test.metric,
                    'p_value': test.p_value,
                    'corrected_p_value': test.corrected_p_value,
                    'effect_size': test.effect_size,
                    'ci_lower': test.ci_lower,
                    'ci_upper': test.ci_upper,
                    'is_significant': test.is_significant
                })
            
            serializable_results[dataset_name] = dataset_dict
        
            # Save to file
            filename = f'statistical_analysis_{dataset_name}.json'
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
            print(f"Statistical analysis results saved to {filename}")
    
    def print_summary_with_statistics(self, analysis_results: Dict[str, Any] = None):
        """
        Print summary with statistical analysis if available.
        
        Args:
            analysis_results: Optional statistical analysis results
        """
        # Print standard summary first
        self.print_summary()
        
        # Add statistical summary if available
        if analysis_results and self.statistical_analyzer:
            print("\n" + "="*60)
            print("STATISTICAL SUMMARY")
            print("="*60)
            
            for dataset_name, dataset_results in analysis_results.items():
                best_method = dataset_results.get('best_method')
                best_rouge = dataset_results.get('best_combined_rouge')
                n_methods = dataset_results.get('n_methods', 0)
                
                print(f"\n{dataset_name.upper()}:")
                if best_method and best_rouge is not None:
                    print(f"  Best Method: {best_method} (Combined_ROUGE: {best_rouge:.4f})")
                
                # Count significant differences
                significance_tests = dataset_results.get('significance_tests', [])
                n_significant = sum(1 for test in significance_tests if test.is_significant)
                
                print(f"  Methods Compared: {n_methods}")
                print(f"  Significant Differences: {n_significant}/{len(significance_tests)}")
                
                if significance_tests:
                    print(f"  Significance Tests: {len(significance_tests)} (Holm-Bonferroni corrected)")
    
    def run_comprehensive_benchmark_with_statistics(self, 
                                                  datasets: List[str] = ['cnn_dailymail', 'arxiv'],
                                                  methods: Optional[List[str]] = None,
                                                  max_samples: int = 50,
                                                  max_sentences: int = 3) -> tuple:
        """
        Run comprehensive benchmark with statistical analysis.
        
        Returns:
            Tuple of (benchmark_results, statistical_analysis_results)
        """
        # Run standard benchmark
        benchmark_results = self.run_comprehensive_benchmark(
            datasets=datasets,
            methods=methods, 
            max_samples=max_samples,
            max_sentences=max_sentences
        )
        
        # Run statistical analysis if enabled
        statistical_results = {}
        if self.enable_statistical_analysis and methods:
            statistical_results = self.run_statistical_analysis(datasets, methods)
            
            # Save statistical results
            self.save_statistical_results(statistical_results)
        
        return benchmark_results, statistical_results