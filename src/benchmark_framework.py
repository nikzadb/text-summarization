import time
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .summarizers.traditional import TextRankSummarizer, TFIDFRankSummarizer
from .summarizers.llm import OpenSourceLLMSummarizer, T5Summarizer, DistilBARTSummarizer
from .summarizers.gemini import GeminiSummarizer, HybridTFIDFRankGeminiSummarizer, HybridTextRankGeminiSummarizer
from .summarizers.openai_gpt import GPT5MiniSummarizer
from .dataset_loader import DatasetLoader
from .evaluation_metrics import EvaluationMetrics
from .lambda_simulation import LambdaSimulator


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
    def __init__(self, use_lambda_simulation: bool = True):
        self.use_lambda_simulation = use_lambda_simulation
        self.summarizers = {}
        self.evaluator = EvaluationMetrics()
        self.results = []
        
        self._initialize_summarizers()
    
    def _initialize_summarizers(self):
        """Initialize only lightweight traditional methods at startup.
        Heavy models will be loaded on-demand and cleaned up after use."""
        self.summarizers = {
            'textrank': TextRankSummarizer(),
            'tfidfrank': TFIDFRankSummarizer(),
        }
        
        # Define heavy models that require on-demand loading
        self.heavy_model_classes = {
            'distilbart': lambda: DistilBARTSummarizer(),
            'bart': lambda: OpenSourceLLMSummarizer('facebook/bart-large-cnn'),
            't5': lambda: T5Summarizer(),
            'gemini': lambda: GeminiSummarizer(),
            'GPT-5-mini': lambda: GPT5MiniSummarizer(),
            'hybrid_textrank_gemini': lambda: HybridTextRankGeminiSummarizer(),
            'hybrid_tfidfrank_gemini': lambda: HybridTFIDFRankGeminiSummarizer()
        }
    
    def benchmark_method(self, 
                        method_name: str, 
                        dataset_samples: List[Dict[str, str]], 
                        dataset_name: str,
                        max_sentences: int = 3) -> BenchmarkResult:
        
        # Check if method exists in either lightweight or heavy models
        if method_name not in self.summarizers and method_name not in self.heavy_model_classes:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Load model on-demand if it's a heavy model
        summarizer = None
        is_heavy_model = method_name in self.heavy_model_classes
        
        if is_heavy_model:
            print(f"🔄 Loading {method_name} model...")
            try:
                summarizer = self.heavy_model_classes[method_name]()
                print(f"✓ {method_name} model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {method_name}: {e}")
                raise
        else:
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
        
        # Clean up heavy model resources after benchmarking
        if is_heavy_model and summarizer:
            print(f"🧹 Cleaning up {method_name} model...")
            try:
                summarizer.cleanup()
                del summarizer
                summarizer = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                print(f"✓ {method_name} cleanup completed")
            except Exception as e:
                print(f"Warning: Error during {method_name} cleanup: {e}")
        
        # Save intermediate results after each method
        self._save_intermediate_results()
        
        return result
    
    def run_comprehensive_benchmark(self, 
                                  datasets: List[str] = ['cnn_dailymail', 'arxiv'],
                                  methods: Optional[List[str]] = None,
                                  max_samples: int = 50,
                                  max_sentences: int = 3,
                                  perform_statistical_analysis: bool = True) -> List[BenchmarkResult]:
        
        if methods is None:
            methods = list(self.summarizers.keys()) + list(self.heavy_model_classes.keys())
        
        loader = DatasetLoader('benchmark')
        all_results = []
        
        for dataset_name in datasets:
            print(f"\n=== Loading {dataset_name} dataset ===")
            try:
                dataset_samples = loader.load_dataset(dataset_name, max_samples=max_samples)
                dataset_stats = loader.get_data_statistics()
                print(f"Dataset stats: {dataset_stats}")
                
                for method in methods:
                    if method in self.summarizers or method in self.heavy_model_classes:
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
        
        # Perform statistical analysis if requested
        if perform_statistical_analysis and all_results:
            print("\n" + "=" * 60)
            print("🔬 PERFORMING STATISTICAL ANALYSIS")
            print("=" * 60)
            self._perform_statistical_analysis_by_dataset(methods)
        
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
    
    def _perform_statistical_analysis_by_dataset(self, methods: List[str]):
        """
        Perform statistical analysis for each dataset separately.
        Groups evaluation results by dataset and performs bootstrap confidence interval
        analysis and statistical comparison with the best method.
        """
        import glob
        
        # Group results by dataset
        datasets_processed = set()
        
        for result in self.results:
            dataset_name = result.dataset
            if dataset_name in datasets_processed:
                continue
                
            datasets_processed.add(dataset_name)
            print(f"\n📊 Analyzing dataset: {dataset_name}")
            print("-" * 50)
            
            # Load detailed results for this dataset
            method_results = {}
            for method in methods:
                detailed_file = f"detailed_results_{dataset_name}_{method}.csv"
                
                if glob.glob(detailed_file):
                    try:
                        # Load detailed CSV results
                        detailed_df = pd.read_csv(detailed_file)
                        
                        # Convert to evaluation format (including performance metrics)
                        individual_scores = {
                            'rouge1_f1': detailed_df['rouge1_f1'].tolist(),
                            'rouge2_f1': detailed_df['rouge2_f1'].tolist(),
                            'rougeL_f1': detailed_df['rougeL_f1'].tolist(),
                            'bert_f1': detailed_df['bert_f1'].tolist(),
                            'processing_time': detailed_df['processing_time'].tolist(),
                            'cost': detailed_df['cost'].tolist()
                        }
                        
                        # Calculate average scores for compatibility
                        average_scores = {}
                        for key, values in individual_scores.items():
                            average_scores[f"{key}_mean"] = np.mean(values)
                            average_scores[f"{key}_std"] = np.std(values)
                        
                        method_results[method] = {
                            'individual_scores': individual_scores,
                            'average_scores': average_scores,
                            'sample_count': len(detailed_df)
                        }
                        
                        print(f"✓ Loaded {len(detailed_df)} samples for {method}")
                        
                    except Exception as e:
                        print(f"⚠️  Could not load detailed results for {method}: {e}")
                        continue
                else:
                    print(f"⚠️  No detailed results found for {method}")
            
            # Perform statistical analysis if we have data
            if len(method_results) >= 2:
                print(f"\n🔬 Statistical Analysis for {dataset_name}")
                print("=" * 50)
                
                try:
                    # Perform comprehensive statistical analysis for ALL metrics
                    comprehensive_results = self.evaluator.comprehensive_statistical_analysis(
                        method_results, dataset_name
                    )
                    
                    # Generate and display comprehensive report
                    comprehensive_report = self.evaluator.generate_comprehensive_report(comprehensive_results)
                    print(comprehensive_report)
                    
                    # Save comprehensive statistical results
                    stats_filename = f"comprehensive_statistical_analysis_{dataset_name}.json"
                    with open(stats_filename, 'w') as f:
                        # Convert numpy types to native Python types for JSON serialization
                        json_compatible_results = self._make_json_compatible(comprehensive_results)
                        json.dump(json_compatible_results, f, indent=2)
                    
                    print(f"\n💾 Comprehensive statistical analysis saved to: {stats_filename}")
                    
                    # Generate enhanced benchmark results CSV with confidence intervals
                    enhanced_csv_filename = f"benchmark_results_with_CI_{dataset_name}.csv"
                    self._generate_enhanced_benchmark_csv(comprehensive_results, enhanced_csv_filename)
                    print(f"💾 Enhanced benchmark results with CIs saved to: {enhanced_csv_filename}")
                    
                except Exception as e:
                    print(f"❌ Error during statistical analysis for {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Not enough methods ({len(method_results)}) for statistical comparison")
    
    def _make_json_compatible(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_compatible(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_compatible(item) for item in obj]
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.bool_, np.bool8, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle other numpy scalars
            return obj.item()
        else:
            return obj
    
    def _generate_enhanced_benchmark_csv(self, comprehensive_results: Dict[str, Any], filename: str):
        """
        Generate enhanced benchmark results CSV with confidence intervals for all metrics.
        
        Creates a CSV that replicates benchmark_results.csv format but adds CI columns
        for every metric (ROUGE scores, BERT score, Combined score, Time, Cost).
        """
        try:
            enhanced_data = []
            dataset_name = comprehensive_results['dataset']
            metrics_analysis = comprehensive_results['metrics_analysis']
            confidence_level = comprehensive_results['confidence_level']
            
            # Get all methods from the first metric analysis
            first_metric = list(metrics_analysis.keys())[0]
            methods = list(metrics_analysis[first_metric]['bootstrap_results'].keys())
            
            for method in methods:
                row_data = {
                    'Method': method,
                    'Dataset': dataset_name
                }
                
                # Add each metric with its confidence interval
                for metric_key, analysis in metrics_analysis.items():
                    bootstrap_info = analysis['bootstrap_results'][method]
                    metric_name = analysis['metric_name']
                    
                    # Format metric name for column headers
                    if metric_key == 'rouge1_f1':
                        base_col = 'ROUGE-1 F1'
                    elif metric_key == 'rouge2_f1':
                        base_col = 'ROUGE-2 F1'
                    elif metric_key == 'rougeL_f1':
                        base_col = 'ROUGE-L F1'
                    elif metric_key == 'bert_f1':
                        base_col = 'BERT F1'
                    elif metric_key == 'combined_score':
                        base_col = 'Combined Score'
                    elif metric_key == 'processing_time':
                        base_col = 'Avg Time (s)'
                    elif metric_key == 'cost':
                        base_col = 'Avg Cost ($)'
                    else:
                        base_col = metric_name
                    
                    # Add mean value
                    row_data[base_col] = bootstrap_info['mean']
                    
                    # Add confidence interval bounds
                    row_data[f'{base_col} CI Lower'] = bootstrap_info['ci_lower']
                    row_data[f'{base_col} CI Upper'] = bootstrap_info['ci_upper']
                    
                    # Add standard error
                    row_data[f'{base_col} Std Error'] = bootstrap_info['std_error']
                    
                    # Add sample size
                    row_data[f'{base_col} Sample Size'] = bootstrap_info['n_samples']
                
                # Add statistical significance information
                for metric_key, analysis in metrics_analysis.items():
                    test_info = analysis['statistical_tests'][method]
                    metric_name = analysis['metric_name']
                    
                    if metric_key == 'combined_score':  # Use combined score as primary significance indicator
                        row_data['Is Best Method'] = test_info['is_best']
                        row_data['P-value vs Best'] = test_info['p_value'] if test_info['p_value'] is not None else 'N/A'
                        row_data['Effect Size'] = test_info['effect_size']
                        row_data['Significantly Different'] = test_info['significantly_different']
                        row_data['Statistical Interpretation'] = test_info['interpretation']
                
                enhanced_data.append(row_data)
            
            # Create DataFrame and save to CSV
            enhanced_df = pd.DataFrame(enhanced_data)
            
            # Sort by combined score (descending) for readability
            enhanced_df = enhanced_df.sort_values('Combined Score', ascending=False)
            
            # Save to CSV with proper formatting
            enhanced_df.to_csv(filename, index=False, float_format='%.6f')
            
            print(f"✓ Enhanced CSV generated with {len(enhanced_data)} methods and {confidence_level*100:.0f}% confidence intervals")
            
        except Exception as e:
            print(f"❌ Error generating enhanced CSV: {e}")
            import traceback
            traceback.print_exc()