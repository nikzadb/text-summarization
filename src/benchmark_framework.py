import time
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .summarizers.traditional import TextRankSummarizer, TFIDFRankSummarizer
from .summarizers.llm import OpenSourceLLMSummarizer, T5Summarizer, DistilBARTSummarizer
from .summarizers.gemini import GeminiSummarizer, HybridTFIDFRankGeminiSummarizer, HybridTextRankGeminiSummarizer
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
        self.summarizers = {
            'textrank': TextRankSummarizer(),
            'tfidfrank': TFIDFRankSummarizer(),
            't5': T5Summarizer(),
            'distilbart': DistilBARTSummarizer(),
            'bart': OpenSourceLLMSummarizer('facebook/bart-large-cnn')
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