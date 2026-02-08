from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from typing import List, Dict, Tuple, Any
import warnings
from transformers import logging
import torch
from tqdm import tqdm

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
        
        return {**rouge_scores, **bert_metrics}
    
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
        
        all_scores = {**rouge_scores, **bert_scores}
        
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
            'combined_score': (
                avg_scores.get('rouge1_f1_mean', 0.0) * 0.25 +
                avg_scores.get('rouge2_f1_mean', 0.0) * 0.25 +
                avg_scores.get('rougeL_f1_mean', 0.0) * 0.25 +
                avg_scores.get('bert_f1_mean', 0.0) * 0.25
            )
        }
    
    def compare_methods(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        comparison = {}
        
        for method_name, evaluation_result in results.items():
            comparison[method_name] = self.get_summary_statistics(evaluation_result)
        
        return comparison