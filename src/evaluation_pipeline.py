"""Step 2 of the pipeline: score generated summaries and compare methods.

Reads the summaries_<dataset>_<method>.csv files written by step 1 (from any
number of run directories, e.g. the CPU and GPU instances' outputs), computes
ROUGE, BERTScore and BLEURT per document, then the cross-method Z-scores,
composite quality and bootstrap statistics over all methods of a dataset at
once. BERTScore and BLEURT are model-based, so this step belongs on the GPU.
"""
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .evaluation_metrics import EvaluationMetrics

SCORE_COLUMNS = ['rouge1_precision', 'rouge1_recall', 'rouge1_f1',
                 'rouge2_precision', 'rouge2_recall', 'rouge2_f1',
                 'rougeL_precision', 'rougeL_recall', 'rougeL_f1',
                 'bert_precision', 'bert_recall', 'bert_f1', 'bleurt_score']

COLUMN_NAMES = {
    'rouge1_f1': 'ROUGE-1 F1',
    'rouge2_f1': 'ROUGE-2 F1',
    'rougeL_f1': 'ROUGE-L F1',
    'bert_f1': 'BERT F1',
    'bleurt_score': 'BLEURT Score',
    'combined_zscore': 'Combined Zscore',
    'processing_time': 'Avg Time (s)',
    'cost': 'Avg Cost ($)',
}


def find_summaries(input_dirs: List[str], dataset_name: str,
                   methods: Optional[List[str]] = None) -> Dict[str, str]:
    """Map method -> summaries file for one dataset, across all input dirs.

    Looks only in <input_dir>/<dataset>/, not recursively, so the chunked
    variants (written to their own run directory) never enter the main
    comparison by accident."""
    prefix = f'summaries_{dataset_name}_'
    found = {}
    for input_dir in input_dirs:
        for path in sorted(glob.glob(os.path.join(input_dir, dataset_name, f'{prefix}*.csv'))):
            method = os.path.basename(path)[len(prefix):-4]
            if methods and method not in methods:
                continue
            if method in found:
                raise ValueError(f"{method} on {dataset_name} found twice: {found[method]} and {path}")
            found[method] = path
    if methods:
        missing = [m for m in methods if m not in found]
        if missing:
            raise ValueError(f"No summaries for {missing} on {dataset_name} in {input_dirs}")
    return found


class EvaluationPipeline:
    def __init__(self, output_dir: str = 'results', n_bootstrap: int = 10000,
                 confidence_level: float = 0.95, force: bool = False):
        self.output_dir = output_dir
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.force = force
        self.evaluator = EvaluationMetrics()

    def score_method(self, dataset_name: str, method: str, summaries_file: str) -> pd.DataFrame:
        """Per-document metrics for one method, cached as detailed_results_<dataset>_<method>.csv."""
        out_file = os.path.join(self.output_dir, dataset_name, f'detailed_results_{dataset_name}_{method}.csv')
        summaries = pd.read_csv(summaries_file)

        if not self.force and os.path.exists(out_file) and \
                os.path.getmtime(out_file) > os.path.getmtime(summaries_file):
            cached = pd.read_csv(out_file)
            if list(cached['reference_summary_sha256']) == list(summaries['reference_summary_sha256']):
                print(f"⏭ {method} on {dataset_name}: using cached scores in {out_file}")
                return cached

        print(f"\n📏 Scoring {method} on {dataset_name} ({len(summaries)} summaries)")
        references = summaries['reference_summary'].fillna('').astype(str).tolist()
        candidates = summaries['generated_summary'].fillna('').astype(str).tolist()
        scores = self.evaluator.evaluate_batch_summaries(references, candidates)['individual_scores']

        detailed = summaries.copy()
        for column in SCORE_COLUMNS:
            detailed[column] = scores[column]
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        detailed.to_csv(out_file, index=False)
        return detailed

    def analyse_dataset(self, dataset_name: str, detailed: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Cross-method Z-scores, composite quality and bootstrap statistics."""
        # Paired comparisons need every method scored on the same documents in the same order
        reference_order = None
        for method, df in detailed.items():
            order = list(df['reference_summary_sha256'])
            if reference_order is None:
                reference_order, first = order, method
            elif order != reference_order:
                raise ValueError(f"{method} and {first} on {dataset_name} were not run on the same "
                                 f"documents in the same order; regenerate from the manifest.")

        per_doc, method_results = {}, {}
        for method, df in detailed.items():
            per_doc[method] = {k: df[k].tolist() for k in
                               ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bert_f1', 'bleurt_score',
                                'processing_time', 'cost']}
            method_results[method] = {
                'average_scores': {f'{k}_mean': float(np.mean(v)) for k, v in per_doc[method].items()},
            }

        # Per-metric Z-scores for the table; the composite and all CIs/tests come
        # from the paired bootstrap, whose point estimate is the same composite
        zscores = self.evaluator.compute_cross_method_zscore(method_results)
        analysis = self.evaluator.paired_bootstrap_analysis(
            per_doc, dataset_name, n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level)
        print(self.evaluator.generate_comprehensive_report(analysis))

        dataset_dir = os.path.join(self.output_dir, dataset_name)
        with open(os.path.join(dataset_dir, f'comprehensive_statistical_analysis_{dataset_name}.json'), 'w') as f:
            json.dump(_json_compatible(analysis), f, indent=2)

        table = self._results_table(analysis, zscores)
        table.to_csv(os.path.join(dataset_dir, f'benchmark_results_with_CI_{dataset_name}.csv'),
                     index=False, float_format='%.6f')
        return table

    def _results_table(self, analysis: Dict, zscores: Dict) -> pd.DataFrame:
        """One row per method: metric means with paired-bootstrap CIs, per-metric
        Z-scores, and the paired test against the best method on the composite."""
        rows = []
        metrics = analysis['metrics_analysis']
        for method in metrics['rouge1_f1']['bootstrap_results']:
            row = {'Method': method, 'Dataset': analysis['dataset']}
            for key, result in metrics.items():
                col = COLUMN_NAMES.get(key, result['metric_name'])
                ci = result['bootstrap_results'][method]
                row[col] = ci['mean']
                row[f'{col} CI Lower'] = ci['ci_lower']
                row[f'{col} CI Upper'] = ci['ci_upper']
                row[f'{col} Std Error'] = ci['std_error']
                row[f'{col} Sample Size'] = ci['n_samples']
            for key, z in zscores[method].items():
                if key.endswith('_zscore') and key != 'combined_zscore':
                    row[f'{COLUMN_NAMES.get(key[:-7], key[:-7])} Z'] = z
            composite = metrics['combined_zscore']
            test = composite['statistical_tests'][method]
            row['Rank-1 Share'] = composite['rank1_share'][method]
            row['Is Best Method'] = test['is_best']
            row['P-value vs Best'] = test['p_value'] if test['p_value'] is not None else 'N/A'
            row['Diff vs Best CI Lower'] = test.get('difference_ci_lower', 0.0)
            row['Diff vs Best CI Upper'] = test.get('difference_ci_upper', 0.0)
            row['Effect Size'] = test['effect_size']
            row['Significantly Different'] = test['significantly_different']
            row['Statistical Interpretation'] = test['interpretation']
            row['Total Time (s)'] = row['Avg Time (s)'] * row['Avg Time (s) Sample Size']
            row['Total Cost ($)'] = row['Avg Cost ($)'] * row['Avg Cost ($) Sample Size']
            rows.append(row)
        return pd.DataFrame(rows).sort_values('Combined Zscore', ascending=False)

    def run(self, input_dirs: List[str], datasets: List[str],
            methods: Optional[List[str]] = None) -> pd.DataFrame:
        tables = []
        for dataset_name in datasets:
            print(f"\n=== {dataset_name} ===")
            files = find_summaries(input_dirs, dataset_name, methods)
            if not files:
                print(f"⚠ No summaries for {dataset_name} in {input_dirs}; skipping")
                continue
            for method, path in files.items():
                print(f"  {method:20} {path}")
            detailed = {m: self.score_method(dataset_name, m, p) for m, p in files.items()}
            if len(detailed) < 2:
                print(f"⚠ Only {len(detailed)} method for {dataset_name}; no cross-method statistics")
                continue
            tables.append(self.analyse_dataset(dataset_name, detailed))

        summary = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if not summary.empty:
            os.makedirs(self.output_dir, exist_ok=True)
            summary.to_csv(os.path.join(self.output_dir, 'benchmark_results.csv'), index=False, float_format='%.6f')
        return summary


def _json_compatible(obj):
    if isinstance(obj, dict):
        return {k: _json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_compatible(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
