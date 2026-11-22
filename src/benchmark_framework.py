"""Step 1 of the pipeline: generate summaries and measure latency and cost.

No evaluation happens here, so this step runs on whichever instance a method
is priced at (CPU for extractive and API methods, GPU for local transformers)
without loading any evaluation models. Each (dataset, method) pair is written
to its own summaries_<dataset>_<method>.csv, which evaluate.py scores later.
"""
import gc
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import torch
from tqdm import tqdm

from .summarizers.traditional import TextRankSummarizer, TFIDFRankSummarizer
from .summarizers.llm import BARTSummarizer, DistilBARTSummarizer
from .summarizers.long_context import PegasusSummarizer, LongT5Summarizer
from .summarizers.chunked import ChunkedSummarizer
from .summarizers.gemini import GeminiSummarizer
from .summarizers.openai_gpt import GPT54MiniSummarizer
from .dataset_loader import DatasetLoader

# Built on demand and cleaned up after each (dataset, method) run
SUMMARIZERS = {
    'textrank': TextRankSummarizer,
    'tfidfrank': TFIDFRankSummarizer,
    'distilbart': lambda: DistilBARTSummarizer(),
    'bart': lambda: BARTSummarizer('facebook/bart-large-cnn'),
    'pegasus': lambda: PegasusSummarizer(),
    'longt5': lambda: LongT5Summarizer(),
    'gemini': lambda: GeminiSummarizer(),
    'GPT-5.4-mini': lambda: GPT54MiniSummarizer(),
    # Chunk-and-aggregate variants for the long-document comparison
    'bart_chunked': lambda: ChunkedSummarizer(BARTSummarizer('facebook/bart-large-cnn')),
    'distilbart_chunked': lambda: ChunkedSummarizer(DistilBARTSummarizer()),
}


def summaries_path(output_dir: str, dataset_name: str, method: str) -> str:
    return os.path.join(output_dir, dataset_name, f'summaries_{dataset_name}_{method}.csv')


def _sha256(text: str) -> str:
    return hashlib.sha256((text or '').strip().encode('utf-8')).hexdigest()


def _environment() -> Dict[str, Any]:
    """Hardware and code version the summaries were generated on."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch': torch.__version__,
        'git_commit': commit,
    }


@dataclass
class GenerationResult:
    method: str
    dataset: str
    sample_count: int
    avg_time: float
    avg_cost: float
    total_time: float
    total_cost: float
    output_file: str


class BenchmarkFramework:
    def __init__(self, output_dir: str = 'runs'):
        self.output_dir = output_dir
        self.results: List[GenerationResult] = []

    def benchmark_method(self, method_name: str, dataset_samples: List[Dict[str, str]],
                         dataset_name: str, max_sentences: int = 3) -> GenerationResult:
        if method_name not in SUMMARIZERS:
            raise ValueError(f"Unknown method: {method_name}")

        print(f"🔄 Loading {method_name}...")
        summarizer = SUMMARIZERS[method_name]()
        if hasattr(summarizer, "set_dataset") and callable(getattr(summarizer, "set_dataset")):
            summarizer.set_dataset(dataset_name)

        started = pd.Timestamp.now(tz='UTC')
        rows = []
        try:
            for i, sample in enumerate(tqdm(dataset_samples, desc=f"{method_name} on {dataset_name}")):
                result = summarizer.benchmark_summarize(sample['article'], max_sentences)
                row = {
                    'sample_id': i,
                    'method': method_name,
                    'dataset': dataset_name,
                    'reference_summary_sha256': _sha256(sample['reference_summary']),
                    'reference_summary': sample['reference_summary'],
                    'generated_summary': result['summary'],
                    'processing_time': result['time_taken'],
                    'cost': result.get('cost', 0.0),
                    'timestamp_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                }
                # API summarizers also report token counts and the API/instance cost split
                for key in ('api_cost', 'aws_cost', 'input_tokens', 'output_tokens'):
                    if key in result:
                        row[key] = result[key]
                rows.append(row)
        finally:
            summarizer.cleanup()
            del summarizer
            gc.collect()

        out_file = summaries_path(self.output_dir, dataset_name, method_name)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_file, index=False)

        meta = {
            'method': method_name,
            'dataset': dataset_name,
            'max_sentences': max_sentences,
            'sample_count': len(rows),
            'started_utc': started.isoformat(),
            'finished_utc': pd.Timestamp.now(tz='UTC').isoformat(),
            **_environment(),
        }
        with open(out_file[:-4] + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        result = GenerationResult(
            method=method_name,
            dataset=dataset_name,
            sample_count=len(rows),
            avg_time=df['processing_time'].mean(),
            avg_cost=df['cost'].mean(),
            total_time=df['processing_time'].sum(),
            total_cost=df['cost'].sum(),
            output_file=out_file,
        )
        self.results.append(result)
        print(f"💾 Saved {len(rows)} summaries to {out_file}")
        return result

    def run_comprehensive_benchmark(self,
                                    datasets: List[str],
                                    methods: List[str],
                                    max_samples: int = 0,
                                    max_sentences: Union[int, Dict[str, int]] = 3,
                                    manifest_dir: Optional[str] = 'manifests',
                                    seed: int = 42,
                                    skip_existing: bool = False) -> List[GenerationResult]:
        loader = DatasetLoader('benchmark', seed=seed)

        for dataset_name in datasets:
            print(f"\n=== Loading {dataset_name} dataset ===")
            manifest_path = os.path.join(manifest_dir, f'{dataset_name}_manifest.csv') if manifest_dir else None
            if manifest_path and os.path.exists(manifest_path):
                dataset_samples = loader.load_fixed_manifest(dataset_name, manifest_path,
                                                             max_samples=max_samples)
            else:
                print(f"⚠ No manifest for {dataset_name}; using seeded sampling (seed={seed}). "
                      f"These documents are reproducible across runs but are NOT the "
                      f"original evaluation set.")
                dataset_samples = loader.load_dataset(dataset_name, max_samples=max_samples)

            n_sentences = max_sentences[dataset_name] if isinstance(max_sentences, dict) else max_sentences
            for method in methods:
                out_file = summaries_path(self.output_dir, dataset_name, method)
                if skip_existing and os.path.exists(out_file) and \
                        len(pd.read_csv(out_file)) == len(dataset_samples):
                    print(f"⏭ {method} on {dataset_name}: {out_file} already complete, skipping")
                    continue
                try:
                    result = self.benchmark_method(method, dataset_samples, dataset_name, n_sentences)
                    print(f"✓ {method}: {result.avg_time:.3f} s/doc, ${result.avg_cost:.6f}/doc")
                except Exception as e:
                    print(f"✗ Failed to benchmark {method} on {dataset_name}: {e}")

        return self.results

    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.results])

    def print_summary(self):
        if not self.results:
            print("No results to display.")
            return
        print("\n=== GENERATION SUMMARY ===")
        print(self.get_results_dataframe().drop(columns=['output_file'])
              .to_string(index=False, float_format='{:.6f}'.format))
