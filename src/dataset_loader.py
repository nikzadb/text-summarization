from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import csv
import hashlib
import os
import random

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKIHOW_CSV = os.path.join(REPO_ROOT, 'data', 'wikihowAll.csv')
MANIFEST_DIR = os.path.join(REPO_ROOT, 'manifests')
# Frozen evaluation sets written by scripts/export_eval_data.py
EVAL_DATA_DIR = os.path.join(REPO_ROOT, 'eval_data')


def reference_hash(text: str) -> str:
    return hashlib.sha256((text or '').strip().encode('utf-8')).hexdigest()


def eval_data_path(dataset_type: str, eval_data_dir: str = EVAL_DATA_DIR) -> Optional[str]:
    """The frozen evaluation set for a dataset (.csv or zipped .csv.zip), if present."""
    for ext in ('.csv', '.csv.zip'):
        path = os.path.join(eval_data_dir, dataset_type + ext)
        if os.path.exists(path):
            return path
    return None


class DatasetLoader:
    def __init__(self, dataset_name: str, subset: Optional[str] = None, seed: int = 42):
        self.dataset_name = dataset_name
        self.subset = subset
        self.seed = seed
        self.dataset = None
        self.loaded_data = []
    
    def _sample_indices(self, n: int, k: int) -> List[int]:
        # A fresh generator per call, so a dataset's sample doesn't depend on
        # which datasets were loaded before it by the same loader
        return random.Random(self.seed).sample(range(n), k)

    def load_cnn_dailymail(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading CNN/DailyMail dataset ({split} split)...")
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = self._sample_indices(len(dataset), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            data.append({
                'id': item.get('id', ''),
                'article': item['article'],
                'highlights': item['highlights'],
                'reference_summary': item['highlights']
            })
        
        self.loaded_data = data
        return data
    
    def load_arxiv_papers(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading arXiv scientific papers dataset ({split} split)...")
        try:
            dataset = load_dataset('scientific_papers', 'arxiv', split=split)
        except:
            print("Note: scientific_papers dataset might require manual download.")
            dataset = load_dataset('ccdv/arxiv-summarization', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = self._sample_indices(len(dataset), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            abstract = item.get('abstract', item.get('summary', ''))
            article = item.get('article', item.get('text', ''))
            
            data.append({
                'id': item.get('id', f"arxiv_{len(data)}"),
                'article': article,
                'abstract': abstract,
                'reference_summary': abstract
            })
        
        self.loaded_data = data
        return data
    
    def load_wikihow(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading WikiHow dataset ({split} split)...")
        # Try the cleaned version first
        # dataset = load_dataset('gursi26/wikihow-cleaned', split=split)
        import pandas as pd
        dataset = pd.read_csv(WIKIHOW_CSV)
        dataset = dataset.dropna(subset=['headline', 'text'])
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.sample(max_samples, random_state=self.seed)

        dataset = dataset.to_dict(orient='records')

        data = []

        for item in dataset:
            
            data.append({
                'id': item.get('id', f"wikihow_{len(data)}"),
                'article': item['text'],
                'summary': item['headline'], 
                'reference_summary': item['headline']
            })
        
        self.loaded_data = data
        return data
    
    def load_govreport(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading GovReport dataset ({split} split)...")
        dataset = load_dataset('ccdv/govreport-summarization', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = self._sample_indices(len(dataset), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            data.append({
                'id': item.get('id', f"govreport_{len(data)}"),
                'article': item['report'],
                'summary': item['summary'],
                'reference_summary': item['summary']
            })
        
        self.loaded_data = data
        return data
    
    def load_samsum(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading SAMSum dataset ({split} split)...")
        try:
            dataset = load_dataset('Samsung/samsum', split=split)
        except:
            try:
                dataset = load_dataset('knkarthick/samsum', split=split)
            except:
                dataset = load_dataset('nyamuda/samsum', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = self._sample_indices(len(dataset), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            data.append({
                'id': item.get('id', f"samsum_{len(data)}"),
                'article': item['dialogue'],
                'summary': item['summary'],
                'reference_summary': item['summary']
            })
        
        self.loaded_data = data
        return data
    
    def load_qmsum(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading QMSum dataset ({split} split)...")
        try:
            dataset = load_dataset('ioeddk/qmsum', split=split)
        except:
            try:
                dataset = load_dataset('mattercalm/qmsum', split=split)
            except:
                dataset = load_dataset('pszemraj/qmsum-cleaned', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = self._sample_indices(len(dataset), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:

            # QMSum has meeting transcripts and query-based summaries
            article = item.get('text', '')
            summary = item.get('answer', '')
            
            data.append({
                'id': item.get('id', f"qmsum_{len(data)}"),
                'article': article,
                'summary': summary,
                'reference_summary': summary
            })
        
        self.loaded_data = data
        return data
    
    def load_mediasum(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading MediaSum dataset ({split} split)...")
        try:
            dataset = load_dataset('ccdv/mediasum', split=split)
        except:
            dataset = load_dataset('nbroad/mediasum', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = self._sample_indices(len(dataset), max_samples)
            dataset = dataset.select(indices)

        data = []
        for item in dataset:
            # MediaSum has interview transcripts and summaries
            transcript = item.get('document', '')
            summary = item.get('summary', '')
            
            data.append({
                'id': item.get('id', f"mediasum_{len(data)}"),
                'article': transcript,
                'summary': summary,
                'reference_summary': summary
            })
        
        self.loaded_data = data
        return data
    
    def load_callsum(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading Aiera Earnings Call Summarization dataset ({split} split)...")
        dataset = load_dataset('Aiera/aiera-ect-sum', split=split)

        if max_samples and max_samples < len(dataset):
            indices = self._sample_indices(len(dataset), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            # MediaSum has interview transcripts and summaries
            transcript = item.get('transcript', '')
            summary = item.get('summary', '')
            
            data.append({
                'id': item.get('id', f"callsum_{len(data)}"),
                'article': transcript,
                'summary': summary,
                'reference_summary': summary
            })
        
        self.loaded_data = data
        return data

    def load_dataset(self, dataset_type: str, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        if dataset_type.lower() in ['cnn_dailymail', 'cnn', 'dailymail']:
            return self.load_cnn_dailymail(split, max_samples)
        elif dataset_type.lower() in ['arxiv', 'scientific_papers']:
            return self.load_arxiv_papers(split, max_samples)
        elif dataset_type.lower() in ['wikihow']:
            return self.load_wikihow(split, max_samples)
        elif dataset_type.lower() in ['govreport']:
            return self.load_govreport(split, max_samples)
        elif dataset_type.lower() in ['samsum']:
            return self.load_samsum(split, max_samples)
        elif dataset_type.lower() in ['qmsum']:
            return self.load_qmsum(split, max_samples)
        elif dataset_type.lower() in ['mediasum']:
            return self.load_mediasum(split, max_samples)
        elif dataset_type.lower() in ['callsum']:
            return self.load_callsum(split, max_samples)        
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def load_fixed_manifest(self, dataset_type: str, manifest_path: str, split: str = 'test',
                            max_samples: int = 0) -> List[Dict[str, str]]:
        """Load exactly the documents listed in a manifest (see
        scripts/export_document_manifest.py), in manifest order, so new
        architectures are scored on the same documents as the original runs.
        max_samples > 0 takes the first N manifest documents, so smaller runs
        are always a prefix of the full evaluation set."""
        with open(manifest_path, newline='') as f:
            wanted = [row['reference_summary_sha256'] for row in csv.DictReader(f)]
        if max_samples and max_samples < len(wanted):
            wanted = wanted[:max_samples]

        # max_samples=0 disables sampling, i.e. loads the whole source split
        by_hash = {}
        for item in self.load_dataset(dataset_type, split=split, max_samples=0):
            by_hash.setdefault(reference_hash(item['reference_summary']), item)

        data = [by_hash[h] for h in wanted if h in by_hash]
        missing = len(wanted) - len(data)
        if missing:
            raise ValueError(
                f"{missing}/{len(wanted)} manifest documents not found in {dataset_type} "
                f"({split} split); the source data differs from the original run's."
            )
        print(f"Loaded {len(data)} documents for {dataset_type} from manifest {manifest_path}")
        self.loaded_data = data
        return data

    def load_evaluation_set(self, dataset_type: str, max_samples: int = 0,
                            manifest_dir: Optional[str] = MANIFEST_DIR,
                            eval_data_dir: str = EVAL_DATA_DIR) -> List[Dict[str, str]]:
        """The documents every experiment runs on, from the first source available:

        1. eval_data/<dataset>.csv(.zip): the frozen evaluation set, checked
           against the manifest (no download, no dependency on dataset versions);
        2. manifests/<dataset>_manifest.csv: the same documents, located in the
           source dataset through the loaders above;
        3. seeded sampling, for datasets without a manifest (not the original set).

        max_samples > 0 takes the first N documents.
        """
        manifest_path = os.path.join(manifest_dir, f'{dataset_type}_manifest.csv') if manifest_dir else None
        has_manifest = bool(manifest_path and os.path.exists(manifest_path))

        frozen = eval_data_path(dataset_type, eval_data_dir)
        if frozen:
            # keep_default_na: empty strings must stay strings, not become NaN
            df = pd.read_csv(frozen, keep_default_na=False)
            hashes = [reference_hash(r) for r in df['reference_summary']]
            if 'reference_summary_sha256' in df and hashes != list(df['reference_summary_sha256']):
                raise ValueError(f"{frozen} is corrupted: reference summaries don't match their hashes")
            if has_manifest:
                with open(manifest_path, newline='') as f:
                    wanted = [row['reference_summary_sha256'] for row in csv.DictReader(f)]
                if hashes != wanted:
                    raise ValueError(f"{frozen} doesn't match {manifest_path}; re-export it with "
                                     f"scripts/export_eval_data.py")
            if max_samples:
                df = df.head(max_samples)
            data = [{'id': r.sample_id, 'article': r.text, 'reference_summary': r.reference_summary}
                    for r in df.itertuples(index=False)]
            print(f"Loaded {len(data)} documents for {dataset_type} from {frozen}")
            self.loaded_data = data
            return data

        if has_manifest:
            print(f"No frozen evaluation set for {dataset_type} in {eval_data_dir}; "
                  f"loading the manifest documents from the source dataset")
            return self.load_fixed_manifest(dataset_type, manifest_path, max_samples=max_samples)

        print(f"⚠ No manifest for {dataset_type}; using seeded sampling (seed={self.seed}). These "
              f"documents are reproducible across runs but are NOT the original evaluation set.")
        return self.load_dataset(dataset_type, max_samples=max_samples)

    def get_sample_data(self, n_samples: int = 10) -> List[Dict[str, str]]:
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        return random.Random(self.seed).sample(self.loaded_data, min(n_samples, len(self.loaded_data)))
    
    def get_data_statistics(self) -> Dict[str, float]:
        if not self.loaded_data:
            return {}
        
        article_lengths = [len(item['article'].split()) for item in self.loaded_data]
        summary_lengths = [len(item['reference_summary'].split()) for item in self.loaded_data]

        return {
            'total_samples': len(self.loaded_data),
            'avg_article_length': sum(article_lengths) / len(article_lengths),
            'avg_summary_length': sum(summary_lengths) / len(summary_lengths),
            'max_article_length': max(article_lengths),
            'min_article_length': min(article_lengths),
            'max_summary_length': max(summary_lengths),
            'min_summary_length': min(summary_lengths)
        }