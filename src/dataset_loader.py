from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import random


class DatasetLoader:
    def __init__(self, dataset_name: str, subset: Optional[str] = None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = None
        self.loaded_data = []
    
    def load_cnn_dailymail(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading CNN/DailyMail dataset ({split} split)...")
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = random.sample(range(len(dataset)), max_samples)
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
            indices = random.sample(range(len(dataset)), max_samples)
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
        try:
            # Try the cleaned version first
            dataset = load_dataset('gursi26/wikihow-cleaned', split=split)
        except:
            try:
                # Fall back to sentence-transformers version
                dataset = load_dataset('sentence-transformers/wikihow', split=split)
            except:
                # Final fallback to wikihow with train split (as it might not have test)
                print("Note: Using train split as test split may not be available")
                dataset = load_dataset('gursi26/wikihow-cleaned', split='train')
        
        if max_samples and max_samples < len(dataset):
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            # Handle different possible field names
            article = item.get('article', item.get('text', item.get('input', '')))
            summary = item.get('summary', item.get('headline', item.get('target', item.get('output', ''))))
            
            data.append({
                'id': item.get('id', f"wikihow_{len(data)}"),
                'article': article,
                'summary': summary,
                'reference_summary': summary
            })
        
        self.loaded_data = data
        return data
    
    def load_govreport(self, split: str = 'test', max_samples: int = 100) -> List[Dict[str, str]]:
        print(f"Loading GovReport dataset ({split} split)...")
        dataset = load_dataset('ccdv/govreport-summarization', split=split)
        
        if max_samples and max_samples < len(dataset):
            indices = random.sample(range(len(dataset)), max_samples)
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
            indices = random.sample(range(len(dataset)), max_samples)
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
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            # QMSum has meeting transcripts and query-based summaries
            meeting_text = item.get('meeting_transcript', item.get('transcript', ''))
            query = item.get('query', '')
            summary = item.get('summary', '')
            
            # Combine query and meeting text as the article
            article = f"Query: {query}\n\nMeeting Transcript: {meeting_text}" if query else meeting_text
            
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
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)
        
        data = []
        for item in dataset:
            # MediaSum has interview transcripts and summaries
            transcript = item.get('transcript', item.get('dialogue', ''))
            summary = item.get('summary', '')
            
            data.append({
                'id': item.get('id', f"mediasum_{len(data)}"),
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
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def get_sample_data(self, n_samples: int = 10) -> List[Dict[str, str]]:
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        return random.sample(self.loaded_data, min(n_samples, len(self.loaded_data)))
    
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