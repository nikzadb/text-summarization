from abc import ABC, abstractmethod
import time
from typing import Dict, Any
import numpy as np
from gensim.summarization import summarize as gensim_summarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class BaseSummarizer(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        pass
    
    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        summary = self.summarize(text, max_sentences)
        end_time = time.time()
        
        return {
            'summary': summary,
            'time_taken': end_time - start_time,
            'method': self.name,
            'cost': self.get_cost()
        }
    
    def get_cost(self) -> float:
        return 0.0


class TextRankSummarizer(BaseSummarizer):
    def __init__(self):
        super().__init__("TextRank")
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        try:
            word_count = len(text.split())
            ratio = min(0.5, (max_sentences * 20) / word_count)
            summary = gensim_summarize(text, ratio=ratio)
            if not summary:
                sentences = text.split('.')[:max_sentences]
                return '. '.join(sentences) + '.'
            return summary
        except Exception as e:
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'


class TFIDFRankSummarizer(BaseSummarizer):
    def __init__(self):
        super().__init__("TFIDFRank")
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            G = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(G)
            
            ranked_sentences = sorted(
                [(scores[i], sentences[i]) for i in range(len(sentences))],
                reverse=True
            )
            
            selected_sentences = [sent for _, sent in ranked_sentences[:max_sentences]]
            return '. '.join(selected_sentences) + '.'
        except Exception as e:
            return '. '.join(sentences[:max_sentences]) + '.'