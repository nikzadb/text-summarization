from abc import ABC, abstractmethod
import time
from typing import Dict, Any
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


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
    """Graph-based TextRank for extractive summarization (Mihalcea & Tarau, 2004).

    Implementation details:
      1) Sentence tokenization (NLTK punkt)
      2) Sentence similarity with TF-IDF (unigrams + bigrams) + cosine similarity
      3) Weighted graph + PageRank
      4) Select top-k sentences, then restore original order for readability
    """

    def __init__(self,
                 *,
                 ngram_range=(1, 2),
                 stop_words: str = "english",
                 damping: float = 0.85,
                 sim_threshold: float = 0.0,
                 min_sentence_len: int = 10):
        super().__init__("TextRank")
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.damping = damping
        self.sim_threshold = sim_threshold
        self.min_sentence_len = min_sentence_len

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)

            # Filter very short sentences but keep mapping to original indices
            kept_sentences = []
            kept_indices = []
            for i, s in enumerate(sentences):
                if len(s.strip()) >= self.min_sentence_len:
                    kept_sentences.append(s)
                    kept_indices.append(i)

            if not kept_sentences:
                return ' '.join(sentences[:max_sentences])

            # TF-IDF sentence representations
            vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range)
            X = vectorizer.fit_transform(kept_sentences)

            # Similarity matrix
            S = cosine_similarity(X)
            np.fill_diagonal(S, 0.0)

            if self.sim_threshold > 0.0:
                S[S < self.sim_threshold] = 0.0

            # Build graph and run PageRank
            graph = nx.from_numpy_array(S)
            scores = nx.pagerank(graph, alpha=self.damping, weight='weight')

            # Rank sentences by score (descending)
            ranked_kept = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top_kept_positions = [idx for idx, _ in ranked_kept[:max_sentences]]

            # Map back to original sentence indices and restore order
            selected_original_indices = sorted(kept_indices[pos] for pos in top_kept_positions)
            return ' '.join(sentences[i] for i in selected_original_indices)

        except Exception:
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:max_sentences])


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