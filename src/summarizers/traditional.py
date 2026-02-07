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
    def __init__(self):
        super().__init__("TextRank")
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        try:
            # Preprocess text and split into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)
            
            # Preprocess sentences
            clean_sentences = []
            for sentence in sentences:
                clean_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence.lower())
                clean_sentences.append(clean_sentence)
            
            # Remove stopwords and tokenize
            stop_words = set(stopwords.words('english'))
            
            # Create word frequency table
            word_freq = {}
            for sentence in clean_sentences:
                words = word_tokenize(sentence)
                for word in words:
                    if word.lower() not in stop_words and len(word) > 1:
                        if word in word_freq:
                            word_freq[word] += 1
                        else:
                            word_freq[word] = 1
            
            # Calculate sentence scores
            sentence_scores = {}
            for i, sentence in enumerate(clean_sentences):
                words = word_tokenize(sentence)
                word_count = 0
                score = 0
                for word in words:
                    if word in word_freq and len(word) > 1:
                        score += word_freq[word]
                        word_count += 1
                
                if word_count > 0:
                    sentence_scores[i] = score / word_count
                else:
                    sentence_scores[i] = 0
            
            # Get top sentences
            ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            top_sentence_indices = [x[0] for x in ranked_sentences[:max_sentences]]
            top_sentence_indices.sort()  # Maintain original order
            
            selected_sentences = [sentences[i] for i in top_sentence_indices]
            return ' '.join(selected_sentences)
            
        except Exception as e:
            # Fallback: return first few sentences
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