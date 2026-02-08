import time
import os
from typing import Dict, Any
import google.generativeai as genai
from .traditional import BaseSummarizer, TextRankSummarizer, TFIDFRankSummarizer


class GeminiSummarizer(BaseSummarizer):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash", max_output_tokens: int = 200):
        super().__init__(f"Gemini-{model_name}")
        self.model_name = model_name
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.max_output_tokens = max_output_tokens
        self._setup_model()
        
        # Pricing per 1K tokens (approximate)
        self.input_price_per_1k = 0.0001  # $0.10 per 1M tokens
        self.output_price_per_1k = 0.0004  # $0.40 per 1M tokens
    
    def _setup_model(self):
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        if not self.model:
            raise RuntimeError("Gemini model not initialized")
        
        prompt = f"""
        Summarize the following text in exactly {max_sentences} sentences. 
        Make the summary concise and capture the key points:
        
        {text}
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"max_output_tokens": self.max_output_tokens}
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'
    
    def get_cost(self) -> float:
        return 0.001
    
    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        summary = self.summarize(text, max_sentences)
        end_time = time.time()
        
        input_tokens = len(text.split()) * 1.3  # Rough approximation
        output_tokens = len(summary.split()) * 1.3
        
        cost = (input_tokens / 1000 * self.input_price_per_1k + 
                output_tokens / 1000 * self.output_price_per_1k)
        
        return {
            'summary': summary,
            'time_taken': end_time - start_time,
            'method': self.name,
            'cost': cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }


class HybridTextRankGeminiSummarizer(BaseSummarizer):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash", max_output_tokens: int = 200):
        super().__init__(f"Hybrid-TextRank-Gemini")
        self.textrank = TextRankSummarizer()
        self.gemini = GeminiSummarizer(api_key, model_name, max_output_tokens)
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        # Step 1: Use TextRank to extract 10 key sentences
        intermediate_summary = self.textrank.summarize(text, max_sentences=15)
        
        # Step 2: Apply Gemini to the TextRank output for final summarization
        final_summary = self.gemini.summarize(intermediate_summary, max_sentences)
        
        return final_summary
    
    def get_cost(self) -> float:
        # Only Gemini has cost, TextRank is free
        return self.gemini.get_cost()


class HybridTFIDFRankGeminiSummarizer(BaseSummarizer):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash", max_output_tokens: int = 200):
        super().__init__(f"Hybrid-TFIDFRank-Gemini")
        self.tfidfrank = TFIDFRankSummarizer()
        self.gemini = GeminiSummarizer(api_key, model_name, max_output_tokens)
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        # Step 1: Use TFIDFRank to extract 15 key sentences
        intermediate_summary = self.tfidfrank.summarize(text, max_sentences=15)
        
        # Step 2: Apply Gemini to the TFIDFRank output for final summarization
        final_summary = self.gemini.summarize(intermediate_summary, max_sentences)
        
        return final_summary
    
    def get_cost(self) -> float:
        # Only Gemini has cost, TFIDFRank is free
        return self.gemini.get_cost()
    
    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        
        # Step 1: TFIDFRank preprocessing
        tfidf_start = time.time()
        intermediate_summary = self.tfidfrank.summarize(text, max_sentences=15)
        tfidf_time = time.time() - tfidf_start
        
        # Step 2: Gemini final summarization
        gemini_start = time.time()
        final_summary = self.gemini.summarize(intermediate_summary, max_sentences)
        gemini_time = time.time() - gemini_start
        
        total_time = time.time() - start_time
        cost = self.get_cost()
        
        return {
            'summary': final_summary,
            'time_taken': total_time,
            'tfidfrank_time': tfidf_time,
            'gemini_time': gemini_time,
            'method': self.name,
            'cost': cost
        }