import time
import os
from typing import Dict, Any
import google.generativeai as genai
from .traditional import BaseSummarizer


class GeminiSummarizer(BaseSummarizer):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash", max_output_tokens: int = 200):
        super().__init__(f"Gemini-{model_name}")
        self.model_name = model_name
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self._setup_model()
        
        # Pricing per 1K tokens (approximate)
        self.input_price_per_1k = 0.00025  # $0.25 per 1M tokens
        self.output_price_per_1k = 0.00075  # $0.75 per 1M tokens
    
    def _setup_model(self):
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name, max_output_tokens=200)
    
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
                generation_config={"max_output_tokens": 8192}
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