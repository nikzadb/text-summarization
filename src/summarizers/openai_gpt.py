import time
import os
from typing import Dict, Any, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .traditional import BaseSummarizer


class GPT5MiniSummarizer(BaseSummarizer):
    """
    GPT-5-mini summarizer with OpenAI API integration for on-demand processing.
    
    This implementation uses the standard OpenAI API for immediate processing,
    suitable for interactive benchmarking and real-time summarization tasks.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-mini",
        max_tokens: int = 500,
        temperature: float = 0.0,
        # Standard API pricing for gpt-5-mini
        # Input: $0.25 per 1M tokens, Output: $2.00 per 1M tokens
        input_price_per_1k: float = 0.00025,   # $0.250 / 1M tokens
        output_price_per_1k: float = 0.002,   # $2.00 / 1M tokens
    ):
        super().__init__(f"GPT-5-mini-{model_name}")
        self.model_name = model_name
        
        # Load environment variables from .env file
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        
        # Initialize OpenAI client
        self.client = None
        self._setup_client()
        
    def _setup_client(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        try:
            # Initialize client using new OpenAI API pattern
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                self.client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment
            print(f"✓ OpenAI client initialized for {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using simple word-based approximation.
        More accurate methods would use tiktoken, but this provides a reasonable estimate.
        """
        # Rough approximation: 1 token ≈ 0.75 words for English text
        word_count = len(text.split())
        return int(word_count / 0.75)
    
    def _estimate_cost(self, input_text: str, output_text: str) -> float:
        """Calculate estimated cost based on token counts."""
        input_tokens = self._estimate_tokens(input_text)
        output_tokens = self._estimate_tokens(output_text)
        
        input_cost = (input_tokens / 1000.0) * self.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_price_per_1k
        
        return input_cost + output_cost
    
    def _create_prompt(self, text: str, max_sentences: int = 3) -> str:
        """Create summarization prompt for GPT-5-mini."""
        return f"""Summarize the following text in exactly {max_sentences} clear, concise sentences that capture the main points and key information:

Text: {text}

Summary:"""
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Synchronous summarization using OpenAI responses API.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        prompt = self._create_prompt(text, max_sentences)
        
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": "You are an expert at creating concise, informative summaries."},
                    {"role": "user", "content": prompt}
                ],
                text={
                    "format": {
                        "type": "text"
                    },
                    "verbosity": "low"
                },
                reasoning={
                    "effort": "low"
                },
                tools=[],
                store=True,
                include=[
                    "reasoning.encrypted_content"
                ]
            )
            
            return response.text.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to simple extractive summary
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'
    
    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Benchmark single summarization request with detailed metrics."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        start_time = time.time()
        prompt = self._create_prompt(text, max_sentences)
        
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": "You are an expert at creating concise, informative summaries."},
                    {"role": "user", "content": prompt}
                ],
                text={
                    "format": {
                        "type": "text"
                    },
                    "verbosity": "low"
                },
                reasoning={
                    "effort": "low"
                },
                tools=[],
                store=True,
                include=[
                    "reasoning.encrypted_content"
                ]
            )
            
            summary = response.text.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to simple extractive summary
            sentences = text.split('.')[:max_sentences]
            summary = '. '.join(sentences) + '.'
        
        end_time = time.time()
        
        # Calculate API cost and AWS CPU cost
        api_cost = self._estimate_cost(prompt, summary)
        processing_time_hours = (end_time - start_time) / 3600.0
        aws_cpu_cost = processing_time_hours * 0.35  # $0.35/hour for CPU instance
        total_cost = api_cost + aws_cpu_cost  # API cost + AWS hosting cost
        
        return {
            "summary": summary,
            "time_taken": end_time - start_time,
            "method": self.name,
            "cost": total_cost,
            "api_cost": api_cost,
            "aws_cost": aws_cpu_cost,
            "input_tokens": self._estimate_tokens(prompt),
            "output_tokens": self._estimate_tokens(summary),
        }
    
    def get_cost(self) -> float:
        """Return base cost (actual costs are calculated per request)."""
        return 0.0
    
    def cleanup(self):
        """Clean up OpenAI client resources"""
        try:
            if hasattr(self, 'client') and self.client is not None:
                del self.client
                self.client = None
            print(f"✓ Cleaned up {self.name} client resources")
        except Exception as e:
            print(f"Warning: Error during cleanup of {self.name}: {e}")