import os
from typing import Dict, Any, Optional, Tuple

import openai
from dotenv import load_dotenv

from .api_utils import RateLimiter, timed_call_with_retry
from .traditional import BaseSummarizer

# Transient failures worth retrying: rate limits, timeouts, connection drops, 5xx
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class GPT54MiniSummarizer(BaseSummarizer):
    """
    GPT-5.4-mini summarizer (reasoning disabled) with OpenAI API integration for on-demand processing.
    
    This implementation uses the standard OpenAI API for immediate processing,
    suitable for interactive benchmarking and real-time summarization tasks.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "gpt-5.4-mini",
        max_tokens: int = 500,
        temperature: float = 0.0,
        # Standard API pricing for gpt-5.4-mini
        # Input: $0.75 per 1M tokens, Output: $4.50 per 1M tokens
        input_price_per_1k: float = 0.00075,   # $0.75 / 1M tokens
        output_price_per_1k: float = 0.0045,   # $4.50 / 1M tokens
        max_calls_per_second: float = 2.5,
    ):
        super().__init__(f"GPT-5.4-mini-{model_name}")
        self.model_name = model_name
        
        # Load environment variables from .env file
        load_dotenv()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self.rate_limiter = RateLimiter(max_calls_per_second)
        
        # Initialize OpenAI client
        self.client = None
        self._setup_client()
        
    def _setup_client(self):
        """Initialize OpenAI client."""
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
        """Create summarization prompt for GPT-5.4-mini."""

        prompt = (
            f"""Summarize the text below in exactly {max_sentences} sentences\n
            Rules:\n
            - Only use information from the text.\n
            - No interpretation or external knowledge.\n
            - No bullet points.\n
            - No introductory phrases.\n
            - No concluding phrases.\n
            - No meta commentary.\n
            - Exactly {max_sentences} sentences.\n            
            Text:{text}\n

            Summary:
            """
        )

        return prompt
    
    def _generate(self, text: str, max_sentences: int = 3) -> Tuple[str, str, float]:
        """
        Send one request and return (prompt, summary, latency_seconds).

        Latency is measured from sending the request until the summary is received,
        excluding rate-limit waits and failed/retried attempts. Raises if the request
        ultimately fails, rather than substituting a fallback summary.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        prompt = self._create_prompt(text, max_sentences)
        
        def call() -> str:
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
                # Disable thinking entirely: no reasoning tokens are generated or billed
                reasoning={
                    "effort": "none"
                },
                max_output_tokens=self.max_tokens,
                tools=[],
                store=False,
            )
            return response.output_text.strip()
        
        summary, latency = timed_call_with_retry(
            call, self.rate_limiter, lambda e: isinstance(e, RETRYABLE_ERRORS)
        )
        return prompt, summary, latency
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Synchronous summarization using OpenAI responses API."""
        return self._generate(text, max_sentences)[1]
    
    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Benchmark single summarization request with detailed metrics."""
        prompt, summary, latency = self._generate(text, max_sentences)
        
        # Calculate API cost and AWS CPU cost
        api_cost = self._estimate_cost(prompt, summary)
        processing_time_hours = latency / 3600.0
        aws_cpu_cost = processing_time_hours * 0.238  # ml.c5.xlarge, $/hour (manuscript Table 1)
        total_cost = api_cost + aws_cpu_cost  # API cost + AWS hosting cost
        
        return {
            "summary": summary,
            "time_taken": latency,
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