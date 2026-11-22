import os
from typing import Dict, Any, Optional, Tuple
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from .api_utils import RateLimiter, timed_call_with_retry
from .traditional import BaseSummarizer, TextRankSummarizer, TFIDFRankSummarizer

# Transient failures worth retrying: rate limits (429), overload (503), timeouts (504), 500
RETRYABLE_ERRORS = (
    google_exceptions.TooManyRequests,
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
)


def _approx_tokens_from_words(word_count: int, multiplier: float = 1.3) -> float:
    # Keep as float for consistency with existing code (approximation).
    return float(word_count) * multiplier


class GeminiSummarizer(BaseSummarizer):
    """Gemini abstractive summarizer with explicit (approximate) token + cost accounting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        max_output_tokens: int = 200,
        token_multiplier: float = 1.3,
        # Pricing per 1K tokens (USD). TODO: verify against Gemini 3.8 Flash pricing;
        # values below are carried over from Gemini 2.5 Flash public pricing:
        # input: $0.75 / 1M tokens => $0.00075 / 1K
        # output: $3.75 / 1M tokens => $0.00375 / 1K
        input_price_per_1k: float = 0.00075,
        output_price_per_1k: float = 0.00375,
        max_calls_per_second: float = 2.5,
    ):
        super().__init__(f"Gemini-{model_name}")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.max_output_tokens = max_output_tokens

        self.token_multiplier = token_multiplier
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self.rate_limiter = RateLimiter(max_calls_per_second)

        self._setup_model()

    def _setup_model(self):
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _estimate_tokens(self, text: str) -> float:
        return _approx_tokens_from_words(len(text.split()), self.token_multiplier)

    def _estimate_cost(self, input_text: str, output_text: str) -> float:
        in_tokens = self._estimate_tokens(input_text)
        out_tokens = self._estimate_tokens(output_text)
        return (in_tokens / 1000.0) * self.input_price_per_1k + (out_tokens / 1000.0) * self.output_price_per_1k

    def _generate(self, text: str, max_sentences: int = 3) -> Tuple[str, float]:
        """
        Send one request and return (summary, latency_seconds).

        Latency is measured from sending the request until the summary is received,
        excluding rate-limit waits and failed/retried attempts. Raises if the request
        ultimately fails, rather than substituting a fallback summary.
        """
        if not self.model:
            raise RuntimeError("Gemini model not initialized")

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
            
            Text:{text}
            """
        )

        def call() -> str:
            response = self.model.generate_content(
                prompt,
                generation_config={"max_output_tokens": self.max_output_tokens},
            )
            return response.text.strip()

        return timed_call_with_retry(call, self.rate_limiter, RETRYABLE_ERRORS)

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        return self._generate(text, max_sentences)[0]

    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        summary, latency = self._generate(text, max_sentences)

        input_tokens = self._estimate_tokens(text)
        output_tokens = self._estimate_tokens(summary)
        cost = (input_tokens / 1000.0) * self.input_price_per_1k + (output_tokens / 1000.0) * self.output_price_per_1k

        # Calculate AWS CPU cost in addition to API cost
        processing_time_hours = latency / 3600.0
        aws_cpu_cost = processing_time_hours * 0.238  # ml.c5.xlarge, $/hour (manuscript Table 1)
        total_cost = cost + aws_cpu_cost  # API cost + AWS hosting cost
        
        return {
            "summary": summary,
            "time_taken": latency,
            "method": self.name,
            "cost": total_cost,
            "api_cost": cost,
            "aws_cost": aws_cpu_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    
    def cleanup(self):
        """Clean up Gemini model resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            print(f"✓ Cleaned up {self.name} model resources")
        except Exception as e:
            print(f"Warning: Error during cleanup of {self.name}: {e}")
