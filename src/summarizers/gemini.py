import time
import os
from typing import Dict, Any, Optional
import google.generativeai as genai

from .traditional import BaseSummarizer, TextRankSummarizer, TFIDFRankSummarizer


def _approx_tokens_from_words(word_count: int, multiplier: float = 1.3) -> float:
    # Keep as float for consistency with existing code (approximation).
    return float(word_count) * multiplier


class GeminiSummarizer(BaseSummarizer):
    """Gemini abstractive summarizer with explicit (approximate) token + cost accounting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_output_tokens: int = 200,
        token_multiplier: float = 1.3,
        # Pricing per 1K tokens (USD). Default reflects Gemini 2.5 Flash public pricing:
        # input: $0.30 / 1M tokens => $0.0003 / 1K
        # output: $2.50 / 1M tokens => $0.0025 / 1K
        input_price_per_1k: float = 0.0003,
        output_price_per_1k: float = 0.0025,
    ):
        super().__init__(f"Gemini-{model_name}")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.max_output_tokens = max_output_tokens

        self.token_multiplier = token_multiplier
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k

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

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        if not self.model:
            raise RuntimeError("Gemini model not initialized")

        prompt = (
            f"Summarize the following text in exactly {max_sentences} sentences.\n"
            f"Make the summary concise and capture the key points:\n\n"
            f"{text}"
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"max_output_tokens": self.max_output_tokens},
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'

    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        summary = self.summarize(text, max_sentences)
        end_time = time.time()

        input_tokens = self._estimate_tokens(text)
        output_tokens = self._estimate_tokens(summary)
        cost = (input_tokens / 1000.0) * self.input_price_per_1k + (output_tokens / 1000.0) * self.output_price_per_1k

        return {
            "summary": summary,
            "time_taken": end_time - start_time,
            "method": self.name,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class _HybridBase(BaseSummarizer):
    """Shared logic for Extract-then-Abstract (E2A) pipelines."""

    def __init__(self, name: str, gemini: GeminiSummarizer, extract_k_map: Optional[dict] = None):
        super().__init__(name)
        self.gemini = gemini
        self.extract_k_map = extract_k_map or {"cnn_dailymail": 15, "arxiv": 50}
        self.extract_k = self.extract_k_map.get("cnn_dailymail", 15)

    def set_dataset(self, dataset_name: str):
        # Called by BenchmarkFramework to ensure consistent regime configuration.
        self.extract_k = self.extract_k_map.get(dataset_name, self.extract_k)

    def _bench(self, intermediate_summary: str, max_sentences: int) -> Dict[str, Any]:
        gemini_start = time.time()
        final_summary = self.gemini.summarize(intermediate_summary, max_sentences)
        gemini_time = time.time() - gemini_start

        # Cost is based on Gemini call only (approx tokens on intermediate input + final output)
        input_tokens = self.gemini._estimate_tokens(intermediate_summary)
        output_tokens = self.gemini._estimate_tokens(final_summary)
        cost = (input_tokens / 1000.0) * self.gemini.input_price_per_1k + (output_tokens / 1000.0) * self.gemini.output_price_per_1k

        return {
            "final_summary": final_summary,
            "gemini_time": gemini_time,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class HybridTextRankGeminiSummarizer(_HybridBase):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash", max_output_tokens: int = 200):
        gemini = GeminiSummarizer(api_key=api_key, model_name=model_name, max_output_tokens=max_output_tokens)
        super().__init__("Hybrid-TextRank-Gemini", gemini=gemini)
        self.textrank = TextRankSummarizer()

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        intermediate_summary = self.textrank.summarize(text, max_sentences=self.extract_k)
        return self.gemini.summarize(intermediate_summary, max_sentences)

    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        start_time = time.time()

        tr_start = time.time()
        intermediate_summary = self.textrank.summarize(text, max_sentences=self.extract_k)
        textrank_time = time.time() - tr_start

        llm = self._bench(intermediate_summary, max_sentences)

        total_time = time.time() - start_time
        return {
            "summary": llm["final_summary"],
            "time_taken": total_time,
            "textrank_time": textrank_time,
            "gemini_time": llm["gemini_time"],
            "method": self.name,
            "cost": llm["cost"],
            "input_tokens": llm["input_tokens"],
            "output_tokens": llm["output_tokens"],
            "hybrid_extract_sentences": self.extract_k,
        }


class HybridTFIDFRankGeminiSummarizer(_HybridBase):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash", max_output_tokens: int = 200):
        gemini = GeminiSummarizer(api_key=api_key, model_name=model_name, max_output_tokens=max_output_tokens)
        super().__init__("Hybrid-TFIDFRank-Gemini", gemini=gemini)
        self.tfidfrank = TFIDFRankSummarizer()

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        intermediate_summary = self.tfidfrank.summarize(text, max_sentences=self.extract_k)
        return self.gemini.summarize(intermediate_summary, max_sentences)

    def benchmark_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        start_time = time.time()

        tfidf_start = time.time()
        intermediate_summary = self.tfidfrank.summarize(text, max_sentences=self.extract_k)
        tfidfrank_time = time.time() - tfidf_start

        llm = self._bench(intermediate_summary, max_sentences)

        total_time = time.time() - start_time
        return {
            "summary": llm["final_summary"],
            "time_taken": total_time,
            "tfidfrank_time": tfidfrank_time,
            "gemini_time": llm["gemini_time"],
            "method": self.name,
            "cost": llm["cost"],
            "input_tokens": llm["input_tokens"],
            "output_tokens": llm["output_tokens"],
            "hybrid_extract_sentences": self.extract_k,
        }
