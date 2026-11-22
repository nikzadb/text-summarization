"""Chunk-and-aggregate long-document strategy (manuscript Section 4.4.1).

Splits the source into non-overlapping token chunks, summarizes each chunk
with the wrapped model, then summarizes the concatenated chunk summaries
down to the target sentence count. Documents that fit in one chunk are
summarized directly, so they get exactly the leading-truncation output.
"""
from typing import List

from .traditional import BaseSummarizer
from .llm import BARTSummarizer


class ChunkedSummarizer(BaseSummarizer):
    def __init__(self, base: BARTSummarizer, chunk_tokens: int = 1000,
                 sentences_per_chunk: int = 2):
        super().__init__(f"{base.name}-chunked")
        self.base = base
        self.chunk_tokens = chunk_tokens
        self.sentences_per_chunk = sentences_per_chunk

    def _chunk(self, text: str) -> List[str]:
        tokenizer = self.base.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.decode(ids[i:i + self.chunk_tokens], skip_special_tokens=True)
                for i in range(0, len(ids), self.chunk_tokens)]

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        chunks = self._chunk(text)
        # Aggregate recursively until the concatenated chunk summaries fit in
        # one chunk (only the longest arXiv documents need a second level).
        while len(chunks) > 1:
            partials = [self.base.summarize(c, self.sentences_per_chunk) for c in chunks]
            chunks = self._chunk(" ".join(partials))
        return self.base.summarize(chunks[0] if chunks else text, max_sentences)

    def get_cost(self) -> float:
        return self.base.get_cost()

    def cleanup(self):
        self.base.cleanup()
