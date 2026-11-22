"""PEGASUS and LongT5 summarizers (reviewer-requested baselines).

Both follow the same protocol as BART/DistilBART: one publicly released
fine-tuned checkpoint is used unchanged across all four datasets (no
per-dataset checkpoint selection, no fine-tuning by us), the same
sentence-count-derived output length bounds, deterministic decoding with
num_beams=4, and fp32 on the same GPU tier. Checkpoint revisions are pinned
so the reported results map to exact weights.
"""
from typing import Optional

import torch

from .llm import BARTSummarizer


class PegasusSummarizer(BARTSummarizer):
    """PEGASUS-Large fine-tuned on CNN/DailyMail (568M params, 1,024-token input).

    Mirrors bart-large-cnn / distilbart-cnn-12-6: a CNN/DailyMail fine-tune
    applied to every regime, rather than the per-dataset PEGASUS checkpoints,
    so no architecture gets in-domain fine-tuning on a regime the others don't.
    """
    MODEL_NAME = "google/pegasus-cnn_dailymail"
    REVISION = "40d588fdab0cc077b80d950b300bf66ad3c75b92"

    # Checkpoint default is num_beams=8; pinned to BART's 4 for equal settings.
    # Other generation settings (length_penalty=0.8) are the checkpoint defaults.
    generate_kwargs = {'num_beams': 4}

    def __init__(self, torch_dtype: Optional[torch.dtype] = None):
        super().__init__(self.MODEL_NAME, revision=self.REVISION,
                         max_input_tokens=1024, torch_dtype=torch_dtype)

    def _postprocess(self, summary: str) -> str:
        # PEGASUS CNN/DM was trained with "<n>" as its sentence separator
        return " ".join(summary.replace("<n>", " ").split())


class LongT5Summarizer(BARTSummarizer):
    """LongT5-TGlobal-Base fine-tuned for long-document summarization on BookSum.

    Google released LongT5 pretrained-only; this is the most widely used public
    summarization fine-tune of the same base checkpoint. BookSum is disjoint from
    all four evaluated datasets, so no regime gets in-domain fine-tuning.
    Input is truncated at 4,096 tokens (4x the BART/PEGASUS limit) to keep
    inference tractable on the 16 GB T4.
    """
    MODEL_NAME = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    REVISION = "a40cddfd840d398653f37d98c384bfe56d677125"
    MAX_INPUT_TOKENS = 4096

    # Checkpoint default is num_beams=2; pinned to 4 for equal settings. Its
    # other defaults (no_repeat_ngram_size=3, repetition_penalty=3.5,
    # length_penalty=0.8) are kept as published, as for the other checkpoints.
    generate_kwargs = {'num_beams': 4}

    def __init__(self, max_input_tokens: int = MAX_INPUT_TOKENS,
                 torch_dtype: Optional[torch.dtype] = None):
        if torch_dtype == torch.float16:
            # T5-family weights overflow in fp16; bf16 is unsupported on T4
            raise ValueError("LongT5 must run in fp32 (fp16 produces NaNs for T5 models)")
        super().__init__(self.MODEL_NAME, revision=self.REVISION,
                         max_input_tokens=max_input_tokens, torch_dtype=torch_dtype)
