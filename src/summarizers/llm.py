import time
import torch
from typing import Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from .traditional import BaseSummarizer


class OpenSourceLLMSummarizer(BaseSummarizer):
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        super().__init__(f"OpenSource-{model_name.split('/')[-1]}")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        try:
            # Check for MPS (Apple Silicon GPU) support
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using device: {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model = self.model.to(device)
            
            self.pipeline = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=device,
                framework="pt"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        
        max_length = max_sentences * 30
        min_length = max_sentences * 10
        
        try:
            result = self.pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']
        except Exception as e:
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'
    
    def get_cost(self) -> float:
        return 0.0


class T5Summarizer(OpenSourceLLMSummarizer):
    def __init__(self):
        self.model_name = "t5-small"
        BaseSummarizer.__init__(self, "T5-small")
        self._load_model()
    
    def _load_model(self):
        try:
            # Check for MPS (Apple Silicon GPU) support
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using device for T5: {device}")
            
            self.pipeline = pipeline(
                "summarization", 
                model=self.model_name,
                device=device,
                framework="pt"
            )
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            raise
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        
        prefixed_text = f"summarize: {text}"
        max_length = max_sentences * 30
        min_length = max_sentences * 10
        
        try:
            result = self.pipeline(
                prefixed_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']
        except Exception as e:
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'


class DistilBARTSummarizer(OpenSourceLLMSummarizer):
    def __init__(self):
        super().__init__("sshleifer/distilbart-cnn-12-6")