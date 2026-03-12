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
            # Check for GPU support (CUDA or MPS)
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
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
        """Return AWS GPU instance cost per hour for transformer models"""
        return 0.95  # $0.95/hour for GPU instance
    
    def cleanup(self):
        """Clean up model resources and free GPU/CPU memory"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Move model to CPU and delete
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
                del self.model
                self.model = None
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            print(f"✓ Cleaned up {self.name} model resources")
        except Exception as e:
            print(f"Warning: Error during cleanup of {self.name}: {e}")


class DistilBARTSummarizer(OpenSourceLLMSummarizer):
    def __init__(self):
        super().__init__("sshleifer/distilbart-cnn-12-6")