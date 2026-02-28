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
        return 0.0


class T5Summarizer(OpenSourceLLMSummarizer):
    def __init__(self):
        self.model_name = "t5-small"
        BaseSummarizer.__init__(self, "T5-small")
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


class PegasusXSummarizer(OpenSourceLLMSummarizer):
    def __init__(self):
        super().__init__("google/pegasus-x-large")


class LongformerEncoderDecoderSummarizer(BaseSummarizer):
    def __init__(self, prefer_domain=None):
        super().__init__("LongformerEncoderDecoder")
        self.model_name = "allenai/led-large-16384-arxiv"  # Default: arXiv-trained (more general)
        self.prefer_domain = prefer_domain  # Can be 'news', 'scientific', 'books', etc.
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def set_dataset(self, dataset_name: str):
        """Set preferred model based on dataset domain."""
        self.prefer_domain = self._infer_domain(dataset_name)
    
    def _infer_domain(self, dataset_name: str) -> str:
        """Infer domain from dataset name for model selection."""
        dataset_name = dataset_name.lower()
        
        if any(term in dataset_name for term in ['cnn', 'dailymail', 'news', 'media']):
            return 'news'
        elif any(term in dataset_name for term in ['arxiv', 'scientific', 'paper']):
            return 'scientific' 
        elif any(term in dataset_name for term in ['book', 'literature']):
            return 'books'
        elif any(term in dataset_name for term in ['government', 'gov', 'legal']):
            return 'government'
        else:
            return 'general'
    
    def _load_model(self):
        try:
            # Check for GPU support (CUDA or MPS)
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"Using device for LongformerEncoderDecoder: {device}")
            
            # Try multiple Longformer model approaches
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Select models based on domain preference
            alternative_models = self._get_domain_preferred_models()
            
            # Add general fallbacks
            alternative_models.extend([
                "facebook/bart-large-cnn",            # BART for news
                "t5-base"                             # T5 base model
            ])
            
            model_loaded = False
            for alt_model in alternative_models:
                try:
                    print(f"Trying model: {alt_model}")
                    self.tokenizer = AutoTokenizer.from_pretrained(alt_model)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(alt_model)
                    self.model = self.model.to(device)
                    self.device = device
                    self.actual_model_name = alt_model
                    model_loaded = True
                    print(f"Successfully loaded: {alt_model}")
                    break
                except Exception as model_error:
                    print(f"Failed to load {alt_model}: {model_error}")
                    continue
            
            if not model_loaded:
                raise Exception("Could not load any Longformer-based model")
                
        except Exception as e:
            print(f"Error loading LongformerEncoderDecoder model: {e}")
            print("Falling back to BART model...")
            # Fallback to a working model
            try:
                from transformers import pipeline
                self.pipeline = pipeline("summarization", model="facebook/bart-base")
                self.device = "cpu"
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise
    
    def _get_domain_preferred_models(self):
        """Get prioritized model list based on domain preference."""
        
        # Domain-specific model preferences
        domain_models = {
            'news': [
                "patrickvonplaten/longformer2roberta-cnn_dailymail-fp16",  # CNN/DailyMail optimized
                "allenai/led-large-16384-arxiv",      # Good general model
                "allenai/led-base-16384",             # General LED base
            ],
            'scientific': [
                "allenai/led-large-16384-arxiv",      # Trained on arXiv papers
                "allenai/led-base-16384-arxiv",       # Smaller arXiv model
                "allenai/led-large-16384",            # General LED large
                "allenai/led-base-16384",             # General LED base
            ],
            'books': [
                "pszemraj/led-large-book-summary",    # Book summarization
                "pszemraj/led-base-book-summary",     # Smaller book model
                "allenai/led-large-16384",            # General LED large
                "allenai/led-base-16384",             # General LED base
            ],
            'government': [
                "allenai/led-large-16384",            # General LED large
                "allenai/led-base-16384",             # General LED base
                "allenai/led-large-16384-arxiv",      # Scientific (often formal)
            ],
            'general': [
                "allenai/led-large-16384-arxiv",      # Good general model (arXiv diverse)
                "allenai/led-large-16384",            # General LED large
                "allenai/led-base-16384-arxiv",       # Smaller arXiv model
                "allenai/led-base-16384",             # General LED base
                "pszemraj/led-base-book-summary",     # Books (diverse content)
                "yhavinga/longformer-base-4096",      # General longformer
            ]
        }
        
        # Get domain-specific models or default to general
        domain = self.prefer_domain or 'general'
        # preferred_models = domain_models.get(domain, domain_models['general'])
        preferred_models = "allenai/led-large-16384"
        
        print(f"Using model preference {preferred_models} for domain: {domain}")
        return preferred_models
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        if hasattr(self, 'pipeline'):
            # Use fallback pipeline
            max_length = max_sentences * 30
            min_length = max_sentences * 10
            try:
                result = self.pipeline(text, max_length=max_length, min_length=min_length, truncation=True)
                return result[0]['summary_text']
            except Exception as e:
                sentences = text.split('.')[:max_sentences]
                return '. '.join(sentences) + '.'
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        # Longformer can handle much longer sequences (up to 4096 tokens)
        max_length = max_sentences * 40
        min_length = max_sentences * 15
        
        try:
            # Determine max length based on model
            if hasattr(self, 'actual_model_name') and 'led' in self.actual_model_name.lower():
                max_input_length = 16384  # LED can handle much longer sequences
            else:
                max_input_length = 4096   # Standard Longformer length
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                max_length=max_input_length,
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Error during summarization: {e}")
            # Fallback to simple extractive summary
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'
    
    def get_cost(self) -> float:
        return 0.0