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
            
            # Try different pipeline tasks based on model type
            try:
                self.pipeline = pipeline(
                    "text2text-generation", 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    device=device,
                    framework="pt"
                )
            except Exception:
                # Fallback to summarization for older transformers or BART models
                try:
                    self.pipeline = pipeline(
                        "summarization", 
                        model=self.model, 
                        tokenizer=self.tokenizer,
                        device=device,
                        framework="pt"
                    )
                except Exception:
                    # Use manual generation
                    self.pipeline = None
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
            # Handle different output formats
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text']
                elif 'summary_text' in result[0]:
                    return result[0]['summary_text']
                else:
                    # Take the first value if structure is different
                    return str(list(result[0].values())[0])
            return str(result)
        except Exception as e:
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'
    
    def get_cost(self) -> float:
        return 0.0


class RetrievalAugmentedSummarizer(BaseSummarizer):
    """
    Retrieval-Augmented Summarizer implementing a two-stage context selection strategy.
    
    This method implements the approach described in the specification:
    1. Document segmentation: Split documents into fixed-length chunks
    2. Semantic embedding: Use sentence-transformer to embed chunks
    3. Retrieval: Given a generic query, retrieve top-k most similar chunks
    4. Generation: Concatenate retrieved chunks and summarize with LED model
    
    This approach isolates semantic context selection from generation and
    avoids dataset-specific tuning.
    """
    
    def __init__(self, chunk_size: int = 512, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__("Retrieval-Augmented-Summarizer")
        self.chunk_size = chunk_size  # Fixed-length chunks in characters
        self.embedding_model_name = embedding_model
        self.generation_model_name = "allenai/led-large-16384"  # General LED checkpoint
        
        # Model components
        self.embedding_model = None
        self.tokenizer = None
        self.generation_model = None
        self.pipeline = None
        self.device = None
        
        # Generic summarization query for retrieval
        self.generic_query = "Summarize the key points and main ideas from this document."
        
        self._load_models()
    
    def _load_models(self):
        """Load both the sentence-transformer and LED generation models."""
        try:
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"  
            else:
                self.device = "cpu"
            
            print(f"Loading Retrieval-Augmented-Summarizer components on device: {self.device}")
            
            # Load sentence-transformer for embeddings
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
                print(f"✓ Loaded embedding model: {self.embedding_model_name}")
            except Exception as embed_error:
                print(f"⚠️  Failed to load embedding model: {embed_error}")
                self.embedding_model = None
            
            # Load LED model for generation
            try:
                from transformers import pipeline
                # Try different pipeline tasks for LED model
                pipeline_tasks = ["text2text-generation", "text-generation"]
                
                pipeline_loaded = False
                for task in pipeline_tasks:
                    try:
                        self.pipeline = pipeline(
                            task,
                            model=self.generation_model_name,
                            device=self.device if self.device != "mps" else "cpu",
                            framework="pt"
                        )
                        print(f"✓ Loaded generation model with {task}: {self.generation_model_name}")
                        pipeline_loaded = True
                        break
                    except Exception as task_error:
                        print(f"⚠️  Failed to load with {task}: {task_error}")
                        continue
                
                if not pipeline_loaded:
                    raise Exception("All pipeline tasks failed")
            except Exception as gen_error:
                print(f"⚠️  Failed to load generation model as pipeline: {gen_error}")
                # Try manual loading
                try:
                    from transformers import LEDTokenizer, LEDForConditionalGeneration
                    self.tokenizer = LEDTokenizer.from_pretrained(self.generation_model_name)
                    self.generation_model = LEDForConditionalGeneration.from_pretrained(self.generation_model_name)
                    self.generation_model = self.generation_model.to(self.device)
                    print(f"✓ Loaded generation model manually: {self.generation_model_name}")
                except Exception as manual_error:
                    print(f"⚠️  Manual loading also failed: {manual_error}")
                    self.tokenizer = None
                    self.generation_model = None
                
        except Exception as e:
            print(f"Error loading Retrieval-Augmented-Summarizer: {e}")
            self.embedding_model = None
            self.pipeline = None
            self.tokenizer = None
            self.generation_model = None
    
    def _segment_into_chunks(self, text: str) -> list[str]:
        """
        Segment document into fixed-length chunks.
        
        Args:
            text: Input document text
            
        Returns:
            List of text chunks
        """
        chunks = []
        text = text.strip()
        
        # Simple chunking by character count with word boundaries
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Find word boundary to avoid cutting words
            while end > start and text[end] not in [' ', '.', '!', '?', '\n', '\t']:
                end -= 1
            
            if end == start:  # Safety: if no word boundary found, use original end
                end = start + self.chunk_size
            
            chunks.append(text[start:end].strip())
            start = end
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        return chunks
    
    def _embed_texts(self, texts: list[str]) -> 'numpy.ndarray':
        """
        Generate embeddings for text chunks using sentence-transformer.
        
        Args:
            texts: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        if not self.embedding_model:
            # Fallback: random embeddings (for testing when model fails to load)
            import numpy as np
            return np.random.rand(len(texts), 384)  # MiniLM has 384 dimensions
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Embedding failed: {e}")
            # Fallback: random embeddings
            import numpy as np
            return np.random.rand(len(texts), 384)
    
    def _retrieve_top_k_chunks(self, chunks: list[str], query: str, k: int) -> list[str]:
        """
        Retrieve top-k most semantically similar chunks to the query.
        
        Args:
            chunks: List of text chunks
            query: Retrieval query (generic summarization query)
            k: Number of chunks to retrieve
            
        Returns:
            List of top-k most similar chunks
        """
        if not chunks:
            return []
        
        k = min(k, len(chunks))  # Don't retrieve more than available
        
        try:
            # Embed all chunks and the query
            chunk_embeddings = self._embed_texts(chunks)
            query_embedding = self._embed_texts([query])
            
            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
            
            # Get top-k most similar chunks
            top_k_indices = similarities.argsort()[-k:][::-1]  # Descending order
            
            # Return chunks in original document order (not similarity order)
            selected_chunks = [chunks[i] for i in sorted(top_k_indices)]
            
            return selected_chunks
            
        except Exception as e:
            print(f"Retrieval failed: {e}")
            # Fallback: return first k chunks
            return chunks[:k]
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Two-stage retrieval-augmented summarization.
        
        Args:
            text: Input document text
            max_sentences: Maximum sentences in final summary
            
        Returns:
            Generated summary
        """
        try:
            # Stage 1: Document segmentation and retrieval
            chunks = self._segment_into_chunks(text)
            
            if not chunks:
                return "Unable to process document."
            
            # Determine how many chunks to retrieve based on final summary length
            # Retrieve more chunks for longer summaries, but cap to avoid too much context
            retrieval_ratio = max(3, max_sentences)  # At least 3 chunks, more for longer summaries
            k = min(len(chunks), retrieval_ratio)
            
            # Retrieve top-k semantically relevant chunks using generic query
            selected_chunks = self._retrieve_top_k_chunks(chunks, self.generic_query, k)
            
            # Concatenate retrieved chunks
            retrieved_content = ' '.join(selected_chunks)
            
            # Stage 2: Generation using LED model
            if self.pipeline:
                # Use pipeline for generation
                max_length = max_sentences * 30
                min_length = max_sentences * 10
                
                # For LED models, we might need to add a summarization prompt
                if 'led' in self.generation_model_name.lower():
                    prompt = f"Summarize this text: {retrieved_content}"
                else:
                    prompt = retrieved_content
                
                result = self.pipeline(
                    prompt,
                    max_length=max_length,
                    min_length=min_length,
                    truncation=True,
                    do_sample=False
                )
                # Handle different output formats
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        # For text-generation pipeline, remove the input prompt
                        generated = result[0]['generated_text']
                        if generated.startswith(prompt):
                            generated = generated[len(prompt):].strip()
                        return generated
                    elif 'summary_text' in result[0]:
                        return result[0]['summary_text']
                    else:
                        # Take the first value if structure is different
                        return str(list(result[0].values())[0])
                return str(result)
                
            elif self.generation_model and self.tokenizer:
                # Use model directly
                max_length = max_sentences * 30
                min_length = max_sentences * 10
                
                # LED can handle long sequences (up to 16384 tokens)
                inputs = self.tokenizer(
                    retrieved_content,
                    max_length=16384,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # Generate summary
                with torch.no_grad():
                    summary_ids = self.generation_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                
                # Decode summary
                summary = self.tokenizer.decode(
                    summary_ids[0], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                return summary
            else:
                # Fallback: return concatenated retrieved chunks (extractive summary)
                sentences = retrieved_content.split('.')[:max_sentences]
                return '. '.join(sentences) + '.'
                
        except Exception as e:
            print(f"Retrieval-Augmented-Summarizer failed: {e}")
            # Ultimate fallback: simple extraction from original text
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'
    
    def get_cost(self) -> float:
        """Retrieval-augmented method has minimal cost (open-source models)."""
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
            
            # Try new transformers API first (text2text-generation for T5)
            try:
                self.pipeline = pipeline(
                    "text2text-generation", 
                    model=self.model_name,
                    device=device,
                    framework="pt"
                )
                print(f"✓ Loaded T5 with text2text-generation pipeline")
            except Exception:
                # Fallback to manual loading for T5
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.model = self.model.to(device)
                self.device = device
                self.pipeline = None  # Use manual generation
                print(f"✓ Loaded T5 manually")
                
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            raise
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        prefixed_text = f"summarize: {text}"
        max_length = max_sentences * 30
        min_length = max_sentences * 10
        
        try:
            if self.pipeline:
                # Use pipeline (text2text-generation)
                result = self.pipeline(
                    prefixed_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                # Handle different output formats
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        return result[0]['generated_text']
                    elif 'summary_text' in result[0]:
                        return result[0]['summary_text']
                    else:
                        # Take the first value if structure is different
                        return str(list(result[0].values())[0])
                return str(result)
                
            elif hasattr(self, 'model') and hasattr(self, 'tokenizer'):
                # Use manual generation
                inputs = self.tokenizer(
                    prefixed_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False
                    )
                
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return summary
            else:
                raise RuntimeError("Neither pipeline nor manual model available")
                
        except Exception as e:
            print(f"T5 summarization failed: {e}")
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
            print("Falling back to manual model loading...")
            # Fallback to manual model loading (no pipeline)
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
                self.model = self.model.to("cpu")
                self.device = "cpu"
                self.actual_model_name = "facebook/bart-base"
                print("✓ Fallback to BART manual loading successful")
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

        # Use general checkpoint regardless of the dataset domain type
        # preferred_models = domain_models.get(domain, domain_models['general'])
        preferred_models = ["allenai/led-large-16384"]
        
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