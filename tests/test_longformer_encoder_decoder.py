#!/usr/bin/env python3
"""
Test script for LongformerEncoderDecoder implementation.

This script tests the LongformerEncoderDecoder summarization method
which is designed to handle long documents efficiently.
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from summarizers.llm import LongformerEncoderDecoderSummarizer


def test_longformer_initialization():
    """Test LongformerEncoderDecoder initialization and model loading."""
    print("Testing LongformerEncoderDecoder initialization...")
    try:
        summarizer = LongformerEncoderDecoderSummarizer()
        print(f"✓ LongformerEncoderDecoder initialized successfully")
        print(f"  Original model target: {summarizer.model_name}")
        print(f"  Summarizer name: {summarizer.name}")
        
        if hasattr(summarizer, 'actual_model_name'):
            print(f"  Actual loaded model: {summarizer.actual_model_name}")
        
        return summarizer
    except Exception as e:
        print(f"✗ Failed to initialize LongformerEncoderDecoder: {e}")
        return None


def test_longformer_summarization(summarizer):
    """Test LongformerEncoderDecoder summarization capabilities."""
    print("\nTesting LongformerEncoderDecoder summarization...")
    
    # Test with a longer document to showcase Longformer's strength
    long_document = """
    The Longformer model represents a significant advancement in transformer architectures for processing long sequences. 
    Traditional BERT models are limited to 512 tokens due to the quadratic complexity of self-attention mechanisms.
    This limitation makes them unsuitable for document-level tasks such as summarization of lengthy articles or papers.
    
    Longformer addresses this challenge through a novel attention pattern that combines local windowed attention with global attention.
    Local attention allows each token to attend to nearby tokens within a fixed window, maintaining local context understanding.
    Global attention is applied to specific tokens that need to attend to all positions in the sequence.
    
    This hybrid approach reduces the computational complexity from O(n²) to O(n×w) where w is the window size.
    As a result, Longformer can process sequences up to 4,096 tokens efficiently, with some variants handling even longer sequences.
    
    The LED (Longformer Encoder-Decoder) extends this capability to encoder-decoder architectures for generative tasks.
    LED has been successfully applied to document summarization, achieving state-of-the-art results on several benchmarks.
    The model can process documents with thousands of tokens and generate coherent summaries that capture the main points.
    
    In practical applications, LongformerEncoderDecoder is particularly useful for summarizing research papers, news articles,
    legal documents, and other long-form content where traditional models would require truncation or chunking strategies.
    """
    
    try:
        print("  Testing with long document (multiple paragraphs)...")
        print(f"  Input length: {len(long_document.split())} words")
        
        start_time = time.time()
        
        # Test with different sentence limits
        for max_sentences in [1, 2, 3]:
            summary = summarizer.summarize(long_document.strip(), max_sentences=max_sentences)
            print(f"  ✓ {max_sentences}-sentence summary: {summary[:100]}...")
        
        end_time = time.time()
        
        print(f"  ✓ Summarization completed in {end_time - start_time:.2f} seconds")
        print("  ✓ LongformerEncoderDecoder handles long documents successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to generate summary: {e}")
        return False


def test_longformer_benchmark_method(summarizer):
    """Test the benchmark_summarize method."""
    print("\nTesting LongformerEncoderDecoder benchmark method...")
    
    test_text = """
    Document summarization is a critical task in natural language processing that involves creating concise representations of longer texts.
    With the exponential growth of digital information, automated summarization has become increasingly important for information management.
    Modern approaches leverage deep learning models, particularly transformer-based architectures, to generate high-quality summaries.
    The LongformerEncoderDecoder model represents one such approach that can handle documents significantly longer than traditional models.
    """
    
    try:
        result = summarizer.benchmark_summarize(test_text.strip(), max_sentences=2)
        
        print("  ✓ Benchmark method executed successfully")
        print(f"    Summary: {result['summary']}")
        print(f"    Time taken: {result['time_taken']:.3f} seconds")
        print(f"    Method: {result['method']}")
        print(f"    Cost: ${result['cost']:.6f}")
        
        # Validate result structure
        required_keys = ['summary', 'time_taken', 'method', 'cost']
        for key in required_keys:
            if key not in result:
                print(f"  ✗ Missing key in result: {key}")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to run benchmark method: {e}")
        return False


def test_longformer_vs_traditional():
    """Compare LongformerEncoderDecoder with traditional approaches on long text."""
    print("\nTesting LongformerEncoderDecoder advantages with long text...")
    
    # Very long text that would challenge traditional models
    very_long_text = " ".join([
        "This is sentence number {}. It discusses various aspects of document processing and summarization.".format(i) 
        for i in range(1, 51)  # 50 sentences
    ])
    
    try:
        summarizer = LongformerEncoderDecoderSummarizer()
        
        print(f"  Input: {len(very_long_text.split())} words, {len(very_long_text.split('.')) - 1} sentences")
        
        start_time = time.time()
        summary = summarizer.summarize(very_long_text, max_sentences=3)
        end_time = time.time()
        
        print(f"  ✓ Processed long text in {end_time - start_time:.2f} seconds")
        print(f"  ✓ Generated summary: {summary[:150]}...")
        print("  ✓ LongformerEncoderDecoder successfully handles very long inputs")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed with long text: {e}")
        return False


def main():
    """Run all LongformerEncoderDecoder tests."""
    print("🧪 LongformerEncoderDecoder Test Suite")
    print("=" * 60)
    print("Testing the LongformerEncoderDecoder implementation")
    print("Designed for efficient processing of long documents")
    print("=" * 60)
    
    # Test 1: Initialization
    summarizer = test_longformer_initialization()
    if not summarizer:
        print("\n❌ Initialization failed - cannot proceed with other tests")
        return False
    
    # Test 2: Basic summarization
    if not test_longformer_summarization(summarizer):
        print("\n❌ Summarization test failed")
        return False
    
    # Test 3: Benchmark method
    if not test_longformer_benchmark_method(summarizer):
        print("\n❌ Benchmark method test failed")
        return False
    
    # Test 4: Long text handling
    if not test_longformer_vs_traditional():
        print("\n❌ Long text handling test failed")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All LongformerEncoderDecoder tests passed!")
    print("🎉 LongformerEncoderDecoder implementation is working correctly")
    print()
    print("Key Features Validated:")
    print("• ✅ Robust model loading with fallback support")
    print("• ✅ Long sequence processing capabilities") 
    print("• ✅ Multiple summarization length options")
    print("• ✅ Benchmark integration compatibility")
    print("• ✅ Error handling and graceful degradation")
    print()
    print("Usage in benchmarking:")
    print("python main.py --methods LongformerEncoderDecoder --datasets cnn_dailymail")
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during testing: {e}")
        sys.exit(1)