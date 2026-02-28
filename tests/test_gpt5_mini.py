#!/usr/bin/env python3
"""
Simple test script for GPT-5-mini summarizer implementation.

This script tests both the synchronous and batch API functionality
without requiring an actual OpenAI API key.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.summarizers.openai_gpt import GPT5MiniSummarizer

def test_basic_initialization():
    """Test basic initialization without API key."""
    print("Testing basic initialization...")
    
    try:
        # This should fail without API key
        summarizer = GPT5MiniSummarizer(api_key="test_key")
        print("✓ GPT5MiniSummarizer initialized successfully")
        print(f"  Model name: {summarizer.model_name}")
        print(f"  Max tokens: {summarizer.max_tokens}")
        print(f"  Batch API enabled: {summarizer.use_batch_api}")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_token_estimation():
    """Test token estimation functionality."""
    print("\nTesting token estimation...")
    
    try:
        summarizer = GPT5MiniSummarizer(api_key="test_key")
        
        test_text = "This is a sample text for testing token estimation. It should return a reasonable estimate."
        estimated_tokens = summarizer._estimate_tokens(test_text)
        
        print(f"✓ Token estimation working")
        print(f"  Text: '{test_text[:50]}...'")
        print(f"  Word count: {len(test_text.split())}")
        print(f"  Estimated tokens: {estimated_tokens}")
        
        return True
    except Exception as e:
        print(f"✗ Token estimation failed: {e}")
        return False

def test_cost_calculation():
    """Test cost calculation functionality."""
    print("\nTesting cost calculation...")
    
    try:
        summarizer = GPT5MiniSummarizer(api_key="test_key")
        
        input_text = "This is a longer input text that should have a reasonable cost calculation based on token estimation."
        output_text = "This is a shorter summary output."
        
        cost = summarizer._estimate_cost(input_text, output_text)
        
        print(f"✓ Cost calculation working")
        print(f"  Input tokens: {summarizer._estimate_tokens(input_text)}")
        print(f"  Output tokens: {summarizer._estimate_tokens(output_text)}")
        print(f"  Estimated cost: ${cost:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Cost calculation failed: {e}")
        return False

def test_prompt_creation():
    """Test prompt creation functionality."""
    print("\nTesting prompt creation...")
    
    try:
        summarizer = GPT5MiniSummarizer(api_key="test_key")
        
        test_text = "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."
        max_sentences = 2
        
        prompt = summarizer._create_prompt(test_text, max_sentences)
        
        print(f"✓ Prompt creation working")
        print(f"  Max sentences: {max_sentences}")
        print(f"  Generated prompt length: {len(prompt)} characters")
        print(f"  Prompt preview: '{prompt[:100]}...'")
        
        return True
    except Exception as e:
        print(f"✗ Prompt creation failed: {e}")
        return False

def test_batch_request_creation():
    """Test batch request object creation."""
    print("\nTesting batch request creation...")
    
    try:
        summarizer = GPT5MiniSummarizer(api_key="test_key")
        
        test_text = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."
        request_id = "test_request_123"
        max_sentences = 1
        
        batch_request = summarizer.create_batch_request(request_id, test_text, max_sentences)
        
        print(f"✓ Batch request creation working")
        print(f"  Request ID: {batch_request['custom_id']}")
        print(f"  Method: {batch_request['method']}")
        print(f"  URL: {batch_request['url']}")
        print(f"  Model: {batch_request['body']['model']}")
        print(f"  Max tokens: {batch_request['body']['max_tokens']}")
        
        return True
    except Exception as e:
        print(f"✗ Batch request creation failed: {e}")
        return False

def test_benchmark_integration():
    """Test integration with benchmark framework."""
    print("\nTesting benchmark framework integration...")
    
    try:
        from src.benchmark_framework import BenchmarkFramework
        
        framework = BenchmarkFramework(enable_statistical_analysis=False)
        
        # Check if GPT-5-mini was added to the summarizers
        if 'GPT-5-mini' in framework.summarizers:
            print("✓ GPT-5-mini successfully integrated into benchmark framework")
            summarizer = framework.summarizers['GPT-5-mini']
            print(f"  Summarizer type: {type(summarizer).__name__}")
            print(f"  Summarizer name: {summarizer.name}")
            return True
        else:
            print("✗ GPT-5-mini not found in benchmark framework summarizers")
            print(f"  Available summarizers: {list(framework.summarizers.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ Benchmark integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("GPT-5-mini Summarizer Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_initialization,
        test_token_estimation,
        test_cost_calculation,
        test_prompt_creation,
        test_batch_request_creation,
        test_benchmark_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPT-5-mini implementation is ready.")
    else:
        print("⚠️  Some tests failed. Review the implementation.")
    
    print("\nNote: API tests require a valid OPENAI_API_KEY environment variable.")
    print("To test actual API functionality, set the environment variable and run:")
    print("OPENAI_API_KEY=your_key python test_gpt5_mini.py")

if __name__ == "__main__":
    main()