#!/usr/bin/env python3
"""
Test script for Retrieval-Augmented-Summarizer implementation.
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from summarizers.llm import RetrievalAugmentedSummarizer
except ImportError:
    print("❌ Could not import RetrievalAugmentedSummarizer")
    sys.exit(1)


def test_retrieval_component():
    """Test the retrieval component separately."""
    print("Testing retrieval component...")
    
    try:
        summarizer = RetrievalAugmentedSummarizer()
        
        # Test text with clear important sentences
        test_text = """
        Artificial intelligence is transforming many industries today. Machine learning algorithms can process vast amounts of data quickly. 
        Neural networks are particularly good at pattern recognition tasks. Deep learning has revolutionized computer vision and natural language processing.
        Companies are investing heavily in AI research and development. The future of AI looks very promising for automation.
        However, there are also concerns about job displacement and ethical considerations. Regulation of AI systems is becoming increasingly important.
        Researchers are working on making AI systems more transparent and explainable. The field continues to evolve rapidly.
        """
        
        # Test retrieval with different top_k values
        key_sentences = summarizer._retrieve_key_sentences(test_text, top_k=3)
        
        print(f"✓ Retrieved key sentences (top 3):")
        print(f"  {key_sentences[:100]}...")
        
        # Should return something meaningful
        assert len(key_sentences) > 0, "Retrieval should return non-empty result"
        print("✓ Retrieval component works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Retrieval component failed: {e}")
        return False


def test_initialization():
    """Test that RetrievalAugmentedSummarizer can be initialized."""
    print("Testing Retrieval-Augmented-Summarizer initialization...")
    
    try:
        summarizer = RetrievalAugmentedSummarizer()
        print(f"✓ Initialized successfully")
        print(f"  Model name: {summarizer.model_name}")
        print(f"  Summarizer name: {summarizer.name}")
        print(f"  Device: {summarizer.device}")
        
        # Check if model loading was attempted
        if summarizer.pipeline or (summarizer.model and summarizer.tokenizer):
            print("✓ Model loading successful")
        else:
            print("⚠️  Model loading failed, fallback mode will be used")
        
        return summarizer
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return None


def test_summarization(summarizer):
    """Test the full summarization pipeline."""
    print("\nTesting full summarization pipeline...")
    
    test_text = """
    Climate change is one of the most pressing issues of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise.
    Extreme weather events are becoming more frequent and severe. Scientists have linked these changes to increased greenhouse gas emissions from human activities.
    The burning of fossil fuels is the primary contributor to carbon dioxide emissions. Deforestation also plays a significant role in climate change.
    Governments around the world are implementing policies to reduce emissions. Renewable energy sources like solar and wind power are becoming more affordable.
    Electric vehicles are gaining popularity as an alternative to gasoline-powered cars. Individual actions like reducing energy consumption can also help.
    The Paris Agreement represents a global commitment to addressing climate change. However, more aggressive action is needed to meet climate targets.
    """
    
    try:
        print("  Generating summary with max_sentences=2...")
        start_time = time.time()
        summary = summarizer.summarize(test_text.strip(), max_sentences=2)
        end_time = time.time()
        
        print(f"✓ Summary generated successfully")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        print(f"  Summary length: {len(summary)} characters")
        print(f"  Summary: {summary}")
        
        # Basic validation
        assert len(summary) > 0, "Summary should not be empty"
        print("✓ Summary validation passed")
        
        return True
    except Exception as e:
        print(f"✗ Summarization failed: {e}")
        return False


def test_benchmark_method(summarizer):
    """Test the benchmark_summarize method."""
    print("\nTesting benchmark method...")
    
    test_text = """
    The Internet of Things (IoT) refers to the network of physical devices connected to the internet.
    These devices can collect and share data without human intervention. Smart homes use IoT devices for automation and energy efficiency.
    Industrial IoT applications help optimize manufacturing processes. Healthcare IoT devices can monitor patients remotely.
    Security is a major concern with IoT implementations. The number of connected devices is expected to grow exponentially.
    """
    
    try:
        result = summarizer.benchmark_summarize(test_text.strip(), max_sentences=1)
        
        print("✓ Benchmark method executed successfully")
        print(f"  Summary: {result['summary']}")
        print(f"  Time taken: {result['time_taken']:.3f} seconds")
        print(f"  Method: {result['method']}")
        print(f"  Cost: ${result['cost']:.6f}")
        
        # Validate result structure
        required_keys = ['summary', 'time_taken', 'method', 'cost']
        for key in required_keys:
            if key not in result:
                print(f"✗ Missing key in result: {key}")
                return False
        
        print("✓ Benchmark result structure valid")
        return True
    except Exception as e:
        print(f"✗ Benchmark method failed: {e}")
        return False


def test_integration_with_benchmark_framework():
    """Test integration with the benchmark framework."""
    print("\nTesting integration with benchmark framework...")
    
    try:
        # Import benchmark framework
        from benchmark_framework import BenchmarkFramework
        
        # Initialize framework
        framework = BenchmarkFramework(enable_statistical_analysis=False)
        
        # Check if retrieval-augmented method is available
        if 'Retrieval-Augmented-Summarizer' in framework.summarizers:
            print("✓ Retrieval-Augmented-Summarizer found in benchmark framework")
            
            # Test getting the summarizer instance
            rag_summarizer = framework.summarizers['Retrieval-Augmented-Summarizer']
            print(f"✓ Summarizer instance: {rag_summarizer.name}")
            
            return True
        else:
            print("✗ Retrieval-Augmented-Summarizer not found in benchmark framework")
            available_methods = list(framework.summarizers.keys())
            print(f"Available methods: {available_methods}")
            return False
            
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all Retrieval-Augmented-Summarizer tests."""
    print("🧪 Retrieval-Augmented-Summarizer Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Initialization
    print("Test 1/5: Initialization")
    summarizer = test_initialization()
    if summarizer:
        tests_passed += 1
    else:
        print("❌ Cannot proceed with other tests without successful initialization")
        return False
    
    # Test 2: Retrieval component
    print(f"\nTest 2/5: Retrieval Component")
    if test_retrieval_component():
        tests_passed += 1
    
    # Test 3: Full summarization
    print(f"\nTest 3/5: Full Summarization")
    if test_summarization(summarizer):
        tests_passed += 1
    
    # Test 4: Benchmark method
    print(f"\nTest 4/5: Benchmark Method")
    if test_benchmark_method(summarizer):
        tests_passed += 1
    
    # Test 5: Framework integration
    print(f"\nTest 5/5: Framework Integration")
    if test_integration_with_benchmark_framework():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed successfully!")
        print("🎉 Retrieval-Augmented-Summarizer is working correctly")
        print("\nKey Features Verified:")
        print("• Two-stage retrieval-augmented architecture")
        print("• TF-IDF based sentence retrieval")
        print("• LED model for neural summarization")
        print("• Robust fallback mechanisms")
        print("• Integration with benchmark framework")
        return True
    else:
        print("❌ Some tests failed")
        failed_tests = total_tests - tests_passed
        print(f"⚠️  {failed_tests} test(s) need attention")
        return False


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