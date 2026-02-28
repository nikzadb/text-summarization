#!/usr/bin/env python3
"""
Test script for Pegasus-X summarizer implementation.
"""

import sys
import os
import time
import pytest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from summarizers.llm import PegasusXSummarizer


@pytest.fixture(scope="module")
def summarizer():
    """Create a PegasusXSummarizer instance for testing."""
    return PegasusXSummarizer()


def test_pegasus_x_summarization(summarizer):
    """Test that Pegasus-X can generate summaries."""
    print("\nTesting Pegasus-X summarization...")
    
    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
    For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
    Modern machine learning techniques are integral to most AI systems.
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
        
        return True
    except Exception as e:
        print(f"✗ Failed to generate summary: {e}")
        return False


def test_pegasus_x_benchmark_method(summarizer):
    """Test the benchmark_summarize method."""
    print("\nTesting Pegasus-X benchmark method...")
    
    test_text = """
    Climate change refers to long-term shifts in global or regional climate patterns.
    Since the mid-20th century, humans have been the dominant driver of climate change, primarily due to fossil fuel burning.
    The burning of fossil fuels increases heat-trapping greenhouse gas levels in Earth's atmosphere.
    This leads to rising global temperatures and changes in weather patterns.
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
        
        return True
    except Exception as e:
        print(f"✗ Failed to run benchmark method: {e}")
        return False