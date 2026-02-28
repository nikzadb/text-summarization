#!/usr/bin/env python3
"""
CUDA memory cleanup utility
"""

import gc
import os

def clear_cuda_memory():
    """Clear all CUDA memory and Python garbage collection."""
    try:
        import torch
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory info
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            cached = torch.cuda.memory_reserved() / (1024**3)  # GB
            
            print(f"🧹 CUDA memory cleared")
            print(f"📊 Memory allocated: {allocated:.2f} GB")
            print(f"📊 Memory cached: {cached:.2f} GB")
            
            return True
        else:
            print("⚠️  CUDA not available")
            return False
    except ImportError:
        print("⚠️  PyTorch not available")
        return False

def clear_python_memory():
    """Clear Python garbage collection."""
    gc.collect()
    print("🧹 Python garbage collection completed")

def main():
    print("🔧 CUDA Memory Cleanup Utility")
    print("=" * 40)
    
    clear_cuda_memory()
    clear_python_memory()
    
    print("✅ Memory cleanup completed")

if __name__ == '__main__':
    main()