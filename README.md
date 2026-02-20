# Text Summarization Benchmarking Tool

A comprehensive benchmarking framework for evaluating various text summarization techniques across multiple datasets. This tool supports traditional extractive methods, transformer-based models, and API-based services with detailed performance analysis.

## Features

### Summarization Methods
- **Traditional Methods**: TextRank, TF-IDF Rank
- **Transformer Models**: T5, DistilBART, BART (facebook/bart-large-cnn)
- **API Integration**: Google Gemini
- **Hybrid Methods**: TextRank+Gemini, TF-IDF+Gemini combinations

### Supported Datasets
- **CNN/DailyMail**: News article summarization
- **arXiv**: Scientific paper abstracts
- **WikiHow**: Instructional content
- **GovReport**: Government report summaries
- **SAMSum**: Dialogue summarization
- **QMSum**: Meeting transcript summaries
- **MediaSum**: Interview transcript summaries

### Evaluation Metrics
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)
- **BERTScore**: Semantic similarity evaluation
- **Performance Metrics**: Processing time, API costs
- **Combined Scoring**: Weighted metric combination


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text-summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
# For Gemini API access
export GEMINI_API_KEY=your_api_key_here
```

## Quick Start

### Basic Usage
```bash
# Run benchmark with default settings (CNN/DailyMail dataset, all methods)
python main.py

# Benchmark specific datasets and methods
python main.py --datasets cnn_dailymail arxiv --methods textrank t5 gemini

# Limit sample size and customize output
python main.py --max-samples 50 --output results.csv
```

### Advanced Usage
```bash
# Enable AWS Lambda simulation
python main.py --use-lambda --datasets samsum --methods textrank tfidfrank

# Custom summary length
python main.py --max-sentences 5 --datasets govreport

# Multiple datasets with API methods
python main.py \
    --datasets cnn_dailymail arxiv wikihow \
    --methods gemini hybrid_textrank_gemini \
    --max-samples 100 \
    --gemini-api-key YOUR_API_KEY
```