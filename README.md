# Text Summarization Benchmarking Tool

A comprehensive benchmarking framework for evaluating various text summarization techniques across multiple datasets. This tool supports traditional extractive methods, transformer-based models, and API-based services with detailed performance analysis.

## Features

### Summarization Methods
- **Traditional Methods**: TextRank, TF-IDF Rank
- **Transformer Models**: DistilBART, BART (facebook/bart-large-cnn)
- **API Integration**: Google Gemini-2.5-Flash, OpenAI GPT-5-mini

### Supported Datasets
- **CNN/DailyMail**: News article summarization
- **arXiv**: Scientific paper abstracts
- **WikiHow**: Instructional content
- **GovReport**: Government report summaries
- **MediaSum**: Interview transcript summaries

### Evaluation Metrics
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)
- **BERTScore**: Semantic similarity evaluation
- **BLEURT**: Learned semantic similarity
- **Performance Metrics**: Processing time, API costs
- **Combined Scoring**: Weighted metric combination


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text-summarization
```

2. Create virtual env
```bash
python3 -m venv .venv
```

3. Install dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

3. Create .env and add environment variables:
```bash
# For Gemini API access
export GEMINI_API_KEY=your_gemini_api_key_here

# For OpenAI API access
export OPEN_API_KEY=your_openai_api_key_here
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
# Custom summary length
python main.py --max-sentences 5 --datasets govreport

# Multiple datasets with API methods
python main.py \
    --datasets cnn_dailymail arxiv wikihow \
    --methods gemini hybrid_textrank_gemini \
    --max-samples 100
```
