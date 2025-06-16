# SciSage

An intelligent academic paper analysis and content generation system that leverages AI to understand research papers and generate comprehensive analysis reports.

## Overview

SciSage is a comprehensive toolkit for academic research that combines paper information extraction, intelligent analysis, and automated content generation. The system is designed to help researchers quickly understand and analyze academic papers through AI-powered workflows.

## Features

- **Automated Paper Information Extraction**: Robust crawling and extraction of arXiv papers with multi-source fallback mechanisms
- **Intelligent Paper Understanding**: Deep comprehension of research papers using advanced language models
- **Structured Outline Generation**: Automatic creation of well-organized analysis outlines
- **Section-wise Content Generation**: Detailed analysis generation for each paper section
- **Multi-model Support**: Compatible with both local and cloud-based AI models (GPT-4, local models)
- **Robust Error Handling**: Comprehensive retry mechanisms and fallback strategies

## Project Structure

```
SciSage/
├── benchmark/           # Paper information extraction tools
│   ├── get_paper_info.py   # Main paper extraction script
│   └── readme.md           # Benchmark module documentation
├── core/               # Core processing pipeline
│   ├── main_workflow_opt_for_paper.py  # Main workflow orchestrator
│   ├── paper_understant_query.py       # Paper understanding module
│   ├── paper_outline_opt.py            # Outline generation
│   ├── section_writer_opt.py           # Content writing
│   ├── paper_global_reflection_opt.py  # Global reflection
│   ├── model_factory.py                # Model management
│   ├── configuration.py                # Global configuration
│   └── ...                            # Supporting utilities
└── README.md           # This file
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-repo/SciSage.git
cd SciSage
pip install -r requirements.txt
```

### 2. Configuration

#### Model Setup
Edit `core/model_factory.py` to configure your AI models:

```python
llm_map = {
    "gpt-4": AzureChatOpenAI(
        openai_api_type="azure",
        openai_api_version="2024-05-01-preview",
        azure_deployment="gpt-4",
        azure_endpoint="your-endpoint",
        api_key="your-api-key",
    ),
    # Add more model configurations
}
```

#### Pipeline Configuration
Set model preferences in `core/configuration.py`:

```python
PAPER_UNDERSTANDING_MODEL = "gpt-4"
OUTLINE_GENERATION_MODEL = "gpt-4o-mini"
CONTENT_GENERATION_MODEL = "gpt-4"
REFLECTION_MODEL = "gpt-4"
```

### 3. Extract Paper Information

```bash
cd benchmark
python get_paper_info.py
```

### 4. Run Analysis Pipeline

```bash
cd core
python main_workflow_opt_for_paper.py
```

## Usage Examples

### Paper Information Extraction

```python
import asyncio
from benchmark.get_paper_info import process_arxiv_papers

arxiv_urls = [
    "https://arxiv.org/abs/2306.11646",
    "https://arxiv.org/abs/2102.12982",
]

# Extract paper information
asyncio.run(process_arxiv_papers(arxiv_urls, "papers.jsonl"))
```

### Paper Analysis Pipeline

```python
from core.main_workflow_opt_for_paper import run_analysis_pipeline

# Run complete analysis pipeline
result = run_analysis_pipeline(
    paper_data="paper_content.json",
    output_path="analysis_output.json"
)
```

## Core Modules

### Benchmark Module
- **Paper Extraction**: Robust extraction from multiple sources (ar5iv, arxiv-vanity, PDF)
- **Metadata Collection**: Comprehensive paper metadata from arXiv API
- **Reference Processing**: Automatic reference extraction using Semantic Scholar

### Core Processing Pipeline
- **Paper Understanding**: Deep analysis of paper content and structure
- **Outline Generation**: Intelligent creation of analysis outlines
- **Content Writing**: Section-by-section detailed analysis generation
- **Global Reflection**: Holistic review and optimization of generated content

## Configuration Options

### Multi-source Paper Extraction
The system tries multiple sources in priority order:
1. ar5iv (HTML rendered papers)
2. arxiv-vanity (clean paper rendering)
3. arXiv HTML (official HTML version)
4. PDF extraction (fallback using pdfminer)

### Model Selection
- **Cloud Models**: GPT-4, GPT-4o-mini via Azure OpenAI
- **Local Models**: Support for locally hosted models
- **Fallback Mechanisms**: Automatic model switching on failures

### Proxy Support
Configure proxy settings for network-restricted environments:

```python
proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}
```

## Output Format

The system generates structured JSON outputs containing:
- Original paper metadata and content
- Generated analysis outline
- Section-wise detailed analysis
- Global reflection and insights
- Processing metadata and statistics

## Requirements

- Python 3.8+
- crawl4ai
- langchain
- arxiv
- requests
- pdfminer
- semanticscholar
- Additional dependencies in `requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- arXiv for providing open access to research papers
- Semantic Scholar for reference data
- The open-source community for the underlying libraries

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation in each module's README
- Review the example files for usage patterns