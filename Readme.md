# SciSage

An intelligent academic paper analysis system that leverages AI to understand research papers and generate comprehensive analysis reports.

📄 **Paper**: https://arxiv.org/abs/2506.12689
📊 **Benchmark**: https://huggingface.co/datasets/BAAI/SurveyScope

## Features

- **Multi-source Paper Extraction**: Robust crawling from arXiv with fallback mechanisms
- **Intelligent Analysis**: AI-powered paper understanding and outline generation
- **Structured Content Generation**: Section-wise detailed analysis with proper citations
- **Multi-model Support**: Compatible with GPT-4, local models, and cloud services

## Quick Start

### 1. Installation
```bash
git clone https://github.com/your-repo/SciSage.git
cd SciSage
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Configure models in core/model_factory.py
# Set API keys and endpoints in core/configuration.py
```


### 4. Run Analysis
```bash
cd core
python main_workflow_opt_for_paper.py
```


## Project Structure

```
SciSage/
├── benchmark/              # Paper extraction tools
│   └── get_paper_info.py   # Multi-source paper crawler for benchmark build
├── core/                   # Analysis pipeline
│   ├── main_workflow_opt_for_paper.py  # Main orchestrator
│   ├── paper_outline_opt.py            # Outline generation
│   ├── paper_poolish_opt.py            # Content polishing
│   ├── model_factory.py                # Model management
│   └── configuration.py                # Settings
└── eval/                   # Evaluation tools
```

## Configuration

### Model Setup
Edit [`core/model_factory.py`](core/model_factory.py):
```python
llm_map = {
    "gpt-4": AzureChatOpenAI(...),
    "gpt-4o-mini": AzureChatOpenAI(...),
}
```

### Pipeline Settings
Edit [`core/configuration.py`](core/configuration.py):
```python
OUTLINE_GENERATION_MODEL = "gpt-4o-mini"
CONTENT_GENERATION_MODEL = "gpt-4"
REFLECTION_MODEL = "gpt-4"
```

## Example Usage

```python
# Extract paper information
from benchmark.get_paper_info import process_arxiv_papers

arxiv_urls = ["https://arxiv.org/abs/2306.11646"]
await process_arxiv_papers(arxiv_urls, "papers.jsonl")

# Run analysis pipeline
from core.main_workflow_opt_for_paper import run_analysis_pipeline

result = run_analysis_pipeline(
    paper_data="paper_content.json",
    output_path="analysis_output.json"
)
```

## Requirements

- Python 3.8+
- langchain, arxiv, requests
- See [`requirements.txt`](requirements.txt) for full dependencies

## License

MIT License - see LICENSE file for details.