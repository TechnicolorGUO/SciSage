# SciSage Core

The core processing pipeline for intelligent academic paper analysis and content generation.

## Overview

SciSage Core implements a multi-stage AI pipeline:

1. **Paper Understanding** - Analyze input papers and extract insights
2. **Outline Generation** - Create structured analysis outlines
3. **Content Writing** - Generate detailed section content
4. **Global Reflection** - Review and optimize generated content
5. **Final Polish** - Apply refinements and generate visualizations

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Configure Models
1. Edit [`model_factory.py`](model_factory.py):
```python
llm_map = {
    "gpt-4": AzureChatOpenAI(
        azure_endpoint="your-endpoint",
        api_key="your-key"
    ),
}
```

2. Edit [`run.sh`](run.sh), set the environment.


### Configure Pipeline
Edit [`configuration.py`](configuration.py):
```python
PAPER_UNDERSTANDING_MODEL = "gpt-4"
OUTLINE_GENERATION_MODEL = "gpt-4o-mini"
CONTENT_GENERATION_MODEL = "gpt-4"
```

### Run Pipeline
```bash
bash run.sh
```

## Key Components

- [`main_workflow_opt_for_paper.py`](main_workflow_opt_for_paper.py) - Main pipeline orchestrator
- [`paper_outline_opt.py`](paper_outline_opt.py) - Outline generation with reflection
- [`section_writer_opt.py`](section_writer_opt.py) - RAG-based content writing
- [`paper_global_reflection_opt.py`](paper_global_reflection_opt.py) - Global optimization
- [`paper_poolish_opt.py`](paper_poolish_opt.py) - Final polishing and formatting

## Configuration Options

### Model Selection
- **Cloud Models**: GPT-4, GPT-4o-mini via Azure OpenAI
- **Local Models**: Support for locally hosted models via [`local_model_langchain.py`](local_model_langchain.py)

### Prompt Customization
Modify prompts in [`prompt_manager.py`](prompt_manager.py) for different:
- Analysis styles
- Content formats
- Reflection criteria

### Parallel Processing
Configure concurrency in [`configuration.py`](configuration.py):
```python
global_semaphores.rag_semaphore = asyncio.Semaphore(2)
global_semaphores.section_reflection_semaphore = asyncio.Semaphore(3)
```

## Examples

See [`example.py`](example.py) and [`example_full_data.json`](example_full_data.json) for usage examples.

## Architecture

The system uses a modular, async-first design enabling:
- Independent stage processing
- Easy model swapping
- Robust error handling
- Parallel execution optimization