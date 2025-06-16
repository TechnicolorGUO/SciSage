# SciSage Core

The core processing module of SciSage project, providing a complete workflow for academic paper understanding, outline generation, and content writing.

## Overview

SciSage Core implements an intelligent academic paper processing pipeline that includes:

- **Paper Understanding & Analysis**: Deep comprehension of input papers to extract key insights
- **Intelligent Outline Generation**: Structure-aware outline creation based on paper content
- **Section-wise Content Generation**: Detailed analysis generation for each section
- **Global Reflection & Optimization**: Holistic review and refinement of generated content
- **Multi-model Support**: Compatible with both local and cloud-based AI models

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Model Configuration

Edit [`model_factory.py`](model_factory.py) to configure available models:

```python
llm_map = {
    "gpt-4": AzureChatOpenAI(
        openai_api_type="azure",
        openai_api_version="your-api-version",
        azure_deployment="gpt-4",
        azure_endpoint="your-endpoint",
        api_key="your-api-key",
    ),
    "gpt-4o-mini": AzureChatOpenAI(...),
    # Add more model configurations
}
```

### 3. Global Configuration

Edit [`configuration.py`](configuration.py) to set models for different stages:

```python
# Configure model names for different pipeline stages
PAPER_UNDERSTANDING_MODEL = "gpt-4"
OUTLINE_GENERATION_MODEL = "gpt-4o-mini"
CONTENT_GENERATION_MODEL = "gpt-4"
REFLECTION_MODEL = "gpt-4"
```

### 4. Run the Pipeline

```bash
python3 main_workflow_opt_for_paper.py
```

## Core Components

### Main Workflow
- [`main_workflow_opt_for_paper.py`](main_workflow_opt_for_paper.py): Main processing pipeline orchestrating all stages

### Paper Processing Modules
- [`paper_understant_query.py`](paper_understant_query.py): Paper understanding and query processing
- [`paper_outline_opt.py`](paper_outline_opt.py): Intelligent outline generation
- [`section_writer_opt.py`](section_writer_opt.py): Section-wise content writing
- [`paper_global_reflection_opt.py`](paper_global_reflection_opt.py): Global content reflection and optimization

### Supporting Components
- [`model_factory.py`](model_factory.py): Model initialization and management
- [`local_model_langchain.py`](local_model_langchain.py): Local model integrations
- [`prompt_manager.py`](prompt_manager.py): Prompt template management
- [`configuration.py`](configuration.py): Global configuration settings

### Utilities
- [`log.py`](log.py): Logging utilities
- [`debug.py`](debug.py): Debugging tools
- [`fallback.py`](fallback.py): Fallback mechanisms for model failures

## Workflow Stages

1. **Paper Understanding**: Analyze input paper content and extract key information
2. **Outline Generation**: Create structured outline based on paper analysis
3. **Section Writing**: Generate detailed content for each outline section
4. **Section Reflection**: Reflect generate detailed content for each outline section
5. **Global Reflection**: Review and optimize the complete generated content
6. **Final Polish**: Apply final refinements and formatting

## Configuration

### Model Selection
Configure different models for different stages in [`configuration.py`](configuration.py):
- High-capability models (e.g., GPT-4) for complex reasoning tasks
- Efficient models (e.g., GPT-4o-mini) for simpler tasks
- Local models for privacy-sensitive scenarios

### Prompt Customization
Modify prompts in [`prompt_manager.py`](prompt_manager.py) to customize:
- Paper analysis instructions
- Outline generation guidelines
- Content writing styles
- Reflection criteria

## Examples

Check [`example.py`](example.py) and [`example_full_data.json`](example_full_data.json) for usage examples and sample data formats.

## Requirements

See [`requirements.txt`](requirements.txt) for detailed dependencies.

## Architecture

The system follows a modular architecture where each processing stage is independent and configurable, allowing for:
- Easy model swapping
- Stage-specific optimization
- Parallel processing capabilities
- Robust error handling and fallback mechanisms