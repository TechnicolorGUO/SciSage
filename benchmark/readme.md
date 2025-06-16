# ArXiv Paper Information Extractor

A robust tool for extracting comprehensive information from arXiv papers, including full-text content and references.

## Features

- **Multi-source Content Extraction**: Fetches paper content from multiple sources with fallback mechanisms
- **Complete Metadata Extraction**: Retrieves comprehensive paper metadata from arXiv API
- **Reference Processing**: Automatically fetches paper references using Semantic Scholar
- **Robust Error Handling**: Implements retry logic and proxy support
- **Incremental Processing**: Supports resuming and updating existing datasets

## Installation

```bash
pip install crawl4ai arxiv requests pdfminer semanticscholar
```

## Usage

```shell
python get_paper_info.py
```



## Output Format

Each line in the output JSONL file contains:
- Paper metadata (title, authors, abstract, etc.)
- Full-text content extracted from multiple sources
- Complete reference list with metadata
- Additional metadata (categories, DOI, etc.)
