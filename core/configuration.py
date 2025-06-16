#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Configuration settings for SciSage pipeline
# [Date]         : Last updated April 21, 2025
# ==================================================================

import os
import asyncio
from dataclasses import dataclass, field, fields
from typing import Any, Optional, List
import asyncio
from contextlib import asynccontextmanager

# ==========================================================================
# MODEL CONFIGURATION
# ==========================================================================

# Complete list of all valid models that can be used in the system
ALL_VALID_MODELS: List[str] = [
    "gpt-4",  # OpenAI GPT-4 base model
    "gpt-4-32k",  # OpenAI GPT-4 with 32k context
    "gpt-4o-mini",  # OpenAI GPT-4o mini model
    "gpt-35",  # OpenAI GPT-3.5 Turbo
    "r1-llama70b",  # Anthropic Claude
    "Qwen25-72B",  # Alibaba Qwen 72B parameter model
    "Qwen25-7B",  # Alibaba Qwen 7B parameter model
    "qwq-32b",  # QWQ 32B model
    "llama3-70b",  # Meta LLaMA3 70B model
    "opensholar",  # OpenScholar specialized academic model
    "Qwen3-32B",
]

# Models used for generating multiple outlines
# This list contains models that will be used in parallel to generate diverse outlines


# ==========================================================================
# PROCESS CONTROL PARAMETERS
# ==========================================================================
# -- QUERY INTENT --
DEFAULT_MODEL_FOR_QUERY_INTENT = (
    "Qwen3-32B"  # Model used for generating search queries from outline content points
)

# -- OUTLINE --
DEFAULT_MODEL_FOR_OUTLINE: str = "Qwen3-32B"  # Changed from gpt-4-32k to Qwen25-72B
OUTLINE_GENERAOR_MODELS: List[str] = [
    "Qwen3-32B",  # Currently using only Qwen model for outline generation
    "Qwen3-14B",
    "Qwen25-72B",  # Another Qwen model for generating outlines
    "llama3-70b",
    # "gpt-4o-mini",
]
MODEL_GEN_QUERY: str = "Qwen3-32B"

OUTLINE_REFLECTION_MAX_TURNS: int = (
    2  # Maximum number of outline reflections to perform
)
OUTLINE_MAX_SECTIONS: int = 4  # Maximum number of sections to generate in the outline
OUTLINE_MIN_DEPTH: int = 1  # Minimum depth of the outline tree structure

# -- SECTION --
# Enable selective reflection on specific sections
DO_SELECT_REFLECTION: bool = True
# Number of concurrent section writer processes
SECTION_WRITER_CONCURRENCY: int = 2
# Number of concurrent section reflection processes
SECTION_REFLECTION_CONCURRENCY: int = 2
#  Number of turns for each section reflection
SECTION_REFLECTION_MAX_TURNS: int = 0

MAX_SECTION_RETRY_NUM: int = 3  # Maximum number of retries for section generation

DEFAULT_MODEL_FOR_SECTION_WRITER: str = "Qwen3-32B"
SECTION_SUMMARY_MODEL: str = "Qwen3-32B"  # gpt-4-32k
SECTION_REFLECTION_MODEL_LST = ["Qwen3-32B", "llama3-70b"]


# ---- GLOBAL REFLECTION ----
DO_GLOBAL_REFLECTION: bool = True  # Enable global reflection on entire paper
GLOBAL_REFLECTION_MAX_TURNS: int = 1  # Maximum number of global reflection turns
DEFAULT_MODEL_FOR_GLOBAL_REFLECTION: str = "Qwen3-32B"

# --- ABSTRACT & CONCLUSION ---
GLOBAL_ABSTRACT_CONCLUSION_MAX_TURNS: int = 1
MODEL_GEN_ABSTRACT_CONCLUSION: str = "Qwen3-32B"

# -- POOLISH --
DEFAULT_MODEL_FOR_SECTION_NAME_REFLECTION: str = "Qwen3-32B"

# ==========================================================================
# SERVICE CONFIGURATION
# ==========================================================================

# Default URL for the RAG (Retrieval Augmented Generation) service
DEFAULT_RAG_SERVICE_URL: str = "http://xxxx:9528/chat"

# Debug mode flag - enables additional logging and debug information when True
DEBUG: bool = False

# Debug mode, if enabled, will limit the number of key points to be processed
DEBUG_KEY_POINTS_LIMIT: int = 1

# ==========================================================================
# RESOURCE MANAGEMENT
# ==========================================================================

# Maximum number of concurrent RAG requests
SEMAPHORE_LIMIT: int = 2

# Semaphore for limiting concurrent access to the RAG service
# This prevents overwhelming the service with too many simultaneous requests
rag_semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


# 添加一个全局信号量池来控制不同类型操作的并行度
class GlobalSemaphorePool:
    """并行度控制"""

    # 段落写作，
    rag_semaphore = asyncio.Semaphore(2)
    # 段落反思，控制reflection_semaphore的并行度
    section_reflection_semaphore = asyncio.Semaphore(3)
    # 全文润色，控制标题重命名的并行度
    section_name_refine_semaphore = asyncio.Semaphore(2)


# 创建全局信号量池实例
global_semaphores = GlobalSemaphorePool()
