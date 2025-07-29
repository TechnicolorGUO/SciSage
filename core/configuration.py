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
from typing import List, Dict, Any, Optional, Union

import asyncio
from contextlib import asynccontextmanager
from openai import OpenAI  # 引入 OpenAI 客户端

# ==========================================================================
# MODEL CONFIGURATION
# ==========================================================================

# Complete list of all valid models that can be used in the system
ALL_VALID_MODELS: List[str] = [
    "gpt-4",  # OpenAI GPT-4 base model
    "gpt-4-32k",  # OpenAI GPT-4 with 32k context
    "gpt-4o-mini",  # OpenAI GPT-4o mini model
    "Qwen25-72B",  # Alibaba Qwen 72B parameter model
    "Qwen25-7B",  # Alibaba Qwen 7B parameter model
    "qwq-32b",  # QWQ 32B model
    "llama3-70b",  # Meta LLaMA3 70B model
    "Qwen3-32B",
]



@dataclass
class ModelConfig:
    """Configuration for LLM models"""

    url: str
    max_len: int
    temperature: float = 0.8
    model_name: str = ""
    top_p: float = 0.9
    top_k: int = 20
    min_p: int = 0
    retry_attempts: int = 20
    timeout: int = 200
    think_bool: bool = False
    openai_client: Optional[Any] = None
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)


HOST = os.getenv("LOCAL_LLM_HOST", "http://0.0.0.0")

print(f"Using LLM host: {HOST}")
# Model configurations
MODEL_CONFIGS = {
    "llama3-70b": ModelConfig(
        url=f"{HOST}:9087/v1/chat/completions",
        max_len=131072,
    ),
    "Qwen3-8B": ModelConfig(
        url=f"{HOST}:9096/v1",
        max_len=131072,
        model_name="Qwen/Qwen3-8B",
        think_bool=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url=f"{HOST}:9096/v1",
        ),
    ),
    "Qwen3-14B": ModelConfig(
        url=f"{HOST}:9095/v1",
        max_len=131072,
        model_name="Qwen/Qwen3-14B",
        think_bool=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url=f"{HOST}:9095/v1",
        ),
    ),
    "Qwen3-32B": ModelConfig(
        url=f"{HOST}:9094/v1",
        max_len=131072,
        model_name="Qwen/Qwen3-32B",
        think_bool=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url=f"{HOST}:9094/v1",
        ),
    ),
    "Qwen3-32B-think": ModelConfig(
        url=f"{HOST}:9094/v1",
        max_len=131072,
        model_name="Qwen/Qwen3-32B",
        think_bool=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url=f"{HOST}:9094/v1",
        ),
    ),
}


# Models used for generating multiple outlines
# This list contains models that will be used in parallel to generate diverse outlines


# ==========================================================================
# PROCESS CONTROL PARAMETERS
# ==========================================================================
# -- QUERY INTENT --
DEFAULT_MODEL_FOR_QUERY_INTENT = (
    "Qwen3-8B"  # Model used for generating search queries from outline content points
)

# -- OUTLINE --
DEFAULT_MODEL_FOR_OUTLINE: str = "Qwen3-8B"  # Changed from gpt-4-32k to Qwen25-72B
# Currently using only Qwen model for outline generation
OUTLINE_GENERAOR_MODELS: List[str] = [
    # "Qwen3-32B",
    # "Qwen3-14B",
    "Qwen3-8B",
]
MODEL_GEN_QUERY: str = "Qwen3-8B"

# Maximum number of outline reflections to perform
OUTLINE_REFLECTION_MAX_TURNS: int = 2
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

DEFAULT_MODEL_FOR_SECTION_RETRIVAL: str = "Qwen3-8B"


DEFAULT_MODEL_FOR_SECTION_WRITER: str = "Qwen3-8B"
DEFAULT_MODEL_FOR_SECTION_WRITER_IMAGE_EXTRACT:str = "Qwen3-8B"
DEFAULT_MODEL_FOR_SECTION_WRITER_RERANK:str = "Qwen3-8B"

SECTION_SUMMARY_MODEL: str = "Qwen3-8B"  # gpt-4-32k
SECTION_REFLECTION_MODEL_LST = ["Qwen3-8B", "llama3-70b"]


# ---- GLOBAL REFLECTION ----
DO_GLOBAL_REFLECTION: bool = True  # Enable global reflection on entire paper
GLOBAL_REFLECTION_MAX_TURNS: int = 1  # Maximum number of global reflection turns
DEFAULT_MODEL_FOR_GLOBAL_REFLECTION: str = "Qwen3-8B"

# --- ABSTRACT & CONCLUSION ---
GLOBAL_ABSTRACT_CONCLUSION_MAX_TURNS: int = 1
MODEL_GEN_ABSTRACT_CONCLUSION: str = "Qwen3-8B"

# -- POOLISH --
DEFAULT_MODEL_FOR_SECTION_NAME_REFLECTION: str = "Qwen3-8B"

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
