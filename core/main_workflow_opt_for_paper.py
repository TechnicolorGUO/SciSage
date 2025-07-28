#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : AI Assistant
# [Descriptions] : Optimized main orchestration for academic paper generation pipeline
# ==================================================================

import asyncio
import json
import os
import random
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from functools import wraps
from fallback import create_reflection_fallback

from paper_understant_query import process_query_async
from paper_outline_opt import generate_paper_outline_async

# from section_writer_opt import section_writer_async
from section_writer_opt_local import section_writer_async
from section_reflection_opt import section_reflection
from paper_global_reflection_opt import global_reflection
from paper_abstract_conclusion_opt import generate_abstract_conclusion
from paper_poolish_opt import process_poolish_data

from log import logger
from configuration import (
    OUTLINE_REFLECTION_MAX_TURNS,
    OUTLINE_MAX_SECTIONS,
    OUTLINE_MIN_DEPTH,
    MAX_SECTION_RETRY_NUM,
    DO_SELECT_REFLECTION,
    SECTION_REFLECTION_MAX_TURNS,
    DEFAULT_MODEL_FOR_SECTION_WRITER,
    DO_GLOBAL_REFLECTION,
    GLOBAL_REFLECTION_MAX_TURNS,
    GLOBAL_ABSTRACT_CONCLUSION_MAX_TURNS,
    DEBUG_KEY_POINTS_LIMIT,
    DEFAULT_RAG_SERVICE_URL,
    DEBUG,
)
from utils import format_sections_for_global_reflection
from models import SectionData


# Define process states as an enum
class ProcessState(Enum):
    INITIALIZED = auto()
    OUTLINE_GENERATED = auto()
    SECTIONS_WRITTEN = auto()
    SECTIONS_REFLECTED = auto()
    GLOBAL_REFLECTION_DONE = auto()
    ABSTRACT_CONCLUSION_GENERATED = auto()
    PAPER_POLISHED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class PaperGenerationState:
    """State container for the paper generation process"""

    task_id: str
    user_name: str
    user_query: str
    timestamp: str = ""
    rewrite_query: str = ""
    query_type: str = ""
    research_field: str = ""
    rag_service_url: str = ""
    paper_title: str = ""
    process_state: ProcessState = ProcessState.INITIALIZED
    error_message: str = ""
    retries: Dict[str, int] = field(default_factory=dict)

    # Data structures for each stage
    outline_details: Dict[str, Any] = field(default_factory=dict)
    outline_structure_w_query: Dict[str, Any] = field(default_factory=dict)
    outline_structure_wo_query: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, SectionData] = field(default_factory=dict)
    global_reflection_result: Dict[str, Any] = field(default_factory=dict)
    abstract_conclusion: Dict[str, Any] = field(default_factory=dict)
    final_paper: Dict[str, Any] = field(default_factory=dict)
    gen_keywords: Dict[str, Any] = field(default_factory=dict)
    # Process tracking
    execution_times: Dict[str, float] = field(default_factory=dict)
    # Add step error tracking
    step_errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        result = asdict(self)
        # Convert enum to string for JSON serialization
        result["process_state"] = self.process_state.name
        # Convert SectionData objects to dict
        # if "sections" in result:
        #     result["sections"] = {k: asdict(v) for k, v in result["sections"].items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperGenerationState":
        """Create instance from dictionary"""
        # Convert string back to enum
        if "process_state" in data and isinstance(data["process_state"], str):
            data["process_state"] = ProcessState[data["process_state"]]

        # Convert sections dict back to SectionData objects
        if "sections" in data and isinstance(data["sections"], dict):
            sections = {}
            for k, v in data["sections"].items():
                if isinstance(v, dict):
                    sections[k] = SectionData(**v)
                else:
                    sections[k] = v
            data["sections"] = sections

        # Ensure backward compatibility for step_errors field
        if "step_errors" not in data:
            data["step_errors"] = {}

        return cls(**data)

    def has_step_error(self, step_name: str) -> bool:
        """Check if a specific step has error"""
        return step_name in self.step_errors

    def clear_step_error(self, step_name: str):
        """Clear error for a specific step"""
        if step_name in self.step_errors:
            del self.step_errors[step_name]

    def set_step_error(self, step_name: str, error_message: str):
        """Set error for a specific step"""
        self.step_errors[step_name] = error_message


def find_existing_process_file(
    user_query: str, output_dir: str = "temp"
) -> Optional[str]:
    """Find existing _process.json file for the given query"""
    if not os.path.exists(output_dir):
        return None

    # Clean query for filename matching
    clean_query = user_query.replace(" ", "_")[:50]  # Limit length

    # Look for files that match the pattern
    for filename in os.listdir(output_dir):
        if filename.endswith("_process.json") and clean_query in filename:
            return os.path.join(output_dir, filename)

    return None


def load_existing_state(filepath: str) -> Optional[PaperGenerationState]:
    """Load existing state from _process.json file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = PaperGenerationState.from_dict(data)

        # state.step_errors = {"abstract_conclusion":"error in abstract_conclusion step"}
        # state.process_state = ProcessState.GLOBAL_REFLECTION_DONE

        logger.info(f"Loaded existing state from {filepath}")
        logger.info(f"Current process state: {state.process_state.name}")

        return state, filepath
    except Exception as e:
        logger.error(f"Failed to load existing state from {filepath}: {e}")
        return None


def determine_resume_point(
    state: PaperGenerationState, do_global_reflection
) -> ProcessState:
    """Determine from which step to resume based on state and errors"""

    # Check for step-specific errors first
    if state.has_step_error("outline"):
        logger.info("Found error in outline step, resuming from outline generation")
        return ProcessState.INITIALIZED

    if state.has_step_error("sections"):
        logger.info("Found error in sections step, resuming from section processing")
        return ProcessState.OUTLINE_GENERATED

    if state.has_step_error("global_reflection"):
        logger.info(
            "Found error in global reflection step, resuming from global reflection"
        )
        return ProcessState.SECTIONS_WRITTEN

    if state.has_step_error("abstract_conclusion"):
        logger.info(
            "Found error in abstract/conclusion step, resuming from abstract generation"
        )
        if do_global_reflection:
            return ProcessState.GLOBAL_REFLECTION_DONE
        else:
            return ProcessState.SECTIONS_WRITTEN

    if state.has_step_error("polish"):
        logger.info("Found error in polish step, resuming from paper polishing")
        return ProcessState.ABSTRACT_CONCLUSION_GENERATED

    # Check process state and data integrity
    if state.process_state == ProcessState.ERROR:
        logger.info("Process state is ERROR, starting from beginning")
        return ProcessState.INITIALIZED

    if state.process_state == ProcessState.COMPLETED:
        logger.info("Process already completed successfully")
        return ProcessState.COMPLETED

    # Check data integrity for each step
    if state.process_state.value >= ProcessState.OUTLINE_GENERATED.value:
        if not state.outline_structure_wo_query or not state.paper_title:
            logger.info("Outline data incomplete, resuming from outline generation")
            return ProcessState.INITIALIZED

    if state.process_state.value >= ProcessState.SECTIONS_WRITTEN.value:
        if not state.sections:
            logger.info("Sections data incomplete, resuming from section processing")
            return ProcessState.OUTLINE_GENERATED

    if (
        do_global_reflection
        and state.process_state.value >= ProcessState.GLOBAL_REFLECTION_DONE.value
    ):
        if not state.global_reflection_result:
            logger.info(
                "Global reflection data incomplete, resuming from global reflection"
            )
            return ProcessState.SECTIONS_WRITTEN

    if state.process_state.value >= ProcessState.ABSTRACT_CONCLUSION_GENERATED.value:
        if not state.abstract_conclusion:
            logger.info(
                "Abstract/conclusion data incomplete, resuming from abstract generation"
            )
            if do_global_reflection:
                return ProcessState.GLOBAL_REFLECTION_DONE
            else:
                return ProcessState.SECTIONS_WRITTEN

    if state.process_state.value >= ProcessState.PAPER_POLISHED.value:
        if not state.final_paper:
            logger.info("Final paper data incomplete, resuming from paper polishing")
            return ProcessState.ABSTRACT_CONCLUSION_GENERATED

    # If everything looks good, continue from current state
    logger.info(f"Resuming from current state: {state.process_state.name}")
    return state.process_state


# Async semaphore pool for concurrent operation management
class AsyncSemaphorePool:
    """Centralized semaphore management for concurrent operations"""

    def __init__(self):
        self.semaphores = {
            "section_writer": asyncio.Semaphore(2),  # Limit section_writer concurrency
            "section_reflection": asyncio.Semaphore(2),  # Limit reflection concurrency
            "global": asyncio.Semaphore(5),  # General purpose concurrency limit
        }

    async def with_semaphore(self, name: str, func: Callable, *args, **kwargs):
        """Execute function with semaphore protection"""
        semaphore = self.semaphores.get(name, self.semaphores["global"])
        async with semaphore:
            return await func(*args, **kwargs)


# Create global semaphore pool
semaphore_pool = AsyncSemaphorePool()


# Decorator for error handling and retry logic
def with_retry_and_fallback(max_retries: int = 3, retry_delay: float = 2.0):
    """Decorator to add retry logic and fallback mechanism to async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            operation_name = func.__name__
            retry_count = 0

            while retry_count < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Operation {operation_name} failed (attempt {retry_count}/{max_retries}): {str(e)}"
                    )
                    if retry_count >= max_retries:
                        logger.error(
                            f"Operation {operation_name} failed after {max_retries} attempts: {str(e)}\n"
                            f"Traceback: {traceback.format_exc()}"
                        )
                        # Handle fallback based on function type
                        if operation_name == "section_writer_async":
                            return {}  # Empty dict as fallback for section writer
                        elif operation_name == "section_reflection":
                            # Create fallback response using the original content
                            params = args[0] if args else kwargs.get("params", {})
                            return create_reflection_fallback(params, str(e))
                        raise  # Re-raise exception for other functions

                    await asyncio.sleep(retry_delay)  # Wait before retry

            return None  # Should not reach here

        return wrapper

    return decorator


# Augmented section writer with semaphore and retry logic
@with_retry_and_fallback(max_retries=MAX_SECTION_RETRY_NUM)
async def section_writer_with_retry(
    params: Dict[str, Any], rag_service_url: str
) -> Dict[str, Any]:
    """Enhanced section writer with retry logic and semaphore control"""
    return await semaphore_pool.with_semaphore(
        "section_writer", section_writer_async, params, rag_service_url
    )


# Add this function at the module level, before the PaperGenerationPipeline class
async def process_section_with_semaphore(
    semaphore, state, section_name, section_info, params
):
    """Process a section with semaphore control"""
    async with semaphore:
        return (
            section_name,
            await process_section(state, section_name, section_info, params),
        )


# Augmented section reflection with semaphore and retry logic
@with_retry_and_fallback(max_retries=3)
async def section_reflection_with_retry(params: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced section reflection with retry logic and semaphore control"""
    return await semaphore_pool.with_semaphore(
        "section_reflection", section_reflection, params
    )


async def process_section(
    state: PaperGenerationState, section_name: str, section_info: Dict[str, Any], params
) -> Optional[SectionData]:
    """Process a single section through writing and reflection stages"""

    parent_section = params.get("parent_section", "")
    section_task_id = params.get("section_task_id", "")
    do_section_reflection = params.get("do_section_reflection", DO_SELECT_REFLECTION)
    rag_model = params.get("section_writer_model", DEFAULT_MODEL_FOR_SECTION_WRITER)
    section_reflection_max_turns = params.get(
        "section_reflection_max_turns", SECTION_REFLECTION_MAX_TURNS
    )

    # Skip conclusion sections - they'll be handled separately
    if any(
        keyword in section_name.lower()
        for keyword in ["conclusion", "summary", "discussion"]
    ):
        logger.info(f"Skipping conclusion section: {section_name}")
        return None

    logger.info(f"Processing section: {section_name}")

    # Create section data object
    section_data = SectionData(
        section_name=section_name,
        parent_section=parent_section,
        section_index=section_info.get("section_index", 0),
        key_points=section_info.get("key_points", []),
        search_queries=section_info.get("search_queries", []),
    )
    # Validate key points and search queries
    # if section_data.is_conclustion:

    if not section_data.key_points:
        logger.info(f"Skipping section without key points: {section_name}")
        return section_data

    if len(section_data.key_points) != len(section_data.search_queries):
        logger.error(
            f"Key points and search queries length mismatch for {section_name}"
        )
        section_data.search_queries = (
            section_data.search_queries[: len(section_data.key_points)]
            if len(section_data.search_queries) > len(section_data.key_points)
            else section_data.search_queries
            + [""] * (len(section_data.key_points) - len(section_data.search_queries))
        )

    if DEBUG:
        logger.info(f"DEBUG mode: Using limited key points for section writer")
        section_data.key_points = section_data.key_points[:DEBUG_KEY_POINTS_LIMIT]
        section_data.search_queries = section_data.search_queries[
            :DEBUG_KEY_POINTS_LIMIT
        ]

    # Prepare section parameters
    section_params = {
        "section_name": section_name,
        "section_index": section_data.section_index,
        "parent_section": section_name,
        "user_query": state.user_query,
        "paper_title": state.paper_title,
        "section_key_points": section_data.key_points,
        "search_queries": section_data.search_queries,
        "sub_task_id": section_task_id,
        "rag_model": rag_model,
    }

    # Generate section content
    start_time = time.time()
    section_contents = await section_writer_with_retry(
        section_params, state.rag_service_url
    )
    logger.debug(f"process_section section_contents: {section_contents}")
    state.execution_times[f"section_writer_{section_name}"] = time.time() - start_time

    if not section_contents:
        logger.error(
            f"Failed to generate content for section {section_name} after multiple attempts"
        )
        return section_data

    # Store section contents
    section_data.content = section_contents

    # Process section reflection if enabled
    if do_section_reflection and section_reflection_max_turns > 0:
        logger.info(f"Performing reflection for section: {section_name}")
        # Prepare reflection parameters for the entire section content
        reflection_params = {
            "outline": state.outline_structure_wo_query,
            "user_query": state.user_query,
            "paper_title": state.paper_title,
            "section_name": section_name,
            "section_contents": section_contents,  # Pass entire section_contents dictionary
            "section_index": section_data.section_index,
            "rag_service_url": state.rag_service_url,
            "parent_section": parent_section,
            "section_reflection_max_turns": section_reflection_max_turns,
        }
        # Perform reflection for the entire section
        logger.info(f"[{section_name}]: Running section reflection...")
        start_time = time.time()
        reflection_result = await section_reflection_with_retry(reflection_params)
        state.execution_times[f"section_reflection_{section_name}"] = (
            time.time() - start_time
        )
        # Store reflection results
        section_data.reflection_results = reflection_result.get("section_contents", {})
    else:
        logger.info(f"Section reflection disabled, skipping ...")
        section_data.reflection_results = section_contents

    return section_data


class PaperGenerationPipeline:
    """Main pipeline class for paper generation process"""

    def __init__(
        self,
        user_name: str,
        user_query: str,
        task_id: str = "",
        output_dir: str = "temp",
        rag_service_url: str = DEFAULT_RAG_SERVICE_URL,
        **kwargs,
    ):
        """Initialize the pipeline with basic parameters"""
        self.output_dir = output_dir
        self.kwargs = kwargs

        self.do_global_reflection = self.kwargs.get(
            "do_global_reflection", DO_GLOBAL_REFLECTION
        )
        self.exist_filepath = None

        # Check for existing process file
        existing_file = find_existing_process_file(user_query, output_dir)

        if existing_file:
            logger.info(f"Found existing process file: {existing_file}")
            existing_state, self.exist_filepath = load_existing_state(existing_file)

            if existing_state and existing_state.user_query == user_query:
                logger.info("Resuming from existing state")
                self.state = existing_state
                # Update rag_service_url in case it changed
                self.state.rag_service_url = rag_service_url
                self.resume_from = determine_resume_point(
                    self.state, self.do_global_reflection
                )
            else:
                logger.info("Existing state invalid, starting fresh")
                self.state = self._create_new_state(
                    user_name, user_query, rag_service_url, task_id
                )
                self.resume_from = ProcessState.INITIALIZED
        else:
            logger.info("No existing process file found, starting fresh")
            self.state = self._create_new_state(
                user_name, user_query, rag_service_url, task_id
            )
            self.resume_from = ProcessState.INITIALIZED

        logger.info(f"Pipeline will resume from: {self.resume_from.name}")

    def _create_new_state(
        self, user_name: str, user_query: str, rag_service_url: str, task_id: str
    ) -> PaperGenerationState:
        """Create a new state object"""
        task_id = f"{user_name}_{str(uuid.uuid4())}" if not task_id else task_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return PaperGenerationState(
            task_id=task_id,
            user_name=user_name,
            user_query=user_query,
            timestamp=timestamp,
            rag_service_url=rag_service_url,
            gen_keywords=self.kwargs,
        )

    async def generate_outline(self, processed_query=None, query_intent=None) -> bool:
        """Generate paper outline"""
        query_to_use = processed_query or self.state.user_query
        logger.info(f"Generating paper outline for query: {query_to_use}")

        try:
            start_time = time.time()
            outline_result = await generate_paper_outline_async(
                query_to_use,
                query_intent=query_intent,
                max_reflections=self.kwargs.get(
                    "outline_max_reflections", OUTLINE_REFLECTION_MAX_TURNS
                ),
                max_sections=self.kwargs.get(
                    "outline_max_sections", OUTLINE_MAX_SECTIONS
                ),
                min_depth=self.kwargs.get("outline_min_depth", OUTLINE_MIN_DEPTH),
            )
            logger.debug(f"Generating paper outline result: {outline_result}")
            self.state.execution_times["outline_generation"] = time.time() - start_time

            self.state.outline_details = outline_result
            self.state.paper_title = outline_result.get(
                "title", f"Research on {self.state.user_query}"
            )
            self.state.outline_structure_w_query = outline_result.get(
                "outline_with_query", {}
            )
            self.state.outline_structure_wo_query = outline_result.get(
                "final_outline", {}
            )
            self.state.research_field = outline_result.get("research_field", "Unknown")
            self.state.query_type = outline_result.get("query_type", "Unknown")

            # Validate outline structure
            if not self.state.outline_structure_w_query:
                logger.error("Outline structure is empty or invalid")
                self.state.error_message = "Outline structure is empty or invalid"
                self.state.process_state = ProcessState.ERROR
                return False

            self.state.process_state = ProcessState.OUTLINE_GENERATED
            return True

        except Exception as e:
            logger.error(
                f"Failed to generate paper outline: {e}\n{traceback.format_exc()}"
            )
            self.state.error_message = f"Failed to generate paper outline: {str(e)}"
            self.state.process_state = ProcessState.ERROR
            return False

    async def process_sections(self, max_concurrency: int = 1) -> bool:
        """Process all sections with limited concurrency"""
        logger.info("Processing paper sections...")

        if self.state.process_state != ProcessState.OUTLINE_GENERATED:
            logger.error("Cannot process sections without outline")
            return False

        try:
            # Handle DEBUG mode - process only one section for testing
            if DEBUG:
                logger.warning(
                    "Debug mode enabled: Processing only one section for testing"
                )

                section_items = list(self.state.outline_structure_w_query.items())
                # first_section_name, first_section_info = random.choice(section_items)
                first_section_name, first_section_info = section_items[0]
                logger.info(
                    f"Debug mode: Randomly selected section '{first_section_name}' for processing"
                )

                # Check if section already exists
                if first_section_name not in self.state.sections:
                    section_data = await process_section(
                        self.state, first_section_name, first_section_info, {}
                    )
                    if section_data:
                        self.state.sections[first_section_name] = section_data
                        self._save_progress()  # Save after each section
                else:
                    logger.info(
                        f"Section '{first_section_name}' already exists, skipping..."
                    )

            else:
                # Identify sections that need to be processed
                sections_to_process = []
                for (
                    section_name,
                    section_info,
                ) in self.state.outline_structure_w_query.items():
                    if section_name not in self.state.sections:
                        sections_to_process.append((section_name, section_info))
                    else:
                        logger.info(
                            f"Section '{section_name}' already exists, skipping..."
                        )

                if not sections_to_process:
                    logger.info(
                        "All sections already processed, skipping section processing..."
                    )
                    self.state.process_state = ProcessState.SECTIONS_WRITTEN
                    return True

                logger.info(
                    f"Processing {len(sections_to_process)} remaining sections..."
                )

                # Create semaphore for limiting concurrent section processing
                semaphore = asyncio.Semaphore(max_concurrency)

                # Process sections concurrently with semaphore control
                if max_concurrency == 1:
                    # Sequential processing with progress saving after each section
                    for section_name, section_info in sections_to_process:
                        section_sub_task_id = (
                            f"{self.state.task_id}##{section_info['section_index']:04d}"
                        )
                        params = {
                            "parent_section": "",
                            "section_task_id": section_sub_task_id,
                            "do_section_reflection": DO_SELECT_REFLECTION,
                            "section_writer_model": self.kwargs.get(
                                "section_writer_model", DEFAULT_MODEL_FOR_SECTION_WRITER
                            ),
                            **self.kwargs,
                        }

                        logger.info(f"Processing section: {section_name}")
                        section_data = await process_section(
                            self.state, section_name, section_info, params
                        )

                        if section_data:
                            self.state.sections[section_name] = section_data
                            logger.info(
                                f"Section '{section_name}' completed, saving progress..."
                            )
                            self._save_progress()  # Save after each section
                        else:
                            logger.warning(f"Failed to process section: {section_name}")
                else:
                    # Concurrent processing - save after all sections complete
                    tasks = []
                    for section_name, section_info in sections_to_process:
                        section_sub_task_id = (
                            f"{self.state.task_id}##{section_info['section_index']:04d}"
                        )
                        params = {
                            "parent_section": "",
                            "section_task_id": section_sub_task_id,
                            "do_section_reflection": DO_SELECT_REFLECTION,
                            "section_writer_model": self.kwargs.get(
                                "section_writer_model", DEFAULT_MODEL_FOR_SECTION_WRITER
                            ),
                            **self.kwargs,
                        }
                        tasks.append(
                            process_section_with_semaphore(
                                semaphore,
                                self.state,
                                section_name,
                                section_info,
                                params,
                            )
                        )

                    section_results = await asyncio.gather(*tasks)

                    # Add successful results to state and save progress
                    for section_name, section_data in section_results:
                        if section_data:
                            self.state.sections[section_name] = section_data

                    # Save progress after all concurrent sections complete
                    self._save_progress()

            if not self.state.sections:
                logger.error("No sections were successfully processed")
                self.state.error_message = "No sections were successfully processed"
                self.state.process_state = ProcessState.ERROR
                return False

            self.state.process_state = ProcessState.SECTIONS_WRITTEN
            return True

        except Exception as e:
            logger.error(f"Error processing sections: {e}\n{traceback.format_exc()}")
            self.state.error_message = f"Failed to process paper sections: {str(e)}"
            self.state.process_state = ProcessState.ERROR
            return False

    async def perform_global_reflection(self, max_iterations: int = 3) -> bool:
        """Perform global reflection on all sections"""
        logger.info("Performing global reflection...")

        if self.state.process_state != ProcessState.SECTIONS_WRITTEN:
            logger.error("Cannot perform global reflection without processed sections")
            return False

        try:
            # Format sections for global reflection
            formatted_sections = format_sections_for_global_reflection(
                self.state.sections
            )
            logger.debug(f"formatted_sections: {formatted_sections}")

            # Handle DEBUG mode
            if DEBUG:
                logger.warning(
                    "Debug mode enabled: Performing global reflection on limited content"
                )

                # Limit to one section in debug mode
                first_section_name = next(iter(formatted_sections.keys()))
                formatted_sections = {
                    first_section_name: formatted_sections[first_section_name]
                }
                max_iterations = 1

            # Prepare global reflection parameters
            reflection_params = {
                "paper_title": self.state.paper_title,
                "user_query": self.state.user_query,
                "outline": self.state.outline_structure_wo_query,
                "sections_content": formatted_sections,
                "rag_service_url": self.state.rag_service_url,
                "max_iterations": max_iterations,
            }

            # Perform global reflection
            start_time = time.time()
            global_reflection_result = await global_reflection(reflection_params)

            logger.debug(f"global_reflection_result: {global_reflection_result}")

            self.state.execution_times["global_reflection"] = time.time() - start_time

            self.state.global_reflection_result = global_reflection_result
            self.state.process_state = ProcessState.GLOBAL_REFLECTION_DONE
            return True

        except Exception as e:
            logger.error(
                f"Failed during global reflection: {e}\n{traceback.format_exc()}"
            )
            self.state.error_message = (
                f"Failed during global paper reflection: {str(e)}"
            )
            self.state.process_state = ProcessState.ERROR
            return False

    async def generate_abstract_conclusion(self) -> bool:
        """Generate abstract and conclusion sections"""
        logger.info("Generating abstract and conclusion...")

        expected_state = (
            ProcessState.GLOBAL_REFLECTION_DONE
            if self.do_global_reflection
            else ProcessState.SECTIONS_WRITTEN
        )
        logger.info(f"generate_abstract_conclusion expected_state: {expected_state}")

        if self.state.process_state != expected_state:
            logger.error(
                f"Cannot generate abstract/conclusion without {'global reflection' if self.do_global_reflection else 'processed sections'}"
            )
            return False

        try:
            # Prepare abstract and conclusion parameters
            abstract_params = {
                "paper_title": self.state.paper_title,
                "user_query": self.state.user_query,
                "outline": self.state.outline_structure_wo_query,
                "sections_content": (
                    self.state.global_reflection_result["sections_content"]
                    if self.do_global_reflection
                    else format_sections_for_global_reflection(self.state.sections)
                ),
                "reflection_num": self.kwargs.get(
                    "global_abstract_conclusion_max_turns",
                    GLOBAL_ABSTRACT_CONCLUSION_MAX_TURNS,
                ),
            }

            # Generate abstract and conclusion
            start_time = time.time()
            abstract_conclusion_result = await generate_abstract_conclusion(
                abstract_params
            )
            logger.debug(f"abstract_conclusion_result: {abstract_conclusion_result}")

            self.state.execution_times["abstract_conclusion"] = time.time() - start_time
            self.state.abstract_conclusion = abstract_conclusion_result
            self.state.process_state = ProcessState.ABSTRACT_CONCLUSION_GENERATED
            return True

        except Exception as e:
            logger.error(
                f"Failed to generate abstract/conclusion: {e}\n{traceback.format_exc()}"
            )
            self.state.error_message = (
                f"Failed to generate abstract and conclusion: {str(e)}"
            )
            self.state.process_state = ProcessState.ERROR
            return False

    async def polish_paper(
        self,
    ) -> bool:
        """Polish the final paper"""
        logger.info("Polishing the final paper ...")

        if self.state.process_state != ProcessState.ABSTRACT_CONCLUSION_GENERATED:
            logger.error("Cannot polish paper without abstract and conclusion")
            return False

        try:
            # Polish the paper
            start_time = time.time()
            final_paper = await process_poolish_data(
                paper_title=self.state.paper_title,
                outline=self.state.outline_structure_wo_query["sections"],
                sections_content=(
                    self.state.global_reflection_result["sections_content"]
                    if self.do_global_reflection
                    else format_sections_for_global_reflection(self.state.sections)
                ),
                abstract_conclusion=self.state.abstract_conclusion,
            )
            logger.debug(f"final_paper: {final_paper}")

            self.state.execution_times["paper_polish"] = time.time() - start_time
            final_paper["paper_title"] = self.state.paper_title

            self.state.final_paper = final_paper

            self.state.process_state = ProcessState.PAPER_POLISHED
            return True

        except Exception as e:
            logger.error(
                f"Failed during paper polishing: {e}\n{traceback.format_exc()}"
            )
            self.state.error_message = f"Failed during paper polishing: {str(e)}"
            self.state.process_state = ProcessState.ERROR
            return False

    async def generate_paper(
        self,
    ) -> Dict[str, Any]:
        """Main function to generate a complete academic paper with resume capability"""
        logger.info(f"Starting paper generation for: {self.state.user_query}")
        logger.info(f"Resuming from state: {self.resume_from.name}")
        start_time = time.time()

        try:
            # Step 0: Process and analyze query intent (if starting fresh)
            if self.resume_from == ProcessState.INITIALIZED:
                if self.kwargs.get("do_query_understand", True):
                    try:
                        self.state.clear_step_error("outline")  # Clear previous error
                        query_analysis = await process_query_async(
                            self.state.user_query
                        )
                        processed_query = query_analysis.get(
                            "final_query", self.state.user_query
                        )
                        query_intent = query_analysis.get("intent", {})
                        self.state.rewrite_query = processed_query
                        self.state.research_field = query_intent.get(
                            "research_field", "Unknown"
                        )
                        self.state.query_type = query_intent.get(
                            "query_type", "Unknown"
                        )

                        logger.info(f"Query processed: {processed_query}")
                        logger.info(f"Query intent analysis: {query_intent}")
                    except Exception as e:
                        logger.error(f"Error analyzing query intent: {str(e)}")
                        query_analysis = {
                            "original_query": self.state.user_query,
                            "final_query": self.state.user_query,
                        }
                        processed_query = self.state.user_query
                        query_intent = {}
                else:
                    logger.info("DO NOT DO QUERY UNDERSTANDING")
                    query_analysis = {
                        "original_query": self.state.user_query,
                        "final_query": self.state.user_query,
                    }
                    processed_query = self.state.user_query
                    query_intent = {}

                self._save_progress()

                # Step 1: Generate outline
                if not await self.generate_outline(
                    processed_query=processed_query, query_intent=query_intent
                ):
                    self.state.set_step_error("outline", self.state.error_message)
                    return self._create_error_result()

            # Save progress after outline
            self._save_progress()

            if not self.kwargs.get("only_outline", False):
                # Step 2: Process sections
                if self.resume_from.value <= ProcessState.OUTLINE_GENERATED.value:
                    self.state.clear_step_error("sections")  # Clear previous error
                    if not await self.process_sections():
                        self.state.set_step_error("sections", self.state.error_message)
                        return self._create_error_result()

                    # Save progress after sections
                    self._save_progress()

                # Step 3: Global reflection
                if (
                    self.do_global_reflection
                    and self.resume_from.value <= ProcessState.SECTIONS_WRITTEN.value
                ):
                    self.state.clear_step_error(
                        "global_reflection"
                    )  # Clear previous error
                    if not await self.perform_global_reflection(
                        max_iterations=self.kwargs.get(
                            "global_reflection_max_turns", GLOBAL_REFLECTION_MAX_TURNS
                        )
                    ):

                        self.state.set_step_error(
                            "global_reflection", self.state.error_message
                        )
                        return self._create_error_result()

                    # Save progress after global reflection
                    self._save_progress()

                # Step 4: Generate abstract, conclusion and section intro
                expected_state = (
                    ProcessState.GLOBAL_REFLECTION_DONE
                    if self.do_global_reflection
                    else ProcessState.SECTIONS_WRITTEN
                )
                if self.resume_from.value <= expected_state.value:
                    self.state.clear_step_error(
                        "abstract_conclusion"
                    )  # Clear previous error
                    if not await self.generate_abstract_conclusion():
                        self.state.set_step_error(
                            "abstract_conclusion", self.state.error_message
                        )
                        return self._create_error_result()

                    # Save progress after abstract/conclusion
                    self._save_progress()

                # Step 5: Polish paper
                if (
                    self.resume_from.value
                    <= ProcessState.ABSTRACT_CONCLUSION_GENERATED.value
                ):
                    self.state.clear_step_error("polish")  # Clear previous error
                    if not await self.polish_paper():
                        self.state.set_step_error("polish", self.state.error_message)
                        return self._create_error_result()

                # Finalize and return results
                self.state.process_state = ProcessState.COMPLETED
                self.state.execution_times["total"] = time.time() - start_time

                # Save final results
                self._save_progress()

            return {
                "paper_title": self.state.paper_title,
                "user_query": self.state.user_query,
                "final_paper": self.state.final_paper,
                "outline": self.state.outline_structure_wo_query,
                "process_data": self.state.to_dict(),
            }

        except Exception as e:
            logger.error(
                f"Unexpected error in generate_paper: {e}\n{traceback.format_exc()}"
            )
            self.state.error_message = f"Unexpected error: {str(e)}"
            self.state.process_state = ProcessState.ERROR
            self._save_progress()  # Save error state
            return self._create_error_result()

    def _save_progress(self):
        """Save current progress to file"""
        try:
            if self.exist_filepath is None:
                # Generate filename based on current state
                user_query = self.state.user_query.replace(" ", "_")[:50]
                # timestamp = self.state.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.output_dir}/{user_query}_process.json"
            else:
                filename = self.exist_filepath

            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            # Save current state
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"Progress saved to {filename}")

        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def _create_error_result(self) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            "process_data": self.state.to_dict(),
            "error": self.state.error_message,
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            "task_id": self.state.task_id,
            "current_state": self.state.process_state.name,
            "paper_title": self.state.paper_title,
            "error": self.state.error_message,
            "execution_times": self.state.execution_times,
        }


def save_results(results: Dict[str, Any], output_dir: str = "temp") -> str:
    """Save the process results to files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename based on metadata
    user_query = results.get("user_query", "query").replace(" ", "_")[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename_base = f"{output_dir}/{user_query}_{timestamp}"
    filename_base = f"{output_dir}/{user_query}"

    # Save full process data
    process_file = f"{filename_base}_process.json"
    with open(process_file, "w", encoding="utf-8") as f:
        json.dump(results["process_data"], f, ensure_ascii=False, indent=2)

    # Save final paper if available
    if "final_paper" in results:
        paper_file = f"{filename_base}_paper.json"
        with open(paper_file, "w", encoding="utf-8") as f:
            json.dump(results["final_paper"], f, ensure_ascii=False, indent=2)

    if "markdown_content" in results.get("final_paper", {}):
        # Save markdown content if available
        markdown_file = f"{filename_base}_process.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(
                results.get("final_paper", {}).get(
                    "markdown_content", "# Error: Markdown content not generated"
                )
            )

    return process_file


async def main():
    """Main entry point for running the paper generation pipeline"""
    # Example parameters
    user_name = "researcher"
    task_id = "20250504"
    user_query = "Give an overview of capabilities and use case these AI agent Frameworks: LangGraph"
    user_query = "Detailed introduction to the content and implementation methods of multimodal RAG"

    query_lst = """我想了解人工智能在医疗领域的应用
The relationship between staying up late and cancer
The impact of family on children's growth
Is long-term coffee drinking good for the body
Introduce the mainstream agent design patterns to me
The detection method of large language model hallucinations and the method of reducing hallucinations
Introduce the content and implementation method of multimodal RAG""".split(
        "\n"
    )

    query_lst = query_lst[:1]
    query_lst = ["The relationship between staying up late and cancer"]
    already_query_lst = []
    for user_query in query_lst:
        if user_query in already_query_lst:
            continue
        if len(user_query)<=4:
            print(f"empty query: '{user_query}'")
            continue
        print(f"Starting paper generation for query: '{user_query}'")
        start_time = time.time()
        # Create and run the pipeline
        output_dir = "eval/surveyscope_high_parameter"
        os.makedirs(output_dir, exist_ok=True)

        kwargs = {
            "outline_max_reflections": 1,
            "outline_max_sections": 3,
            "outline_min_depth": 1,
            "section_writer_model": "Qwen3-32B",
            "do_section_reflection": False,
            "section_reflection_max_turns": 1,
            "do_global_reflection": False,
            "global_reflection_max_turns": 1,
            "global_abstract_conclusion_max_turns": 1,
        }

        pipeline = PaperGenerationPipeline(
            user_name, user_query, task_id, output_dir, **kwargs
        )
        results = await pipeline.generate_paper()

        # Calculate duration
        duration = time.time() - start_time
        minutes, seconds = divmod(duration, 60)
        print(
            f"Paper generation completed in {int(minutes)} minutes, {int(seconds)} seconds"
        )

        # Save results
        output_path = save_results(results, output_dir)
        print(f"Results saved to {output_path}")

        # Check for errors
        if "error" in results:
            print(f"Process encountered an error: {results['error']}")
        else:
            print(f"Successfully generated paper: '{results.get('paper_title', '')}'")


if __name__ == "__main__":
    asyncio.run(main())
