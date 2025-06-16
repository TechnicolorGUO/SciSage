#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Optimized section reflection implementation with complete history tracking
# ==================================================================
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Tuple, Union, TypeVar, cast
import asyncio
import uuid
import random
import hashlib
from functools import lru_cache
import json
import time
import traceback
from contextlib import asynccontextmanager

from log import logger
from models import SectionSummary, Reference
from model_factory import llm_map, chat_models, local_chat_models
from configuration import (
    global_semaphores,
    SECTION_REFLECTION_MAX_TURNS,
    SECTION_REFLECTION_MODEL_LST,
    SECTION_SUMMARY_MODEL
)
# from section_writer_opt import section_writer_async
from section_writer_opt_local import section_writer_async
from prompt_manager import (
    get_section_reflection_evaluation_system_prompts,
    get_section_reflection_evaluation_system_prompts_v2,
    get_section_reflection_eval_prompt,
    get_section_summary_prompt_template,
    get_section_summary_prompt_template_v2,
    get_section_name_refinement_prompt,
)


class SectionReflectionResult(BaseModel):
    """Model for section reflection results."""

    meets_requirements: bool = Field(description="Whether section meets requirements")
    feedback: str = Field(description="Feedback for improvement if needed")
    improvement_queries: List[str] = Field(
        default_factory=list, description="Suggested queries for regenerating content"
    )

    @validator("feedback")
    def validate_feedback(cls, v):
        """Ensure feedback is not empty."""
        if not v or not v.strip():
            return "No specific feedback provided by evaluators."
        return v


class ReflectionCacheKey:
    """Cache key for reflection results."""

    def __init__(self, section_name: str, content_hash: str):
        self.section_name = section_name
        self.content_hash = content_hash

    def __hash__(self):
        return hash((self.section_name, self.content_hash))

    def __eq__(self, other):
        if not isinstance(other, ReflectionCacheKey):
            return False
        return (
            self.section_name == other.section_name
            and self.content_hash == other.content_hash
        )


class SectionReflectionState(BaseModel):
    """State container for section reflection workflow."""

    section_name: str = Field(description="Name of the section")
    parent_section: Optional[str] = Field(
        default=None, description="Parent section name"
    )
    user_query: str = Field(description="Original user query")
    section_content: Dict[str, Any] = Field(description="Section content to reflect on")
    paper_title: str = Field(description="Paper title")
    outline: Dict[str, Any] = Field(description="Paper outline")
    max_reflections: int = Field(default=3, description="Maximum reflection cycles")
    summary: Optional[SectionSummary] = Field(
        default=None, description="Generated section summary"
    )


class EvaluationDetail(BaseModel):
    """Model for storing detailed evaluation information."""

    model: str = Field(description="Model used for evaluation")
    meets_requirements: bool = Field(description="Whether requirements are met")
    feedback: str = Field(description="Detailed feedback")
    improvement_queries: List[str] = Field(
        default_factory=list, description="Suggested improvement queries"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class SummaryDetail(BaseModel):
    """Model for storing summary generation details."""

    model_used: str = Field(description="Model used for summary")
    raw_summary: str = Field(description="Raw summary text")
    final_sentences: List[str] = Field(description="Processed summary sentences")
    error: Optional[str] = Field(default=None, description="Error message if any")


def content_hash(section_content: Dict[str, Any]) -> str:
    """Generate a hash of section content for caching and comparison."""
    content_str = json.dumps(section_content, sort_keys=True)
    return hashlib.md5(content_str.encode()).hexdigest()


@lru_cache(maxsize=32)
def get_evaluation_models(seed: int = None) -> List[str]:
    """Get a consistent set of evaluation models based on seed.

    Args:
        seed: Random seed for reproducible model selection

    Returns:
        List of model names to use for evaluation
    """
    if seed is not None:
        random.seed(seed)

    # Use local models preferentially - they're faster and cheaper
    # available_models = list(local_chat_models.keys())
    available_models = SECTION_REFLECTION_MODEL_LST

    # Fallback to any models if no local models are available
    if not available_models:
        available_models = list(chat_models.keys())
        logger.warning("No local models available, using cloud models instead")

    # Always use at least one model, but prefer 2-3 for consensus
    num_models = min(3, max(1, len(available_models)))
    return random.sample(available_models, num_models)


def get_section_info_from_outline(
    section_name: str, outline: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract section information from outline.

    Args:
        section_name: Name of the section to find
        outline: Complete paper outline

    Returns:
        Section information dictionary or empty dict if not found
    """
    # Check for direct match in sections
    sections = outline.get("sections", {})

    # Handle both list and dict formats for sections
    if isinstance(sections, dict):
        for section in sections.values():
            if section.get("name") == section_name:
                return section
    elif isinstance(sections, list):
        for section in sections:
            if section.get("name") == section_name:
                return section

    # Check for deeper nested sections
    for section in sections.values() if isinstance(sections, dict) else sections:
        subsections = section.get("subsections", [])
        for subsection in subsections:
            if subsection.get("name") == section_name:
                return subsection

    logger.warning(f"Section {section_name} not found in outline")
    return {}


@asynccontextmanager
async def evaluation_timer(section_name: str, model_name: str):
    """Context manager to track evaluation timing.

    Args:
        section_name: Name of the section being evaluated
        model_name: Name of the model being used
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.debug(
            f"Evaluation of {section_name} with {model_name} took {elapsed:.2f}s"
        )


async def evaluate_section(
    section_name: str,
    section_content: Dict[str, Any],
    paper_title: str,
    user_query: str,
    parent_section: Optional[str],
    outline: Dict[str, Any],
    rag_service_url: str,
) -> Tuple[str, SectionReflectionResult, List[EvaluationDetail]]:
    """Evaluate a single section using multiple LLMs and prompts in parallel.

    Args:
        section_name: Name of the section to evaluate
        section_content: Content of the section
        paper_title: Title of the paper
        user_query: Original user query
        parent_section: Parent section name if any
        outline: Complete paper outline
        rag_service_url: URL for RAG service

    Returns:
        Tuple of (section_name, merged_result, evaluation_details)
    """
    parser = PydanticOutputParser(pydantic_object=SectionReflectionResult)

    # Get models and prompts - with reproducible selection
    section_hash = int(hashlib.md5(section_name.encode()).hexdigest(), 16) % 1000
    models = get_evaluation_models(section_hash)
    prompts = get_section_reflection_evaluation_system_prompts_v2()

    # Get section key points from outline
    # section_info = get_section_info_from_outline(section_name, outline)
    section_key_point = section_content.get("section_key_point", "")

    # Store detailed model evaluations for tracking
    evaluation_details: List[EvaluationDetail] = []

    async def run_evaluation(
        model_name: str, prompt_text: str
    ) -> Tuple[SectionReflectionResult, EvaluationDetail]:
        """Run a single evaluation with specified model and prompt."""
        try:
            async with evaluation_timer(section_name, model_name):
                # Avoid creating a new LLM for each evaluation
                llm = llm_map.get(model_name)
                if not llm:
                    logger.warning(
                        f"Model {model_name} not found, falling back to gpt-4"
                    )
                    model_name = "gpt-4"
                    llm = llm_map.get("gpt-4")

                    # Ultimate fallback to first available model
                    if not llm:
                        available_models = list(llm_map.keys())
                        if not available_models:
                            raise ValueError("No models available for evaluation")
                        model_name = available_models[0]
                        llm = llm_map[model_name]
                        logger.warning(f"Falling back to {model_name} as last resort")

                llm = llm.with_config(
                    {"temperature": 0.7}
                )  # Use consistent temperature

                prompt = get_section_reflection_eval_prompt(
                    prompt_text,
                    paper_title,
                    user_query,
                    section_name,
                    parent_section,
                    section_key_point,
                    section_content,
                    parser.get_format_instructions(),
                )

                chain = prompt | llm | parser
                result = await chain.ainvoke({})
                logger.info(f"Completed evaluation with model {model_name}")

                # Create detailed evaluation record
                evaluation_detail = EvaluationDetail(
                    model=model_name,
                    meets_requirements=result.meets_requirements,
                    feedback=result.feedback,
                    improvement_queries=result.improvement_queries,
                )

                return result, evaluation_detail
        except Exception as e:
            logger.error(f"Error in evaluation with model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())

            # Return a default result in case of failure
            default_result = SectionReflectionResult(
                meets_requirements=False,
                feedback=f"Evaluation failed with model {model_name}: {str(e)}",
                improvement_queries=[],
            )

            error_detail = EvaluationDetail(
                model=model_name,
                meets_requirements=False,
                feedback=default_result.feedback,
                improvement_queries=default_result.improvement_queries,
                error=str(e),
            )

            return default_result, error_detail

    # Create evaluation tasks for each model-prompt combination
    evaluation_tasks = []
    for i, model in enumerate(models):
        # Use a different prompt for each model to get diverse perspectives
        prompt = prompts[i % len(prompts)]
        evaluation_tasks.append(run_evaluation(model, prompt))

    # Execute all evaluations concurrently with timeout protection
    try:
        # Use gather with return_exceptions to avoid failing all evaluations
        # if one times out or errors
        eval_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Filter out exceptions and convert to proper result format
        filtered_results = []
        for i, result in enumerate(eval_results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation task {i} failed: {str(result)}")
                model = models[i % len(models)]

                # Create fallback result
                default_result = SectionReflectionResult(
                    meets_requirements=False,
                    feedback=f"Evaluation failed: {str(result)}",
                    improvement_queries=["Try with more comprehensive search query"],
                )

                error_detail = EvaluationDetail(
                    model=model,
                    meets_requirements=False,
                    feedback=default_result.feedback,
                    improvement_queries=default_result.improvement_queries,
                    error=str(result),
                )

                filtered_results.append((default_result, error_detail))
            else:
                filtered_results.append(result)

        eval_results = filtered_results
    except Exception as e:
        # If gather itself fails, provide a single fallback evaluation
        logger.error(f"All evaluations failed: {str(e)}")
        logger.error(traceback.format_exc())

        default_result = SectionReflectionResult(
            meets_requirements=False,
            feedback=f"All evaluations failed: {str(e)}",
            improvement_queries=[],
        )

        error_detail = EvaluationDetail(
            model="fallback",
            meets_requirements=False,
            feedback=default_result.feedback,
            improvement_queries=default_result.improvement_queries,
            error=str(e),
        )

        eval_results = [(default_result, error_detail)]

    # Separate results and details
    results = [r[0] for r in eval_results]
    evaluation_details = [r[1] for r in eval_results]

    # Analyze results - use majority vote for requirements
    meets_requirements_count = sum(1 for r in results if r.meets_requirements)
    meets_requirements_majority = meets_requirements_count > len(results) / 2

    # If only one evaluation and it failed, be conservative and say needs improvement
    if len(results) == 1 and isinstance(eval_results[0][1].error, str):
        meets_requirements_majority = False

    # Merge feedback and queries
    merged_feedback = merge_feedback(results, meets_requirements_majority)
    selected_queries = select_top_queries(results, max_queries=3)

    # Create the merged result
    merged_result = SectionReflectionResult(
        meets_requirements=meets_requirements_majority,
        feedback=merged_feedback,
        improvement_queries=selected_queries,
    )

    logger.info(
        f"Section evaluation complete for {section_name}. "
        + f"Meets requirements: {meets_requirements_majority}"
    )

    return section_name, merged_result, evaluation_details


def merge_feedback(
    results: List[SectionReflectionResult], meets_requirements: bool
) -> str:
    """Merge feedback from multiple evaluations.

    Args:
        results: List of evaluation results
        meets_requirements: Whether the section meets requirements overall

    Returns:
        Consolidated feedback string
    """
    all_feedback = [r.feedback for r in results if r.feedback and r.feedback.strip()]

    if not all_feedback:
        return "No specific feedback provided by evaluators."

    if meets_requirements:
        return (
            "OVERALL ASSESSMENT: Section meets requirements. "
            + "Key strengths noted across evaluations: "
            + "; ".join(all_feedback[:2])
        )
    else:
        return (
            "OVERALL ASSESSMENT: Section needs improvement. "
            + "Key issues identified: "
            + "; ".join(all_feedback)
        )


def select_top_queries(
    results: List[SectionReflectionResult], max_queries: int = 3
) -> List[str]:
    """Select top queries from all evaluations.

    Args:
        results: List of evaluation results
        max_queries: Maximum number of queries to return

    Returns:
        List of unique, prioritized queries
    """
    # Collect all queries
    all_queries = []
    for result in results:
        all_queries.extend(result.improvement_queries)

    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for query in all_queries:
        if query and query.strip() and query not in seen:
            seen.add(query)
            unique_queries.append(query)

    return unique_queries[:max_queries]


async def generate_section_summary(
    section_name: str, section_content: Dict[str, Any], paper_title: str
) -> Tuple[List[str], SummaryDetail]:
    """Generate a concise summary for a section.

    Args:
        section_name: Name of the section
        section_content: Content of the section
        paper_title: Title of the paper

    Returns:
        Tuple of (summary_sentences, summary_details)
    """
    summary_details = SummaryDetail(model_used="", raw_summary="", final_sentences=[])

    logger.info(f"Generating summary for section: {section_name}")

    try:
        # Use a consistently good model for summaries
        selected_model = "gpt-4-32k"  # Always use this model for summaries
        selected_model = SECTION_SUMMARY_MODEL  # Always use this model for summaries

        llm = llm_map.get(selected_model)
        if not llm:
            logger.warning(
                f"Summary model {selected_model} not found, falling back to gpt-4"
            )
            selected_model = "gpt-4"
            llm = llm_map.get(selected_model)

            # Ultimate fallback to any available model
            if not llm:
                available_models = list(llm_map.keys())
                if not available_models:
                    raise ValueError("No models available for summary generation")
                selected_model = available_models[0]
                llm = llm_map[selected_model]
                logger.warning(
                    f"Falling back to {selected_model} for summary generation"
                )

        summary_details.model_used = selected_model
        llm = llm.with_config({"temperature": 0.3})  # Lower temp for summaries

        prompt = get_section_summary_prompt_template_v2(
            paper_title, section_name, section_content.get("section_text", "")
        )

        chain = prompt | llm
        result = await chain.ainvoke({})
        summary_raw = result.content.strip()
        summary_details.raw_summary = summary_raw

        # Process summary into sentences
        summary_sentences = process_summary_sentences(summary_raw)
        summary_details.final_sentences = summary_sentences

        return summary_sentences, summary_details

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        logger.error(traceback.format_exc())

        error_message = [
            "Summary generation failed. Please review the section manually."
        ]
        summary_details.error = str(e)
        summary_details.final_sentences = error_message
        return error_message, summary_details


def process_summary_sentences(summary_raw: str) -> List[str]:
    """Process raw summary text into proper sentences.

    Args:
        summary_raw: Raw summary text from LLM

    Returns:
        List of processed summary sentences
    """
    # Split by line breaks first
    summary_sentences = [s.strip() for s in summary_raw.split("\n") if s.strip()]

    if not summary_sentences:
        # Fallback to sentence splitting if no line breaks
        import re

        summary_sentences = [
            s.strip() + "." for s in re.split(r"[.!?]+", summary_raw) if s.strip()
        ]

    # Ensure each sentence ends with proper punctuation
    processed_sentences = []
    for sentence in summary_sentences[:3]:  # Limit to 3 sentences
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        processed_sentences.append(sentence)

    # If still empty, provide a generic placeholder
    if not processed_sentences:
        processed_sentences = [
            "This section contains information relevant to the paper topic."
        ]

    return processed_sentences


async def regenerate_section_content(
    params: Dict[str, Any], rag_service_url: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Regenerate section content based on feedback.

    Args:
        params: Parameters for regeneration
        rag_service_url: URL for RAG service

    Returns:
        Tuple of (regenerated_content, regeneration_details)
    """
    logger.info(
        f"Regenerating content for section: {params.get('section_name', 'unknown')}"
    )

    regeneration_details = {
        "search_queries_used": params.get("search_queries", []),
        "section_key_points": params.get("section_key_points", []),
        "timestamp": str(uuid.uuid4()),  # Use as unique identifier for tracking
        "success": False,
    }

    try:
        # Prepare search queries
        search_queries = params.get("search_queries", [])
        section_key_points = params.get("section_key_points", [])

        if not search_queries:
            # Generate default queries from section key points
            search_queries = [
                f"{params['section_name']}: {point}" for point in section_key_points
            ]
            regeneration_details["search_queries_generated"] = True

        # Ensure we have at least one query for each key point
        if len(search_queries) < len(section_key_points):
            # Duplicate queries if needed
            search_queries = (
                search_queries * ((len(section_key_points) // len(search_queries)) + 1)
            )[: len(section_key_points)]
        elif len(search_queries) > len(section_key_points):
            # Sample queries if we have more than needed
            search_queries_used = random.sample(search_queries, len(section_key_points))
            regeneration_details["search_queries_used"] = search_queries_used
            search_queries = search_queries_used

        # Prepare section_writer_async parameters
        writer_params = {
            "section_name": params["section_name"],
            "section_index": params.get("section_index", 0),
            "parent_section": params.get("parent_section"),
            "user_query": params["user_query"],
            "section_key_points": section_key_points,
            "paper_title": params["paper_title"],
            "search_queries": search_queries,
        }

        # Add feedback for context if available
        if "feedback" in params:
            writer_params["feedback"] = params["feedback"]

        # Call section_writer_async
        result = await section_writer_async(writer_params, rag_service_url)

        logger.info(
            f"Section writer completed for {params.get('section_name', 'unknown')}"
        )
        regeneration_details["success"] = True
        regeneration_details["writer_result"] = {
            "status": "success",
            "content_length": len(str(result)),
            "sub_task_id": params.get("sub_task_id", str(uuid.uuid4())),
        }

        return result, regeneration_details

    except Exception as e:
        logger.error(f"Error regenerating content: {str(e)}")
        logger.error(traceback.format_exc())
        regeneration_details["error"] = str(e)
        regeneration_details["traceback"] = traceback.format_exc()

        # Return empty dict and error details
        return {}, regeneration_details


async def section_reflection_async(
    params: Dict[str, Any],
    rag_service_url: str,
    max_iterations: int = SECTION_REFLECTION_MAX_TURNS,
) -> Tuple[str, Dict[str, Any]]:
    """Asynchronous implementation of section reflection with optimized iteration logic.

    Args:
        params: Parameters for section reflection
        rag_service_url: URL for RAG service
        max_iterations: Maximum number of reflection iterations

    Returns:
        Tuple of (section_name, result_dict)
    """
    # Extract parameters with defaults
    section_name = params.get("section_name", "")
    parent_section = params.get("parent_section")
    user_query = params.get("user_query", "")
    paper_title = params.get("paper_title", "")
    outline = params.get("outline", {})
    section_content = params.get("section_content", {})
    section_index = params.get("section_index")

    # Initialize comprehensive tracking
    section_history = []
    reflection_records = []
    evaluation_history = []
    regeneration_history = []
    meets_requirements = False
    iteration = 0

    # Add metadata timestamps
    start_time = time.time()
    task_id = str(uuid.uuid4())

    # Store initial content
    section_history.append(
        {
            "iteration": 0,
            "content": section_content,
            "timestamp": str(uuid.uuid4()),
            "content_hash": content_hash(section_content),
        }
    )

    logger.info(f"Starting reflection for section: {section_name} (task: {task_id})")

    # Main iteration loop
    while not meets_requirements and iteration < max_iterations:
        iteration_start = time.time()
        logger.info(f"Reflection iteration {iteration+1} for section: {section_name}")

        try:
            # Evaluate current section
            _, reflection_result, eval_details = await evaluate_section(
                section_name=section_name,
                section_content=section_content,
                paper_title=paper_title,
                user_query=user_query,
                parent_section=parent_section,
                outline=outline,
                rag_service_url=rag_service_url,
            )

            meets_requirements = reflection_result.meets_requirements

            # Save evaluation details
            evaluation_entry = {
                "iteration": iteration + 1,
                "timestamp": str(uuid.uuid4()),
                "model_evaluations": [detail.dict() for detail in eval_details],
                "merged_result": {
                    "meets_requirements": reflection_result.meets_requirements,
                    "feedback": reflection_result.feedback,
                    "improvement_queries": reflection_result.improvement_queries,
                },
                "duration_seconds": time.time() - iteration_start,
            }
            evaluation_history.append(evaluation_entry)

            # Record this iteration
            reflection_record = {
                "iteration": iteration + 1,
                "meets_requirements": meets_requirements,
                "feedback": reflection_result.feedback,
                "improvement_queries": (
                    reflection_result.improvement_queries
                    if not meets_requirements
                    else []
                ),
                "evaluation_details": [detail.dict() for detail in eval_details],
            }
            reflection_records.append(reflection_record)

            if meets_requirements:
                logger.info(f"Section {section_name} meets requirements")
                break

            # Section needs improvement - prepare for regeneration
            logger.info(
                f"Section [{section_name}] needs improvement: {reflection_result.feedback[:100]}..."
            )

            # Prepare parameters for section regeneration
            section_key_points = section_content.get("section_key_point", "")
            if not section_key_points:
                logger.warning(
                    f"No section key point found for {section_name}, using section name"
                )
                section_key_points = section_name

            regeneration_params = {
                "section_name": section_name,
                "parent_section": parent_section,
                "user_query": user_query,
                "section_key_points": (
                    [section_key_points]
                    if isinstance(section_key_points, str)
                    else section_key_points
                ),
                "paper_title": paper_title,
                "search_queries": (
                    reflection_result.improvement_queries[:1]
                    if reflection_result.improvement_queries
                    else []
                ),
                "sub_task_id": str(uuid.uuid4()),
                "section_index": section_index,
                "feedback": reflection_result.feedback,  # Include feedback for context
            }

            # Regenerate content
            new_section_content, regen_details = await regenerate_section_content(
                regeneration_params, rag_service_url
            )

            # Track regeneration details
            regeneration_entry = {
                "iteration": iteration + 1,
                "timestamp": str(uuid.uuid4()),
                "params": regeneration_params,
                "details": regen_details,
                "success": bool(new_section_content),  # Check if content was generated
                "duration_seconds": time.time() - iteration_start,
            }
            regeneration_history.append(regeneration_entry)

            # Update section content for next iteration if successful
            if new_section_content:
                # Handle different content formats
                if (
                    isinstance(section_key_points, str)
                    and section_key_points in new_section_content
                ):
                    section_content = new_section_content[section_key_points]
                    logger.info(
                        f"Updated section content by key point: {section_key_points}"
                    )
                elif section_name in new_section_content:
                    section_content = new_section_content[section_name]
                    logger.info(
                        f"Updated section content by section name: {section_name}"
                    )
                elif len(new_section_content) == 1:
                    # If there's only one key, use that content regardless of name
                    first_key = next(iter(new_section_content))
                    section_content = new_section_content[first_key]
                    logger.info(
                        f"Updated section content using first available key: {first_key}"
                    )
                else:
                    logger.warning(
                        f"Could not locate content for section [{section_name}] in regenerated content. Keys: {list(new_section_content.keys())}"
                    )

                # Track section content history
                section_history.append(
                    {
                        "iteration": iteration + 1,
                        "content": section_content,
                        "timestamp": str(uuid.uuid4()),
                        "content_hash": content_hash(section_content),
                    }
                )

                # Check for duplicate content (no change from last iteration)
                if (
                    len(section_history) >= 2
                    and section_history[-1]["content_hash"]
                    == section_history[-2]["content_hash"]
                ):
                    logger.warning(
                        f"No content change detected for section {section_name} after regeneration"
                    )

                    # Force termination if we're getting the same content back
                    if iteration > 0:  # Only do this after at least one iteration
                        logger.info(
                            f"Terminating reflection for section {section_name} due to content duplication"
                        )
                        meets_requirements = True

                        # Add a note to the feedback
                        reflection_records[-1][
                            "feedback"
                        ] += " (Reflection terminated due to content duplication)"
            else:
                logger.warning(
                    f"Failed to regenerate content for [{section_name}], using previous version"
                )

        except Exception as e:
            logger.error(f"Error in reflection iteration {iteration+1}: {str(e)}")
            logger.error(traceback.format_exc())

            # Add error to reflection records
            error_record = {
                "iteration": iteration + 1,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "meets_requirements": False,
                "timestamp": str(uuid.uuid4()),
                "duration_seconds": time.time() - iteration_start,
            }
            reflection_records.append(error_record)
            break

        iteration += 1

    # Generate summary regardless of reflection outcome
    summary_text, summary_details = await generate_section_summary(
        section_name=section_name,
        section_content=section_content,
        paper_title=paper_title,
    )

    # Convert summary details to dict
    summary_details_dict = (
        summary_details.dict()
        if isinstance(summary_details, BaseModel)
        else summary_details
    )

    # Prepare comprehensive final result
    if meets_requirements:
        reflection = {
            "reflection_records": reflection_records,
            "feedback": None,
            "meets_requirements": True,
            "evaluation_history": evaluation_history,
            "regeneration_history": regeneration_history,
            "section_history": section_history,
            "summary_details": summary_details_dict,
        }
    else:
        # Get the latest feedback or error message
        latest_feedback = "Unable to meet requirements"
        if reflection_records:
            latest_record = reflection_records[-1]
            if "feedback" in latest_record:
                latest_feedback = latest_record["feedback"]
            elif "error" in latest_record:
                latest_feedback = f"Error: {latest_record['error']}"

        reflection = {
            "feedback": latest_feedback,
            "meets_requirements": False,
            "reflection_records": reflection_records,
            "max_iterations_reached": iteration >= max_iterations,
            "evaluation_history": evaluation_history,
            "regeneration_history": regeneration_history,
            "section_history": section_history,
            "summary_details": summary_details_dict,
        }

    # Add section summary to content
    section_content["section_summary"] = summary_text

    # Add performance metrics
    total_duration = time.time() - start_time

    # Final comprehensive result
    result = {
        "section_name": section_name,
        "section_content": section_content,
        "reflection": reflection,
        # Add section metadata for easy reference
        "section_metadata": {
            "section_name": section_name,
            "parent_section": parent_section,
            "section_index": section_index,
            "section_key_point": section_content.get("section_key_point", ""),
            "iterations_performed": iteration,
            "meets_requirements": meets_requirements,
            "task_id": task_id,
        },
        # Include full history for complete tracking
        "process_history": {
            "initial_content": section_history[0]["content"] if section_history else {},
            "final_content": section_content,
            "evaluations": evaluation_history,
            "regenerations": regeneration_history,
            "content_versions": section_history,
        },
        # Performance metrics
        "performance": {
            "total_duration_seconds": total_duration,
            "start_time": start_time,
            "end_time": time.time(),
            "iterations": iteration,
        },
    }

    return section_name, result


async def section_reflection(params: Dict[str, Any]) -> Dict[str, Any]:
    """Async implementation of section reflection with complete process tracking.

    Args:
        params: Parameters for section reflection

    Returns:
        Dictionary containing reflection results
    """
    # Start timer and logging
    ts = time.time()
    logger.info(
        f"Running section_reflection for {params.get('section_name', 'unknown')}"
    )

    # Extract parameters with defaults
    paper_title = params.get("paper_title", "")
    user_query = params.get("user_query", "")
    rag_service_url = params.get("rag_service_url", "http://120.92.91.62:9528/chat")
    section_name = params.get("section_name", "")
    section_reflection_max_turns = params.get("section_reflection_max_turns",1)

    try:
        # Get section_contents - the entire dictionary of key_points and their content
        section_contents = params.get("section_contents", {})

        # Create a dictionary to store reflection results for all key points
        reflection_results = {}
        tasks = []
        # Process each key point in the section contents
        for key_point, content in section_contents.items():
            # Prepare section content structure for this key point
            section_content = {
                "section_key_point": key_point,
                "section_text": content.get("section_text", ""),
                "main_figure_data": content.get("main_figure_data", ""),
                "main_figure_caption": content.get("main_figure_caption", ""),
            }

            # Add report index list if available
            if "reportIndexList" in content:
                section_content["reportIndexList"] = content["reportIndexList"]

            # Prepare section parameters for this key point
            section_params = {
                "section_name": key_point,  # Use key point as section name for reflection
                "parent_section": params.get("parent_section", None),
                "user_query": user_query,
                "paper_title": paper_title,
                "section_content": section_content,
                "section_index": content.get("section_index", 0),
                "outline": params.get("outline", {}),
                "max_iterations": section_reflection_max_turns,
            }

            # Run the async reflection function for this key point
            logger.info(f"Reflection starting for key point: {key_point}")

            async def process_key_point(kp, params):
                async with global_semaphores.section_reflection_semaphore:
                    logger.info(f"Reflection starting for key point: {kp}")
                    return await section_reflection_async(params, rag_service_url)

            # 添加到任务列表
            tasks.append(process_key_point(key_point, section_params))

        # 并行执行所有任务，但受信号量控制
        results = await asyncio.gather(*tasks)

        # 处理结果
        for i, (key_point, _) in enumerate(section_contents.items()):
            key_name, key_reflection_result = results[i]
            reflection_results[key_point] = key_reflection_result.get(
                "section_content", {}
            )

        # Calculate execution time
        te = time.time()
        execution_time = te - ts

        # Prepare the final result with all reflection results
        final_result = {
            "section_name": section_name,
            "section_contents": reflection_results,
            "reflection_performance": {
                "execution_time_seconds": execution_time,
                "start_timestamp": ts,
                "end_timestamp": te,
            },
        }

        logger.info(
            f"Section [{section_name}] reflection completed in {execution_time:.2f}s"
        )
        return final_result

    except Exception as e:
        # Handle any unhandled exceptions at the top level
        te = time.time()
        logger.error(f"Critical error in section_reflection: {str(e)}")
        logger.error(traceback.format_exc())

        # Return error information in a structured format
        return {
            "section_name": section_name,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed",
            "reflection_performance": {
                "execution_time_seconds": te - ts,
                "start_timestamp": ts,
                "end_timestamp": te,
            },
        }


def build_section_reflection_workflow():
    """
    Build and compile a workflow for section reflection and improvement.

    This function creates a workflow that:
    1. Evaluates a section's content quality and relevance
    2. Improves the section through iterative refinement if needed
    3. Generates a concise summary of the final section

    Returns:
        A compiled StateGraph workflow for section reflection
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(SectionReflectionState)

    # Define the reflection node that processes the section
    async def section_reflection_node(state):
        section_name = state.section_name
        section_content = state.section_content

        # Get default RAG service URL or use a configured one
        rag_service_url = "http://120.92.91.62:9528/chat"

        params = {
            "section_name": section_name,
            "parent_section": state.parent_section,
            "user_query": state.user_query,
            "section_content": section_content,
            "paper_title": state.paper_title,
            "outline": state.outline,
            "section_key_point": section_content.get("section_key_point", ""),
            "section_text": section_content.get("section_text", ""),
            "main_figure_data": section_content.get("main_figure_data", ""),
            "main_figure_caption": section_content.get("main_figure_caption", ""),
            "reportIndexList": section_content.get("reportIndexList", []),
            "rag_service_url": rag_service_url,
        }

        try:
            # Call the main reflection function
            result = await section_reflection(params)

            if "error" in result:
                # Handle error case
                logger.error(f"Error in section_reflection_node: {result['error']}")
                return state.model_dump()  # Return unchanged state on error

            # Update state with reflection results
            new_state = SectionReflectionState(
                section_name=state.section_name,
                parent_section=state.parent_section,
                user_query=state.user_query,
                section_content=result["section_content"],
                paper_title=state.paper_title,
                outline=state.outline,
                max_reflections=state.max_reflections,
                summary=(
                    SectionSummary(
                        section_name=state.section_name,
                        summary=result["section_content"].get("section_summary", []),
                        parent_section=state.parent_section,
                    )
                    if "section_summary" in result["section_content"]
                    else None
                ),
            )

            return new_state.model_dump()

        except Exception as e:
            logger.error(f"Error in section_reflection_node: {str(e)}")
            logger.error(traceback.format_exc())
            return state.model_dump()  # Return unchanged state on error

    # Add the node and define workflow
    workflow.add_node("reflect_section", section_reflection_node)
    workflow.add_edge("reflect_section", END)
    workflow.set_entry_point("reflect_section")

    # Compile and return
    return workflow.compile()


# Test functions
async def test_section_reflection_async():
    """Test the asynchronous section_reflection_async function."""
    from example import example_section_reflection_inp

    test_params = example_section_reflection_inp

    try:
        section_name, reflection_results = await section_reflection_async(
            test_params, test_params["rag_service_url"]
        )
        print("Reflection Results (Async):")
        print(json.dumps(reflection_results, indent=2, ensure_ascii=False))

        # Save results to file
        with open("./temp/test_section_reflection_async.json", "w") as f:
            json.dump(reflection_results, f, indent=2, ensure_ascii=False)
        print("Results saved to ./temp/test_section_reflection_async.json")

    except Exception as e:
        print(f"Error during asynchronous reflection test: {e}")
        print(traceback.format_exc())


def test_section_reflection():
    """Test the synchronous section_reflection function."""
    from example import example_section_reflection_inp

    test_params = example_section_reflection_inp

    try:
        reflection_results = asyncio.run(section_reflection(test_params))
        print("Reflection Results (Sync):")
        print(json.dumps(reflection_results, indent=2, ensure_ascii=False))

        # Save results to file
        dest_file = "./temp/test_section_reflection.json"
        with open(dest_file, "w") as f:
            json.dump(reflection_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {dest_file}")

    except Exception as e:
        print(f"Error during synchronous reflection test: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    print("Running section reflection test...")
    # Uncomment to test either function
    # test_section_reflection()
    # asyncio.run(test_section_reflection_async())
