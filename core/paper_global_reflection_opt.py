# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Academic paper global reflection system
# ==================================================================

from functools import lru_cache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from log import logger
from model_factory import llm_map
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional, Tuple, Union, TypedDict, cast
import asyncio
import json
import random
import re
import time
import traceback
from utils import prepare_sections_data
import uuid

from configuration import DEFAULT_MODEL_FOR_GLOBAL_REFLECTION, GLOBAL_REFLECTION_MAX_TURNS

from prompt_manager import (
    get_global_reflection_eval_system,
    get_global_reflection_eval_system_v2,
    get_global_reflection_eval_paper_prompt,
    get_enhanced_search_query_prompt,
    get_enhanced_search_query_prompt_v2,
    get_issue_analysis_prompt,
    get_new_key_point_generation_prompt,
)
from section_reflection_opt import generate_section_summary

# from section_writer_opt import section_writer_async
from section_writer_opt_local import section_writer_async


# Type definitions for improved code clarity
class ImprovementAction(TypedDict):
    section: str
    issues: List[str]
    rewrite: bool


class GlobalReflectionResult(BaseModel):
    """Model for global paper reflection results."""

    meets_requirements: bool = Field(description="Whether paper meets academic requirements")
    feedback: str = Field(description="Feedback for improvement if needed")
    improvement_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Suggested actions for improving specific sections",
    )


class SectionRecommendation(TypedDict):
    section: str
    issues: List[str]
    rewrite_votes: int
    total_votes: int


class ModelEvaluationDetail(TypedDict):
    model: str
    meets_requirements: bool
    feedback: str
    improvement_actions: List[Dict[str, Any]]
    raw_output: str
    parse_error: Optional[str]
    error: Optional[str]
    traceback: Optional[str]


class EvaluationHistory(TypedDict):
    iteration: int
    timestamp: str
    model_evaluations: List[ModelEvaluationDetail]
    merged_result: Dict[str, Any]


class RewriteResult(TypedDict):
    section_name: str
    section_index: Optional[int]
    parent_section: Optional[str]
    rewrite_successful: bool
    section_content: Optional[List[Dict[str, Any]]]
    enhanced_search_queries: Optional[List[str]]
    improvement_issues: Optional[List[str]]
    error: Optional[str]
    traceback: Optional[str]


class RewriteHistory(TypedDict):
    iteration: int
    timestamp: str
    sections_rewritten: int
    rewrite_details: List[RewriteResult]


class SectionToRewrite(TypedDict):
    section_name: str
    section_info: Union[List[Dict[str, Any]], Dict[str, Any]]
    issues: List[str]


class IssueAnalysisResult(BaseModel):
    issue_mapping: Dict[str, List[str]] = Field(description="Mapping from existing key points to the list of issues relevant to them.")
    issues_for_new_keypoints: List[str] = Field(description="List of issues that do not strongly relate to any existing key point and require new ones.")


@lru_cache(maxsize=32)
def get_evaluation_models(seed: Optional[int] = None) -> List[str]:
    """Get a consistent set of evaluation models based on seed.

    Args:
        seed: Optional seed value for reproducible random selection

    Returns:
        List of selected model names
    """
    # if seed is not None:
    #     random.seed(seed)

    # available_models = [
    #     "gpt-4",
    #     "Qwen25-72B",

    # ]

    # # Ensure we don't try to sample more models than available
    # sample_count = min(3, len(available_models))
    # return random.sample(available_models, sample_count)
    return DEFAULT_MODEL_FOR_GLOBAL_REFLECTION


async def run_model_evaluation(
    model_name: str,
    prompt_text: str,
    paper_title: str,
    user_query: str,
    outline: Dict[str, Any],
    sections_data: List[Dict[str, Any]],
    parser: PydanticOutputParser,
) -> Tuple[GlobalReflectionResult, ModelEvaluationDetail]:
    """Run evaluation with one model and prompt combination.

    Args:
        model_name: Name of the LLM to use
        prompt_text: System prompt for the evaluation
        paper_title: Title of the paper being evaluated
        user_query: Original user query
        outline: Paper outline structure
        sections_data: Content of paper sections
        parser: Output parser for structured results

    Returns:
        Tuple of (evaluation result, detailed information)
    """
    evaluation_detail: ModelEvaluationDetail = {
        "model": model_name,
        "meets_requirements": False,
        "feedback": "",
        "improvement_actions": [],
        "raw_output": "",
        "parse_error": None,
        "error": None,
        "traceback": None,
    }

    try:
        # Get appropriate LLM with fallback
        llm = llm_map.get(model_name)
        if not llm:
            logger.warning(f"Model {model_name} not found, falling back to gpt-4")
            model_name = DEFAULT_MODEL_FOR_GLOBAL_REFLECTION
            llm = llm_map.get(DEFAULT_MODEL_FOR_GLOBAL_REFLECTION)
            if not llm:
                raise ValueError(f"No available evaluation models")

        llm = llm.with_config({"temperature": 0.7})

        # Build evaluation prompt
        prompt = get_global_reflection_eval_paper_prompt(prompt_text, paper_title, user_query, outline, sections_data)

        # Log full prompt for debugging
        formatted_messages = prompt.format_messages()
        full_prompt = "\n".join([f"{msg.type}: {msg.content}" for msg in formatted_messages])
        logger.info(f"Eval Paper FULL PROMPT STRING:\n{full_prompt}")

        # Get model response
        raw_chain = prompt | llm
        raw_result = await raw_chain.ainvoke({})
        raw_content = raw_result.content

        evaluation_detail["raw_output"] = raw_content
        logger.info(f"Raw model evaluation from {model_name}: {raw_content[:100]}...")

        # Extract JSON from the response
        json_str = extract_json_from_text(raw_content)

        # Parse the JSON and normalize the result
        normalized_result = parse_and_normalize_evaluation(json_str)

        logger.info(f"Normalized evaluation from model: {model_name}, {normalized_result}")

        # Create a valid GlobalReflectionResult
        reflection_result = GlobalReflectionResult(
            meets_requirements=normalized_result["meets_requirements"],
            feedback=normalized_result["feedback"],
            improvement_actions=normalized_result["improvement_actions"],
        )

        # Update evaluation details
        evaluation_detail.update(
            {
                "meets_requirements": reflection_result.meets_requirements,
                "feedback": reflection_result.feedback,
                "improvement_actions": reflection_result.improvement_actions,
            }
        )

        return reflection_result, evaluation_detail

    except ValidationError as ve:
        # Handle Pydantic validation errors
        logger.error(f"Validation error in {model_name} evaluation: {ve}")
        error_message = f"Model evaluation output failed validation: {str(ve)}"
        return create_fallback_result(model_name, error_message, evaluation_detail)

    except Exception as e:
        # Handle general errors
        logger.error(f"Error in global evaluation with model {model_name}: {str(e)}")
        logger.error(traceback.format_exc())

        error_message = f"Evaluation failed with model {model_name}: {str(e)}"
        evaluation_detail["error"] = str(e)
        evaluation_detail["traceback"] = traceback.format_exc()

        return create_fallback_result(model_name, error_message, evaluation_detail)


def extract_json_from_text(text: str) -> str:
    """Extract JSON string from model output text.

    Args:
        text: Raw text that may contain JSON

    Returns:
        Extracted JSON string
    """
    # Try to extract JSON from code blocks first
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try generic code blocks
    json_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Look for JSON-like structure with curly braces
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Return the original text as a fallback
    return text


def parse_and_normalize_evaluation(json_str: str) -> Dict[str, Any]:
    """Parse and normalize evaluation JSON.

    Args:
        json_str: JSON string to parse

    Returns:
        Normalized dictionary with standardized fields
    """
    try:
        parsed_data = json.loads(json_str)

        # Create normalized result structure
        normalized = {
            "meets_requirements": parsed_data.get("meets_requirements", False),
            "feedback": parsed_data.get("feedback", ""),
            "improvement_actions": [],
        }

        # Normalize improvement actions
        actions = parsed_data.get("improvement_actions", [])
        for action in actions:
            # Extract section name from various possible fields
            section = action.get("section", None) or action.get("section_name", "")
            # Extract issues with fallbacks for different formats
            issues = extract_issues_from_action(action)
            # Determine if rewrite is needed
            rewrite = determine_rewrite_status(action)
            # Add normalized action
            if section:  # Only include if we have a valid section
                normalized_action = {
                    "section": section,
                    "issues": issues,
                    "rewrite": rewrite,
                }
                normalized["improvement_actions"].append(normalized_action)

        return normalized

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        # Return default structure if parsing fails
        return {
            "meets_requirements": False,
            "feedback": "Failed to parse evaluation results.",
            "improvement_actions": [],
        }


def extract_issues_from_action(action: Dict[str, Any]) -> List[str]:
    """Extract issues from action with various field formats.

    Args:
        action: Dictionary containing action details

    Returns:
        List of issue strings
    """
    issues = []

    # Handle "issues" field in list format
    if "issues" in action and isinstance(action["issues"], list):
        issues = [issue for issue in action["issues"] if issue]
    # Handle "issues" field in string format
    elif "issues" in action and isinstance(action["issues"], str):
        issues = [action["issues"]]
    # Handle singular "issue" field
    elif "issue" in action and action["issue"]:
        issues = [action["issue"]]
    # Handle "issues_to_address" field
    elif "issues_to_address" in action:
        if isinstance(action["issues_to_address"], list):
            issues = [issue for issue in action["issues_to_address"] if issue]
        elif isinstance(action["issues_to_address"], str):
            issues = [action["issues_to_address"]]

    # Filter out empty strings and ensure we have at least one issue
    issues = [issue for issue in issues if issue]
    if not issues:
        issues = ["Section needs improvement"]

    return issues


def determine_rewrite_status(action: Dict[str, Any]) -> bool:
    """Determine if a section needs rewriting based on action fields.

    Args:
        action: Dictionary containing action details

    Returns:
        Boolean indicating if rewrite is needed
    """
    # Direct rewrite field
    if "rewrite" in action:
        return bool(action["rewrite"])

    # Check for phrases in other fields
    rewrite_phrases = [
        "complete rewriting",
        "rewrite",
        "full rewrite",
        "major revision",
    ]

    # Check modification_needed field
    if "modification_needed" in action:
        mod_text = str(action["modification_needed"]).lower()
        if any(phrase in mod_text for phrase in rewrite_phrases):
            return True

    # Check rewrite_or_modify field
    if "rewrite_or_modify" in action:
        mod_text = str(action["rewrite_or_modify"]).lower()
        if any(phrase in mod_text for phrase in rewrite_phrases):
            return True

    # Check action_needed field
    if "action_needed" in action:
        action_text = str(action["action_needed"]).lower()
        if any(phrase in action_text for phrase in rewrite_phrases):
            return True

    return False


def create_fallback_result(model_name: str, error_message: str, evaluation_detail: ModelEvaluationDetail) -> Tuple[GlobalReflectionResult, ModelEvaluationDetail]:
    """Create fallback result when evaluation fails.

    Args:
        model_name: Name of the model
        error_message: Error message to include
        evaluation_detail: Partial evaluation detail to complete

    Returns:
        Tuple of (fallback result, completed evaluation detail)
    """
    fallback_result = GlobalReflectionResult(
        meets_requirements=False,
        feedback=error_message,
        improvement_actions=[
            {
                "section": "general",
                "issues": ["Model evaluation output couldn't be processed correctly"],
                "rewrite": False,
            }
        ],
    )

    # Update evaluation detail
    evaluation_detail.update(
        {
            "feedback": fallback_result.feedback,
            "improvement_actions": fallback_result.improvement_actions,
        }
    )

    return fallback_result, evaluation_detail


async def evaluate_paper(
    paper_title: str,
    user_query: str,
    outline: Dict[str, Any],
    sections_content: Dict[str, List[Dict[str, Any]]],
    rag_service_url: str,
) -> Tuple[GlobalReflectionResult, List[ModelEvaluationDetail]]:
    """Evaluate an entire paper using multiple LLMs and prompts in parallel.

    Args:
        paper_title: Title of the paper
        user_query: Original user query
        outline: Paper outline structure
        sections_content: Content of all paper sections
        rag_service_url: URL for RAG service

    Returns:
        Tuple of (merged reflection result, evaluation details)
    """
    parser = PydanticOutputParser(pydantic_object=GlobalReflectionResult)

    # Get models and prompts
    models = get_evaluation_models(hash(paper_title) % 1000)
    prompts = get_global_reflection_eval_system_v2()

    # Prepare sections data once for all evaluations
    sections_data = prepare_sections_data(outline, sections_content)
    logger.info(f"eval paper sections_data: {sections_data}")

    # Run evaluations in parallel
    evaluation_tasks = []
    for i, model in enumerate(models):
        # Use a different prompt for each model for diverse perspectives
        prompt = prompts[i % len(prompts)]
        evaluation_tasks.append(run_model_evaluation(model, prompt, paper_title, user_query, outline, sections_data, parser))

    # Execute all evaluations concurrently
    eval_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

    # Process results, handling any exceptions from gather
    results = []
    evaluation_details = []

    for result in eval_results:
        if isinstance(result, Exception):
            # Handle task failures
            logger.error(f"Evaluation task failed: {str(result)}")
            # Create fallback result for the failed evaluation
            fallback = GlobalReflectionResult(
                meets_requirements=False,
                feedback=f"Evaluation failed: {str(result)}",
                improvement_actions=[],
            )
            error_detail = {
                "model": "unknown",
                "error": str(result),
                "meets_requirements": False,
                "feedback": fallback.feedback,
                "improvement_actions": [],
                "raw_output": "",
            }
            results.append(fallback)
            evaluation_details.append(error_detail)
        else:
            # Process successful results
            reflection_result, detail = result
            results.append(reflection_result)
            evaluation_details.append(detail)

    logger.info(f"evaluate_paper eval_results: {evaluation_details}")

    # Analyze results with majority vote
    meets_requirements_count = sum(1 for r in results if r.meets_requirements)
    meets_requirements_majority = meets_requirements_count > len(results) / 2

    # Merge feedback and improvement actions
    merged_feedback = merge_feedback(results, meets_requirements_majority)
    consolidated_actions = consolidate_improvement_actions(results)

    # Create the merged result
    merged_result = GlobalReflectionResult(
        meets_requirements=meets_requirements_majority,
        feedback=merged_feedback,
        improvement_actions=consolidated_actions,
    )

    logger.info(f"Global paper evaluation complete for {paper_title}. " + f"Meets requirements: {meets_requirements_majority}")

    return merged_result, evaluation_details


def merge_feedback(results: List[GlobalReflectionResult], meets_requirements: bool) -> str:
    """Merge feedback from multiple evaluations.

    Args:
        results: List of evaluation results
        meets_requirements: Whether the paper meets requirements overall

    Returns:
        Merged feedback string
    """
    all_feedback = []

    for r in results:
        if r.feedback:
            # Clean up feedback: remove quotes, normalize spacing
            cleaned = r.feedback.strip("\"'").strip()
            if cleaned:
                all_feedback.append(cleaned)

    if not all_feedback:
        return "No feedback provided by evaluators."

    if meets_requirements:
        return "OVERALL ASSESSMENT: Paper meets academic requirements. " + "Key strengths noted across evaluations: " + "; ".join(all_feedback[:2])
    else:
        # For improvement feedback, extract first sentences for conciseness
        concise_feedback = []
        for feedback in all_feedback:
            logger.debug(f"Original feedback: {feedback}")
            first_sentence = feedback.split(". ")[0] + "."
            if len(first_sentence) > 20:  # Ensure it's a meaningful sentence
                concise_feedback.append(first_sentence)
            else:
                concise_feedback.append(feedback)

        return "OVERALL ASSESSMENT: Paper needs improvement. " + "Key issues identified: " + "; ".join(concise_feedback)


def consolidate_improvement_actions(
    results: List[GlobalReflectionResult],
) -> List[Dict[str, Any]]:
    """Consolidate improvement actions from multiple evaluations.

    Args:
        results: List of evaluation results

    Returns:
        Consolidated list of improvement actions
    """
    # Collect all section recommendations
    section_recommendations: Dict[str, SectionRecommendation] = {}

    # Process all results
    for result in results:
        for action in result.improvement_actions:
            section_name = action.get("section", "") or action.get("section_name", "")
            if not section_name:
                continue

            # Initialize recommendation for this section if not exists
            if section_name not in section_recommendations:
                section_recommendations[section_name] = {
                    "section": section_name,
                    "issues": [],
                    "rewrite_votes": 0,
                    "total_votes": 0,
                }

            # Add unique issues from this action
            if "issues" in action:
                add_unique_issues(section_recommendations[section_name]["issues"], action["issues"])

            # Also check for singular "issue" field
            if "issue" in action and action["issue"]:
                if action["issue"] not in section_recommendations[section_name]["issues"]:
                    section_recommendations[section_name]["issues"].append(action["issue"])

            # Also check for "issues_to_address" field
            if "issues_to_address" in action:
                add_unique_issues(
                    section_recommendations[section_name]["issues"],
                    action["issues_to_address"],
                )

            # Count rewrite votes
            section_recommendations[section_name]["rewrite_votes"] += int(determine_rewrite_status(action))

            # Count total votes for this section
            section_recommendations[section_name]["total_votes"] += 1

    # Convert to final format with rewrite decision
    consolidated = []
    for section, data in section_recommendations.items():
        # Decide rewrite based on majority vote
        rewrite = data["rewrite_votes"] > data["total_votes"] / 2

        # Ensure there's at least one issue
        if not data["issues"]:
            data["issues"] = [f"Section '{section}' needs improvement"]

        consolidated.append({"section": section, "issues": data["issues"], "rewrite": rewrite})

    # Sort by most problematic sections first (rewrite needed, then most issues)
    consolidated.sort(key=lambda x: (not x["rewrite"], len(x["issues"])), reverse=True)
    return consolidated


def add_unique_issues(target_list: List[str], issues: Any) -> None:
    """Add unique issues to the target list.

    Args:
        target_list: List to add issues to
        issues: Issues to add (may be string or list)
    """
    if isinstance(issues, list):
        for issue in issues:
            if issue and issue not in target_list:
                target_list.append(issue)
    elif isinstance(issues, str) and issues:
        if issues not in target_list:
            target_list.append(issues)


async def analyze_issues_against_keypoints(
    section_name: str,
    key_points: List[str],
    issues: List[str],
    llm: Any,  # Pass the LLM instance
) -> IssueAnalysisResult:
    """Analyzes issues using an LLM to map them to existing key points or mark them for new ones."""
    if not issues:
        return IssueAnalysisResult(issue_mapping={}, issues_for_new_keypoints=[])
    if not key_points:
        # If no original key points, all issues need new ones
        return IssueAnalysisResult(issue_mapping={}, issues_for_new_keypoints=issues)

    parser = PydanticOutputParser(pydantic_object=IssueAnalysisResult)
    prompt = get_issue_analysis_prompt(
        section_name=section_name,
        key_points=key_points,
        issues=issues,
        format_instructions=parser.get_format_instructions(),
    )
    logger.info(f"Analyzing issues for section '{section_name}'...")
    try:
        chain = prompt | llm | parser
        result = await chain.ainvoke({})
        logger.info(f"Issue analysis result: {result}")
        # Ensure all original key points exist in the mapping, even if empty
        for kp in key_points:
            result.issue_mapping.setdefault(kp, [])
        return result
    except Exception as e:
        logger.error(f"Error during issue analysis for section '{section_name}': {e}")
        # Fallback: Assume all issues relate to all key points (less optimal)
        fallback_mapping = {kp: issues for kp in key_points}
        return IssueAnalysisResult(issue_mapping=fallback_mapping, issues_for_new_keypoints=[])


async def generate_new_key_point_from_issue(
    section_name: str,
    issue: str,
    paper_title: str,
    user_query: str,
    llm: Any,  # Pass the LLM instance
) -> Optional[str]:
    """Generates a new key point string based on a single issue using an LLM."""
    prompt = get_new_key_point_generation_prompt(
        section_name=section_name,
        issue=issue,
        paper_title=paper_title,
        user_query=user_query,
    )
    logger.info(f"Generating new key point for issue: '{issue}' in section '{section_name}'...")
    try:
        chain = prompt | llm
        result = await chain.ainvoke({})
        new_kp = result.content.strip().strip('"')
        if new_kp:
            logger.info(f"Generated new key point: '{new_kp}'")
            return new_kp
        else:
            logger.warning(f"LLM returned empty string for new key point generation for issue: {issue}")
            return None
    except Exception as e:
        logger.error(f"Error generating new key point for issue '{issue}': {e}")
        return None


async def generate_single_enhanced_query(
    section_name: str,
    key_point: str,
    original_query: Optional[str],
    relevant_issues: List[str],  # Takes only relevant issues
    paper_title: str,
    user_query: str,
    llm: Any,  # Pass the LLM instance to avoid repeated lookups
) -> str:
    """Generate a single enhanced search query for a key point based on relevant issues."""
    try:
        # Build search query prompt using only relevant issues
        search_query_prompt = get_enhanced_search_query_prompt_v2(
            paper_title,
            user_query,
            section_name,
            key_point,
            original_query,
            relevant_issues,  # Pass only relevant issues here
        )

        logger.info(f"Generating single enhanced search query for key point: {key_point} with issues: {relevant_issues}")

        # Log full prompt for debugging
        formatted_messages = search_query_prompt.format_messages()
        full_prompt = "\n".join([f"{msg.type}: {msg.content}" for msg in formatted_messages])
        logger.info(f"GEN SINGLE ENHANCED QUERY FULL PROMPT STRING:\n{full_prompt}")

        # Generate query
        messages = await search_query_prompt.ainvoke({})
        query_result = await llm.ainvoke(messages)

        # Clean up the response
        clean_query = query_result.content.strip()
        clean_query = re.sub(r'^[\d\.)\-\s"\']+\s*', "", clean_query)
        clean_query = re.sub(r'["\']+$', "", clean_query)

        if clean_query:
            return clean_query
        else:
            # Fallback if LLM returns empty string
            logger.warning(f"LLM returned empty query for '{key_point}'. Using fallback.")
            return original_query if original_query else f"research on {key_point}"

    except Exception as e:
        logger.error(f"Error generating single search query for '{key_point}': {str(e)}")
        # Fallback query on error
        return original_query if original_query else f"research on {key_point}"


async def refine_keypoints_and_queries_based_on_issues(
    section_name: str,
    original_key_points: List[str],
    original_search_queries: List[str],
    improvement_issues: List[str],
    paper_title: str,
    user_query: str,
) -> Tuple[List[str], List[str]]:
    """
    Analyzes issues against key points, enhances/generates queries, and generates new key points if needed.
    """
    logger.info(f"Refining key points and queries for section '{section_name}' based on {len(improvement_issues)} issues: {improvement_issues}")

    # Select LLM once for potential multiple calls
    llm = llm_map.get(DEFAULT_MODEL_FOR_GLOBAL_REFLECTION)
    if not llm:
        logger.error(f"LLM '{DEFAULT_MODEL_FOR_GLOBAL_REFLECTION}' not found for refinement. Aborting refinement.")
        # Return originals or basic fallbacks if LLM is unavailable
        return original_key_points, [q if q else f"research on {kp}" for kp, q in zip(original_key_points, original_search_queries)]

    # --- Issue Analysis ---
    analysis_result = await analyze_issues_against_keypoints(
        section_name=section_name,
        key_points=original_key_points,
        issues=improvement_issues,
        llm=llm,
    )
    issues_map = analysis_result.issue_mapping
    issues_for_new_points = analysis_result.issues_for_new_keypoints
    logger.info(f"Issue analysis mapped: {issues_map}")
    logger.info(f"Issues requiring new key points: {issues_for_new_points}")

    final_key_points = list(original_key_points)
    # Use a dictionary to map key points to queries for easier updates
    final_query_map: Dict[str, str] = {}

    # --- Generate/Enhance queries for EXISTING key points ---
    query_tasks_existing = []
    for i, kp in enumerate(original_key_points):
        original_query = original_search_queries[i] if i < len(original_search_queries) else None
        relevant_issues = issues_map.get(kp, [])  # Get issues specifically mapped to this KP
        # If no specific issues mapped by analysis, but there were issues overall, use all as fallback
        if not relevant_issues and improvement_issues:
            logger.warning(f"No specific issues mapped to existing key point '{kp}'. Using all issues as fallback for query generation.")
            relevant_issues = improvement_issues

        # Only generate query if there are relevant issues or it's an original point
        if relevant_issues or kp in original_key_points:
            query_tasks_existing.append(
                generate_single_enhanced_query(
                    section_name,
                    kp,
                    original_query,
                    relevant_issues,
                    paper_title,
                    user_query,
                    llm,
                )
            )
        else:
            # Should not happen if kp is in original_key_points, but as safety
            query_tasks_existing.append(asyncio.create_task(asyncio.sleep(0, result=f"research on {kp}")))  # Placeholder async task

    # --- Generate NEW Key Points (concurrently) ---
    new_kp_tasks = []
    if issues_for_new_points:
        for issue in issues_for_new_points:
            new_kp_tasks.append(generate_new_key_point_from_issue(section_name, issue, paper_title, user_query, llm))

    # Gather results of new key point generation
    generated_new_key_points_results = await asyncio.gather(*new_kp_tasks)
    newly_generated_key_points: List[Tuple[str, str]] = []  # Store as (new_kp, originating_issue)
    for issue, new_kp in zip(issues_for_new_points, generated_new_key_points_results):
        if new_kp and new_kp not in final_key_points:  # Ensure it's valid and unique
            final_key_points.append(new_kp)
            newly_generated_key_points.append((new_kp, issue))
        elif new_kp:
            logger.warning(f"Generated new key point '{new_kp}' already exists or is similar. Skipping.")
        else:
            logger.warning(f"Failed to generate new key point for issue: {issue}")

    # --- Generate queries for NEWLY generated key points ---
    query_tasks_new = []
    if newly_generated_key_points:
        for new_kp, originating_issue in newly_generated_key_points:
            # Use the originating issue as the context for the new query
            query_tasks_new.append(
                generate_single_enhanced_query(
                    section_name,
                    new_kp,
                    None,
                    [originating_issue],
                    paper_title,
                    user_query,
                    llm,
                )
            )

    # --- Gather all query generation results ---
    generated_queries_existing = await asyncio.gather(*query_tasks_existing)
    generated_queries_new = await asyncio.gather(*query_tasks_new)

    # Populate the query map for existing key points
    for i, kp in enumerate(original_key_points):
        if i < len(generated_queries_existing):
            final_query_map[kp] = generated_queries_existing[i]
        else:  # Fallback if something went wrong with task list lengths
            final_query_map[kp] = original_search_queries[i] if i < len(original_search_queries) and original_search_queries[i] else f"research on {kp}"

    # Populate the query map for new key points
    for i, (new_kp, _) in enumerate(newly_generated_key_points):
        if i < len(generated_queries_new):
            final_query_map[new_kp] = generated_queries_new[i]
        else:  # Fallback
            final_query_map[new_kp] = f"research on {new_kp}"

    # --- Final Assembly ---
    # Create the final list of queries in the same order as final_key_points
    final_search_queries = [final_query_map.get(kp, f"research on {kp}") for kp in final_key_points]

    logger.info(f"Refinement resulted in {len(final_key_points)} final key points.")
    logger.debug(f"Final Key Points: {final_key_points}")
    logger.debug(f"Final Search Queries: {final_search_queries}")

    return final_key_points, final_search_queries


async def process_section_for_rewriting(
    section_name: str,
    section_info: List[Dict[str, Any]],
    paper_title: str,
    user_query: str,
    outline: Dict[str, Any],
    rag_service_url: str,
) -> RewriteResult:
    """Process a section that needs rewriting by refining key points and search queries based on issues."""
    try:
        logger.info(f"Processing section '{section_name}' for rewriting. Initial section_info: {section_info}")

        # Ensure section_info is a list
        if not isinstance(section_info, list):
            section_info = [section_info] if section_info else []

        # Extract necessary data
        improvement_issues = extract_improvement_issues(section_info)
        original_search_queries, original_key_points, section_metadata = extract_section_metadata(section_info, section_name)

        parent_section = section_metadata.get("parent_section", "")
        # Attempt to find section_index (this might need adjustment based on actual outline structure)
        section_index = None
        if parent_section and parent_section in outline:
            # Example: If outline[parent_section] is a dict with 'subsections' list
            subsections = outline[parent_section].get("subsections", [])
            for idx, sub in enumerate(subsections):
                if isinstance(sub, dict) and sub.get("name") == section_name:
                    section_index = sub.get("index", idx)  # Use index field or list index
                    break
        elif section_name in outline:  # Example: If outline is flat
            section_index = outline[section_name].get("section_index")

        if section_index is None:
            logger.warning(f"Could not determine section_index for {section_name}. Using default 0.")
            section_index = 0  # Default or fallback

        logger.info(f"Section: '{section_name}', Parent: '{parent_section}', Index: {section_index}")
        logger.info(f"Improvement issues: {improvement_issues}")
        logger.info(f"Original key points: {original_key_points}")
        logger.info(f"Original search queries: {original_search_queries}")

        # Ensure we have fallback issues/key points if empty
        if not improvement_issues:
            improvement_issues = [f"Improve {section_name} section with more detailed content"]
            logger.info(f"No specific issues found, using default: {improvement_issues}")
        if not original_key_points:
            original_key_points = [f"Content for {section_name} section"]
            logger.info(f"No original key points found, using default: {original_key_points}")
        # Ensure search queries list matches key points list length
        while len(original_search_queries) < len(original_key_points):
            original_search_queries.append("")

        # *** CORE LOGIC ***
        # Analyze issues, refine/generate key points, and generate corresponding queries
        final_key_points, final_search_queries = await refine_keypoints_and_queries_based_on_issues(
            section_name=section_name,
            original_key_points=original_key_points,
            original_search_queries=original_search_queries,
            improvement_issues=improvement_issues,
            paper_title=paper_title,
            user_query=user_query,
        )
        logger.info(f"Final key points ({len(final_key_points)}): {final_key_points}")
        logger.info(f"Final search queries ({len(final_search_queries)}): {final_search_queries}")
        # *** END CORE LOGIC CHANGE ***

        # Prepare parameters for section_writer using the final lists
        section_params = {
            "section_name": section_name,
            "section_index": section_index,
            "parent_section": parent_section,
            "user_query": user_query,
            "section_key_points": final_key_points,  # Use final list
            "paper_title": paper_title,
            "search_queries": final_search_queries,  # Use final list
            "rewrite": True,
        }

        # Call section_writer to generate new content based on refined points/queries
        logger.info(f"Calling section_writer_async with params: {section_params}")
        new_content = await section_writer_async(section_params, rag_service_url)
        logger.info(f"Received new content from section_writer_async: {new_content}")

        # Update the section items using the new content and final key points/queries
        new_section_items = await update_section_content(
            section_info=section_info,  # Pass original for structure reference
            final_search_queries=final_search_queries,  # Pass final queries
            final_key_points=final_key_points,  # Pass final key points
            new_content=new_content,
            section_name=section_name,
            paper_title=paper_title,
        )

        return {
            "section_name": section_name,
            "section_index": section_index,
            "parent_section": parent_section,
            "rewrite_successful": True,
            "section_content": new_section_items,
            # Include final points/queries in result for traceability
            "final_key_points": final_key_points,
            "final_search_queries": final_search_queries,
            "improvement_issues": improvement_issues,
            "error": None,
            "traceback": None,
        }

    except Exception as e:
        logger.error(f"Error processing section {section_name} for rewriting: {str(e)}")
        logger.error(traceback.format_exc())
        # Attempt to extract basic info even in case of error for the result dict
        issues = extract_improvement_issues(section_info) if isinstance(section_info, list) else []
        _, key_points, metadata = extract_section_metadata(section_info, section_name) if isinstance(section_info, list) else ([], [], {})
        parent = metadata.get("parent_section", "")
        idx = None  # Index might be hard to get reliably on error

        return {
            "section_name": section_name,
            "section_index": idx,
            "parent_section": parent,
            "rewrite_successful": False,
            "section_content": None,
            "final_key_points": key_points,  # Original key points on error
            "final_search_queries": None,  # Queries might not be available
            "improvement_issues": issues,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def extract_improvement_issues(section_info: List[Dict[str, Any]]) -> List[str]:
    """Extract improvement issues from section info.

    Args:
        section_info: List of section information dictionaries

    Returns:
        List of improvement issues
    """
    improvement_issues = []

    for item in section_info:
        if not isinstance(item, dict):
            continue

        if "improvement_issues" in item:
            issues = item.get("improvement_issues", [])
            if isinstance(issues, list):
                improvement_issues.extend(issues)
            else:
                improvement_issues.append(str(issues))

        # Also check "issues" field
        if "issues" in item:
            issues = item.get("issues", [])
            if isinstance(issues, list):
                improvement_issues.extend(issues)
            else:
                improvement_issues.append(str(issues))

    # Ensure unique issues
    unique_issues = []
    for issue in improvement_issues:
        if issue and issue not in unique_issues:
            unique_issues.append(issue)

    return unique_issues


def extract_section_metadata(section_info: List[Dict[str, Any]], section_name: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Extract metadata from section information.

    Args:
        section_info: List of section information dictionaries
        section_name: Name of the section

    Returns:
        Tuple of (search queries, key points, section metadata)
    """
    original_search_queries = []
    original_key_points = []
    section_metadata = {"parent_section": ""}

    for item in section_info:
        if not isinstance(item, dict):
            continue

        # Extract search queries
        if "search_query" in item and item["search_query"] not in original_search_queries:
            original_search_queries.append(item["search_query"])

        # Extract key points
        if "section_point" in item and item["section_point"] not in original_key_points:
            original_key_points.append(item["section_point"])

        # Extract section metadata (use from first item)
        if "parent_section" in item and not section_metadata["parent_section"]:
            section_metadata["parent_section"] = item["parent_section"]

    # If no key points found, use section name as fallback
    if not original_key_points:
        original_key_points = [f"Content for {section_name}"]

    return original_search_queries, original_key_points, section_metadata


async def update_section_content(
    section_info: List[Dict[str, Any]],
    final_search_queries: List[str],  # Renamed parameter
    final_key_points: List[str],  # Renamed parameter
    new_content: Dict[str, Any],
    section_name: str,
    paper_title: str,
) -> List[Dict[str, Any]]:
    """Update section content based on the final key points and newly generated content."""
    new_section_items = []
    num_key_points = len(final_key_points)
    num_queries = len(final_search_queries)

    logger.info(f"Updating section content for '{section_name}' with {num_key_points} final key points.")

    # Iterate through the final list of key points
    for i in range(num_key_points):
        current_key_point = final_key_points[i]
        # Try to find a corresponding original item for metadata preservation (best effort)
        # This assumes original section_info might roughly correspond to original key points
        original_item = section_info[i] if i < len(section_info) and isinstance(section_info[i], dict) else {}
        new_item = original_item.copy()  # Start with original structure/metadata

        # Get the corresponding search query, with fallback
        current_search_query = final_search_queries[i] if i < num_queries else f"research on {current_key_point}"
        if i >= num_queries:
            logger.warning(f"Query missing for key point '{current_key_point}'. Using fallback: '{current_search_query}'")

        # Find the newly generated content for this specific key point
        current_new_content = new_content.get(current_key_point)
        if not current_new_content:
            # Try fallback matching (case-insensitive substring) if direct key match fails
            found_match = False
            for key, content in new_content.items():
                if current_key_point.lower() in key.lower() or key.lower() in current_key_point.lower():
                    current_new_content = content
                    logger.info(f"Found content for '{current_key_point}' using fallback match with key '{key}'")
                    found_match = True
                    break
            if not found_match:
                logger.warning(f"Could not find generated content for key point: '{current_key_point}'. Content will be missing.")
                # Use placeholder content if none found
                current_new_content = {
                    "section_text": f"[Content generation failed or missing for: {current_key_point}]",
                    "search_query": current_search_query,  # Still associate the query
                }

        logger.debug(f"Processing key point {i+1}/{num_key_points}: '{current_key_point}'")
        logger.debug(f"  - Search Query: '{current_search_query}'")
        logger.debug(f"  - Found Content Keys: {current_new_content.keys() if isinstance(current_new_content, dict) else 'None'}")

        # Update the item dictionary
        new_item["section_point"] = current_key_point  # Ensure key point is correct
        new_item["search_query"] = current_search_query  # Ensure query is correct

        # Update fields from the generated content if it's a dictionary
        if isinstance(current_new_content, dict):
            if "section_text" in current_new_content:
                new_item["section_text"] = current_new_content["section_text"]
            elif "section_text" not in new_item:  # Ensure field exists even if empty
                new_item["section_text"] = ""

            if "reportIndexList" in current_new_content:
                new_item["reportIndexList"] = current_new_content["reportIndexList"]
            # Decide whether to keep old reportIndexList if not in new content, or clear it. Clearing is safer.
            elif "reportIndexList" in new_item:
                new_item["reportIndexList"] = []  # Clear if not provided in new content

            # Preserve other metadata from original_item if not overwritten
            for key, value in original_item.items():
                if key not in new_item and key not in [
                    "section_point",
                    "search_query",
                    "section_text",
                    "reportIndexList",
                    "section_summary",
                    "summary_details",
                ]:
                    new_item[key] = value
        else:
            # Handle case where content is not a dict (unexpected)
            logger.error(f"Unexpected content format for key point '{current_key_point}': {type(current_new_content)}")
            new_item["section_text"] = "[Error: Invalid content format received]"

        # Generate/update summary for the new text
        try:
            text_for_summary = new_item.get("section_text", "")
            if text_for_summary and not text_for_summary.startswith("[Content generation failed"):
                summary_sentences, summary_details = await generate_section_summary(
                    section_name=section_name,
                    section_content={"section_text": text_for_summary},
                    paper_title=paper_title,
                )
                summary_details_dict = summary_details.dict() if isinstance(summary_details, BaseModel) else summary_details
                new_item["section_summary"] = summary_sentences
                new_item["summary_details"] = summary_details_dict
            elif not text_for_summary:
                new_item["section_summary"] = ["Content is empty."]
                new_item["summary_details"] = {}
            else:  # Handle placeholder text
                new_item["section_summary"] = ["Content generation failed."]
                new_item["summary_details"] = {}

        except Exception as e:
            logger.error(f"Error generating summary for key point '{current_key_point}': {str(e)}")
            new_item["section_summary"] = ["Summary generation failed."]
            new_item["summary_details"] = {"error": str(e)}

        # Ensure essential fields like section_name and parent_section are present
        new_item.setdefault("section_name", section_name)
        if "parent_section" not in new_item and "parent_section" in original_item:
            new_item["parent_section"] = original_item["parent_section"]

        new_section_items.append(new_item)

    # Handle cases where the process might yield no items
    if not new_section_items:
        logger.warning(f"Update process resulted in zero items for section '{section_name}'.")
        # Return a single item indicating failure if key points existed but processing failed
        if num_key_points > 0:
            return [
                {
                    "section_name": section_name,
                    "section_point": "Processing Error",
                    "search_query": "",
                    "section_text": "Failed to update section content items.",
                    "section_summary": ["Error during content update."],
                }
            ]
        # Otherwise, return empty list if there were no key points to begin with
        return []

    return new_section_items


async def global_reflection_async(
    params: Dict[str, Any],
    rag_service_url: str,
    max_iterations: int = GLOBAL_REFLECTION_MAX_TURNS,
) -> Dict[str, Any]:
    """Asynchronous implementation of global reflection with iterative improvement.

    Args:
        params: Parameters including paper title, query, outline, and content
        rag_service_url: URL for RAG service
        max_iterations: Maximum number of improvement iterations

    Returns:
        Dictionary with reflection results and history
    """
    # Extract parameters
    paper_title = params.get("paper_title", "")
    user_query = params.get("user_query", "")
    outline = params.get("outline", {})
    sections_content = params.get("sections_content", {})

    # Initialize tracking
    global_history = [
        {
            "iteration": 0,
            "sections_content": sections_content.copy(),
            "timestamp": str(uuid.uuid4()),
        }
    ]
    evaluation_history: List[EvaluationHistory] = []
    rewrite_history: List[RewriteHistory] = []
    meets_requirements = False
    iteration = 0

    logger.info(f"Starting global reflection for paper: {paper_title}")

    # Main iteration loop
    while not meets_requirements and iteration < max_iterations:
        logger.info(f"Global reflection iteration {iteration+1} for paper: {paper_title}")

        try:
            # Evaluate entire paper
            reflection_result, eval_details = await evaluate_paper(
                paper_title=paper_title,
                user_query=user_query,
                outline=outline,
                sections_content=sections_content,
                rag_service_url=rag_service_url,
            )

            logger.info(f"iter-[{iteration}]--eval_details: {eval_details}")
            logger.info(f"iter-[{iteration}]--reflection_result: {dict(reflection_result)}")
            meets_requirements = reflection_result.meets_requirements

            # Save evaluation details
            evaluation_history.append(
                {
                    "iteration": iteration + 1,
                    "timestamp": str(uuid.uuid4()),
                    "model_evaluations": eval_details,
                    "merged_result": {
                        "meets_requirements": meets_requirements,
                        "feedback": reflection_result.feedback,
                        "improvement_actions": reflection_result.improvement_actions,
                    },
                }
            )

            if meets_requirements:
                logger.info(f"Paper {paper_title} meets requirements")
                break

            # Process sections that need improvement
            logger.info(f"Paper [{paper_title}] needs improvement: {reflection_result.feedback}")

            # Identify sections to rewrite
            sections_to_rewrite = identify_sections_to_rewrite(reflection_result, sections_content)
            logger.info(f"Sections to rewrite: {sections_to_rewrite}")
            # Prepare section info for rewriting
            rewrite_tasks = prepare_rewrite_tasks(
                sections_to_rewrite,
                paper_title,
                user_query,
                outline,
                rag_service_url,
            )

            # Execute rewrite tasks concurrently
            rewrite_results = await asyncio.gather(*rewrite_tasks, return_exceptions=True)

            # Process rewrite results, handling any exceptions
            processed_results = []
            for result in rewrite_results:
                if isinstance(result, Exception):
                    logger.error(f"Rewrite task failed: {str(result)}")
                    # Create fallback result
                    error_result: RewriteResult = {
                        "section_name": "unknown",
                        "section_index": None,
                        "parent_section": None,
                        "rewrite_successful": False,
                        "section_content": None,
                        "enhanced_search_queries": None,
                        "improvement_issues": None,
                        "error": str(result),
                        "traceback": traceback.format_exc(),
                    }
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)

            # Track rewrite results
            rewrite_history.append(
                {
                    "iteration": iteration + 1,
                    "timestamp": str(uuid.uuid4()),
                    "sections_rewritten": len(processed_results),
                    "rewrite_details": processed_results,
                }
            )

            # Update sections_content with rewritten content
            sections_updated = update_sections_content(processed_results, sections_content)

            # Store updated content history
            if sections_updated:
                global_history.append(
                    {
                        "iteration": iteration + 1,
                        "sections_content": sections_content.copy(),
                        "timestamp": str(uuid.uuid4()),
                    }
                )
                logger.info(f"Updated paper content with rewritten sections")
            else:
                logger.warning(f"No sections were successfully rewritten for iteration {iteration+1}")

        except Exception as e:
            logger.error(f"Error in global reflection iteration {iteration+1}: {str(e)}")
            logger.error(traceback.format_exc())
            break

        iteration += 1

    # Prepare final result
    result = {
        "paper_title": paper_title,
        "meets_requirements": meets_requirements,
        "final_feedback": (evaluation_history[-1]["merged_result"]["feedback"] if evaluation_history else "No evaluation was completed"),
        "sections_content": sections_content,
        "global_reflection_process_metadata": {
            "iterations_performed": iteration,
            "max_iterations": max_iterations,
            "max_iterations_reached": iteration >= max_iterations,
            "sections_rewritten_count": sum(entry.get("sections_rewritten", 0) for entry in rewrite_history),
        },
        "global_reflection_process_history": {
            "initial_state": global_history[0] if global_history else {},
            "final_state": global_history[-1] if global_history else {},
            "evaluations": evaluation_history,
            "rewrites": rewrite_history,
        },
    }

    return result


def identify_sections_to_rewrite(
    reflection_result: GlobalReflectionResult,
    sections_content: Dict[str, List[Dict[str, Any]]],
) -> List[SectionToRewrite]:
    """Identify sections that need to be rewritten.

    Args:
        reflection_result: Result of paper evaluation
        sections_content: Content of all paper sections

    Returns:
        List of sections that need rewriting
    """
    sections_to_rewrite = []

    for action in reflection_result.improvement_actions:
        if action.get("rewrite", False):
            section_name = action.get("section")
            if section_name and section_name in sections_content:
                # Ensure we have at least one issue
                issues = action.get("issues", [])
                if not issues:
                    issues = [f"Section {section_name} needs improvement"]

                sections_to_rewrite.append(
                    {
                        "section_name": section_name,
                        "section_info": sections_content[section_name],
                        "issues": issues,
                    }
                )

    return sections_to_rewrite


def prepare_rewrite_tasks(
    sections_to_rewrite: List[SectionToRewrite],
    paper_title: str,
    user_query: str,
    outline: Dict[str, Any],
    rag_service_url: str,
) -> List[asyncio.Task]:
    """Prepare rewrite tasks for sections.

    Args:
        sections_to_rewrite: List of sections that need rewriting
        paper_title: Title of the paper
        user_query: Original user query
        outline: Paper outline structure
        rag_service_url: URL for RAG service

    Returns:
        List of rewrite tasks
    """
    rewrite_tasks = []

    for section in sections_to_rewrite:
        # Convert section_info to list format and add improvement issues
        section_info = prepare_section_info_for_rewriting(section)

        # Create rewrite task
        rewrite_tasks.append(
            process_section_for_rewriting(
                section_name=section["section_name"],
                section_info=section_info,
                paper_title=paper_title,
                user_query=user_query,
                outline=outline,
                rag_service_url=rag_service_url,
            )
        )

    return rewrite_tasks


def prepare_section_info_for_rewriting(
    section: SectionToRewrite,
) -> List[Dict[str, Any]]:
    """Prepare section information for rewriting.

    Args:
        section: Section information

    Returns:
        List of section information dictionaries with improvement issues
    """
    # Convert section_info to list format
    if isinstance(section["section_info"], list):
        section_info = []
        for item in section["section_info"]:
            new_item = item.copy() if isinstance(item, dict) else {}
            new_item["improvement_issues"] = section["issues"]
            section_info.append(new_item)
    else:
        # If not a list, convert to a single-item list
        section_info = [{}]
        if isinstance(section["section_info"], dict):
            section_info = [section["section_info"].copy()]

        # Add improvement issues to the first item
        if section_info:
            section_info[0]["improvement_issues"] = section["issues"]

    return section_info


def update_sections_content(
    rewrite_results: List[RewriteResult],
    sections_content: Dict[str, List[Dict[str, Any]]],
) -> bool:
    """Update sections content with rewritten content.

    Args:
        rewrite_results: Results of rewriting sections
        sections_content: Original sections content to update

    Returns:
        Boolean indicating if any sections were updated
    """
    sections_updated = False

    for result in rewrite_results:
        section_name = result.get("section_name")
        if result.get("rewrite_successful") and section_name in sections_content and result.get("section_content"):
            sections_content[section_name] = result["section_content"]
            sections_updated = True

    return sections_updated


async def global_reflection(params: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for global reflection of the entire paper.

    Args:
        params: Parameters including paper title, query, outline, and content

    Returns:
        Dictionary with reflection results and history
    """
    logger.info(f"Starting global reflection process: {params.get('paper_title', '')}")
    logger.info(f"running global_reflection: {params}")

    # Extract parameters with defaults
    paper_title = params.get("paper_title", "")
    user_query = params.get("user_query", "")
    outline = params.get("outline", {})
    sections_content = params.get("sections_content", {})
    rag_service_url = params.get("rag_service_url", "http://xxxxx:9528/chat")
    max_iterations = params.get("max_iterations", GLOBAL_REFLECTION_MAX_TURNS)

    # Handle "sections" field in outline
    if "sections" in outline:
        outline = outline["sections"]

    outline = {k: v for k, v in outline.items() if not v["is_conclusion"]}
    logger.info(f"Outline after filtering conclusions: {outline}")

    # Track execution time
    start_time = time.time()

    try:
        # Call async implementation
        reflection_result = await global_reflection_async(
            {
                "paper_title": paper_title,
                "user_query": user_query,
                "outline": outline,
                "sections_content": sections_content,
            },
            rag_service_url,
            max_iterations,
        )
    except Exception as e:
        logger.error(f"Global reflection failed: {str(e)}")
        logger.error(traceback.format_exc())

        # Create fallback result on failure
        reflection_result = {
            "paper_title": paper_title,
            "meets_requirements": False,
            "final_feedback": f"Global reflection process failed: {str(e)}",
            "sections_content": sections_content,
            "global_reflection_process_metadata": {
                "iterations_performed": 0,
                "max_iterations": max_iterations,
                "max_iterations_reached": False,
                "sections_rewritten_count": 0,
                "error": str(e),
            },
        }

    # Add performance metrics
    end_time = time.time()
    reflection_result["global_reflection_performance"] = {
        "execution_time_seconds": end_time - start_time,
        "start_timestamp": start_time,
        "end_timestamp": end_time,
    }

    logger.info(f"Global reflection complete for: {paper_title}")
    logger.info(reflection_result)

    return reflection_result


# Example usage
def get_test_params() -> Dict[str, Any]:
    """Return example parameters for testing."""
    # from example import example_global_reflection_in
    # return example_global_reflection_in
    from utils import format_sections_for_global_reflection

    src_file = "./example_full_data.json"
    with open(src_file, "r") as f:
        example_global_reflection_in = json.load(f)

    formatted_sections = {}
    for section_name, section_data in example_global_reflection_in["sections"].items():
        section_list = []

        reflection_data = section_data["reflection_results"]
        if not reflection_data and section_data["content"]:
            reflection_data = section_data["content"]

        for key_point, content in reflection_data.items():
            # Skip empty content
            if not content:
                continue

            section_list.append(
                {
                    "section_index": content.get("section_index", 0),
                    "parent_section": content.get("parent_section", ""),
                    "search_query": content.get("search_query", ""),
                    "section_point": content.get("section_key_point", ""),
                    "section_text": content.get("section_text", ""),
                    "section_summary": content.get("section_summary", ""),
                    "main_figure_data": content.get("main_figure_data", ""),
                    "main_figure_caption": content.get("main_figure_caption", ""),
                    "reportIndexList": content.get("reportIndexList", []),
                }
            )
        if section_list:
            formatted_sections[section_name] = section_list

    reflection_params = {
        "paper_title": example_global_reflection_in["paper_title"],
        "user_query": example_global_reflection_in["user_query"],
        "outline": example_global_reflection_in["outline_structure_wo_query"],
        "sections_content": formatted_sections,
        "rag_service_url": example_global_reflection_in["rag_service_url"],
        "max_iterations": 2,
    }
    return reflection_params


async def test_global_reflection_async() -> Optional[Dict[str, Any]]:
    """Test the global reflection function asynchronously."""
    try:
        test_params = get_test_params()
        global_reflection_results = await global_reflection(test_params)
        print(json.dumps(global_reflection_results, indent=2))
        dest_file = "./temp/global_reflection_test_async.json"
        with open(dest_file, "w") as f:
            json.dump(global_reflection_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {dest_file}")
        return global_reflection_results
    except Exception as e:
        print(f"Error during global reflection test: {e}")
        traceback.print_exc()
        return None


def test_global_reflection() -> None:
    """Test the synchronous global_reflection function."""
    try:
        test_params = get_test_params()
        global_reflection_results = asyncio.run(global_reflection(test_params))
        print(json.dumps(global_reflection_results, indent=2))
        dest_file = "./temp/global_reflection_test.json"
        with open(dest_file, "w") as f:
            json.dump(global_reflection_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {dest_file}")
    except Exception as e:
        print(f"Error during synchronous reflection test: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    # test_global_reflection()
    pass
