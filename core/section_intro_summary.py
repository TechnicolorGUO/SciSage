# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Section summary generation and core points extraction
# ==================================================================
from configuration import SECTION_SUMMARY_MODEL
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from log import logger
import time
from langchain_core.prompts import ChatPromptTemplate
from prompt_manager import get_section_summary_intro_prompt
from model_factory import llm_map

async def generate_section_intro_summary(
    section_name: str,
    key_point_contents: Dict[str, Any],
    paper_title: str,
    user_query: str,
) -> str:
    """
    Generates a concise introductory summary for a section based on its key points' content/summaries.

    Args:
        section_name: The name of the section.
        key_point_contents: A dictionary where keys are key point names and values are
                           dictionaries containing 'section_summary' or 'section_text'.
        paper_title: The title of the paper.
        user_query: The original user query.

    Returns:
        A string containing the generated introductory summary.
    """
    logger.info(f"Generating introductory summary for section: {section_name}")
    start_time = time.time()

    # Select LLM
    llm = llm_map.get(SECTION_SUMMARY_MODEL)
    if not llm:
        logger.warning(
            f"Summary model {SECTION_SUMMARY_MODEL} not found, falling back to gpt-4."
        )
        llm = llm_map.get("gpt-4")  # Fallback model

    if not llm:
        logger.error("No suitable LLM found for summary generation.")
        return f"Introduction for section {section_name}."  # Final fallback


    # Extract summaries or text from key points
    key_points_info = []
    for key_point, content_data in key_point_contents.items():
        # Prioritize 'section_summary', fallback to 'section_text'
        summary = content_data.get("section_summary")
        if isinstance(summary, list):  # Handle list summaries from reflection
            summary = " ".join(summary)
        if not summary or not str(summary).strip():
            summary = content_data.get("section_text", "")

        if summary and str(summary).strip():
            # Limit length to avoid excessive context
            summary_text = str(summary)
            key_points_info.append(
                f"Key Point: {key_point}\nContent/Summary: {summary_text}\n---"
            )

    if not key_points_info:
        logger.warning(
            f"No content found to generate summary for section: {section_name}"
        )
        return f"This section delves into {section_name}."  # Basic fallback

    combined_info = "\n".join(key_points_info)

    # Define the prompt template
    prompt_template = get_section_summary_intro_prompt(
        paper_title, user_query, section_name, combined_info
    )

    # Create the chain and invoke
    chain = prompt_template | llm
    try:
        response = await chain.ainvoke(
            {
            }
        )
        summary = response.content.strip()
        logger.info(f"Successfully generated intro summary for section: {section_name}")
        duration = time.time() - start_time
        logger.debug(
            f"Intro summary generation for {section_name} took {duration:.2f}s"
        )
        return summary
    except Exception as e:
        logger.error(
            f"Failed to generate intro summary for section {section_name}: {e}\n{traceback.format_exc()}"
        )
        return f"This section provides an overview of {section_name} based on the research query."  # Error fallback
