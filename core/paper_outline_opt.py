#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Paper Outline Generator with English prompts - Optimized
# ==================================================================
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
from typing import Dict, List, Optional, Any, Union, TypeVar, Callable, Tuple, cast
import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field, asdict

from pydantic import BaseModel, Field, ValidationError
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END

from configuration import (
    OUTLINE_GENERAOR_MODELS,
    DEFAULT_MODEL_FOR_OUTLINE,
    MODEL_GEN_QUERY,
)
from log import logger
from model_factory import llm_map
from prompt_manager import (
    get_research_field_prompt,
    get_outline_generation_prompt,
    get_outline_generation_prompt_v2,
    get_outline_generation_strick_prompt,
    get_outline_synthesis_prompt,
    get_outline_reflection_prompt,
    get_outline_reflection_prompt_v2,
    get_query_generation_prompt,
    get_query_generation_prompt_v2,
    get_outline_improve_prompt,
    get_outline_improve_prompt_v2,
    get_outline_conclusion_judge_prompt,
    get_seed_outline_template,
)
from models import QueryIntent

from utils import safe_invoke


# Configuration constants
MAX_RETRY_ATTEMPTS = 3
THREAD_POOL_SIZE = 4
QUERY_GENERATION_CONCURRENCY = 4
CONCLUSION_IMPORTANCE_SCORE = 0.2
REGULAR_SECTION_IMPORTANCE_SCORE = 0.8

# Default LLM cache - this avoids repeatedly referencing the same model
default_llm = llm_map[DEFAULT_MODEL_FOR_OUTLINE]

# Cached set of conclusion-related keywords
CONCLUSION_KEYWORDS = frozenset(
    {
        "conclusion",
        "conclusions",
        "concluding remarks",
        "summary",
        "final remarks",
        "discussion and conclusion",
        "future work",
        "future directions",
        "future research",
        "limitations and future work",
        "discussion and future work",
        "final thoughts",
        "closing remarks",
        "reference",
    }
)


# ======================================
# Data Models
# ======================================
class OutlineSection(BaseModel):
    """Section structure for the paper outline"""

    title: str = Field(description="Section title")
    key_points: List[str] = Field(description="Key points of the section content")
    subsections: Optional[List["OutlineSection"]] = Field(
        default=None, description="Subsections of this section"
    )


class PaperOutline(BaseModel):
    """Complete paper outline structure"""

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract")
    sections: List[OutlineSection] = Field(description="Paper section structure")


class ReflectionResult(BaseModel):
    """Results of outline reflection"""

    meets_requirements: bool = Field(
        description="Whether it meets the paper writing requirements"
    )
    reasons: Optional[List[str]] = Field(
        default=None, description="Reasons for not meeting requirements"
    )


class ContentSearchQuery(BaseModel):
    """Search query for a specific content point"""

    content_point: str = Field(description="The specific content point")
    query: str = Field(description="Search query for this content point")


class SectionSearchQuery(BaseModel):
    """Search queries for an outline section"""

    section_title: str = Field(description="Title of the section")
    content_queries: List[ContentSearchQuery] = Field(
        description="List of content-specific search queries for this section"
    )
    importance: float = Field(
        description="Importance score of this section", ge=0, le=1
    )
    is_conclusion: bool = Field(
        description="Whether this section is a conclusion section"
    )


class State(BaseModel):
    """State container for the workflow"""

    user_query: str
    research_field: Optional[QueryIntent] = None
    outlines: Optional[List[PaperOutline]] = None
    synthesized_outline: Optional[PaperOutline] = None
    reflection_result: Optional[ReflectionResult] = None
    reflection_count: int = 0
    max_reflections: int = 3
    max_sections: int = 4
    min_depth: int = 2
    final_outline: Optional[PaperOutline] = None
    outline_with_queries: Optional[Dict[str, List[SectionSearchQuery]]] = None
    final_result: Optional[Dict[str, Any]] = None
    error_log: List[str] = Field(default_factory=list)

    def update(self, **kwargs) -> "State":
        """Create a new state with updated values"""
        return State(**{**self.model_dump(), **kwargs})

    def log_error(self, message: str) -> "State":
        """Log an error message to the state"""
        logger.error(message)
        return self.update(error_log=self.error_log + [message])


@lru_cache(maxsize=128)
def is_conclusion_reference_section(section_title: str) -> bool:
    """
    Determine if a section is a conclusion section based on its title

    Args:
        section_title: The title of the section

    Returns:
        Boolean indicating whether the section is a conclusion section
    """
    section_title_lower = section_title.lower()

    # First check our cached keyword set
    for keyword in CONCLUSION_KEYWORDS:
        if keyword in section_title_lower:
            logger.info(f"Section '{section_title}' identified as a conclusion section")
            return True

    # Fallback to model-based prediction
    logger.info(
        f"Using model to determine if '{section_title}' is a conclusion section"
    )
    return is_conclusion_reference_section_model_based(section_title)


def is_conclusion_reference_section_model_based(section_title: str) -> bool:
    """
    Use a language model to determine if a section is a conclusion section.

    Args:
        section_title: The title of the section

    Returns:
        Boolean indicating whether the section is a conclusion section
    """
    prompt = get_outline_conclusion_judge_prompt(section_title)
    chain = prompt | default_llm | StrOutputParser()

    result = (
        safe_invoke(
            chain,
            {},
            "false",  # Default to non-conclusion if there's an error
            f"Failed to determine if '{section_title}' is a conclusion section",
            2,
        )
        .strip()
        .lower()
    )

    if result == "true":
        logger.info(
            f"Section '{section_title}' identified as a conclusion section (model prediction)"
        )
        return True
    elif result == "false":
        logger.info(
            f"Section '{section_title}' is not a conclusion section (model prediction)"
        )
        return False
    else:
        logger.warning(
            f"Unexpected model output for section '{section_title}': {result}"
        )
        return False


def determine_research_field(state: State) -> State:
    """
    Determine research field and paper type from user query

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with research field
    """
    logger.info("Starting determine_research_field step")
    parser = PydanticOutputParser(pydantic_object=QueryIntent)

    prompt = get_research_field_prompt(
        state.user_query, parser.get_format_instructions()
    )

    chain = prompt | default_llm | parser

    research_field = safe_invoke(
        chain,
        {},
        QueryIntent(
            research_field="None",  # Safe default field if parsing fails
            paper_type="None",
            topic=state.user_query,
            explanation="None",
        ),
        "Failed to determine research field",
        2,
    )

    logger.info(f"Research field determined: {research_field}")
    return state.update(research_field=research_field)


def generate_single_outline(
    model_name: str,
    field: QueryIntent,
    user_query: str,
    parser: PydanticOutputParser,
    seed_outline: List[dict],
    max_sections: int = 4,
    min_depth: int = 2,
) -> PaperOutline:
    """
    Generate a single paper outline using specified model with control over section structure

    Args:
        model_name: Name of the LLM to use for generation
        field: Research field information
        user_query: Original user query
        parser: Output parser for structured results
        max_sections: Maximum number of top-level sections to include
        min_depth: Minimum depth of subsection hierarchy (1=sections only, 2=sections with subsections, etc.)

    Returns:
        A structured paper outline
    """
    logger.info(
        f"Generating outline with model: {model_name} (max_sections={max_sections}, min_depth={min_depth})"
    )

    # Add section control parameters to the prompt
    prompt = get_outline_generation_prompt_v2(
        field.research_field,
        field.paper_type,
        field.topic,
        user_query,
        parser.get_format_instructions(),
        max_sections=max_sections,
        min_depth=min_depth,
        seed_outline=seed_outline,
    )

    model = llm_map.get(model_name, default_llm)
    chain = prompt | model | parser

    try:
        outline = safe_invoke(
            chain, {}, None, f"Error generating outline with {model_name}", 3
        )

        if outline is None:
            # If we failed to generate an outline, fall back to a more structured approach
            return _generate_fallback_outline(
                model_name, field, user_query, max_sections, min_depth
            )

        # Validate and enforce structural constraints
        outline = _enforce_outline_constraints(
            outline, max_sections, min_depth, model_name, field, user_query
        )

        logger.info(f"Outline generated with {model_name}: {outline.title}")
        return outline

    except Exception as e:
        logger.error(f"Error generating outline with {model_name}: {str(e)}")
        return _generate_fallback_outline(
            model_name, field, user_query, max_sections, min_depth
        )


def _enforce_outline_constraints(
    outline: PaperOutline,
    max_sections: int,
    min_depth: int,
    model_name: str,
    field: QueryIntent,
    user_query: str,
) -> PaperOutline:
    """
    Enforce structural constraints on the outline

    Args:
        outline: The generated outline
        max_sections: Maximum number of top-level sections
        min_depth: Minimum depth of subsection hierarchy
        model_name: Name of the LLM to use for revisions if needed
        field: Research field information
        user_query: Original user query

    Returns:
        Updated outline that meets constraints
    """
    # Validate section count
    if len(outline.sections) > max_sections:
        logger.warning(
            f"Outline exceeded max_sections ({len(outline.sections)} > {max_sections}), truncating"
        )
        outline.sections = outline.sections[:max_sections]

    # Ensure minimum subsection depth is met
    if min_depth > 1:
        has_sufficient_depth = any(section.subsections for section in outline.sections)

        if not has_sufficient_depth:
            logger.warning(
                f"Outline did not meet minimum depth requirement, requesting revision"
            )
            return _generate_fallback_outline(
                model_name, field, user_query, max_sections, min_depth
            )
    return outline


def _generate_fallback_outline(
    model_name: str,
    field: QueryIntent,
    user_query: str,
    max_sections: int,
    min_depth: int,
) -> PaperOutline:
    """
    Generate a fallback outline with a more structured approach

    Args:
        model_name: Name of the LLM to use for generation
        field: Research field information
        user_query: Original user query
        max_sections: Maximum number of top-level sections
        min_depth: Minimum depth of subsection hierarchy

    Returns:
        A structured paper outline
    """
    logger.info(f"Generating fallback outline with model: {model_name}")
    model = llm_map.get(model_name, default_llm)

    # Use a stricter prompt that emphasizes structure
    structured_prompt = get_outline_generation_strick_prompt(
        field.research_field,
        field.paper_type,
        field.topic,
        f"{user_query} (Generate outline with maximum {max_sections} top-level sections and minimum {min_depth} levels of depth)",
    )

    fallback_chain = structured_prompt | model | JsonOutputParser()

    try:
        result = safe_invoke(
            fallback_chain,
            {},
            {
                "title": f"Paper on {field.topic}",
                "abstract": f"This paper discusses {field.topic}.",
                "sections": [
                    {
                        "title": "Introduction",
                        "key_points": ["Background", "Motivation"],
                        "subsections": None,
                    }
                ],
            },
            f"Error generating fallback outline with {model_name}",
            3,
        )

        # Convert to PaperOutline
        return PaperOutline(**result)

    except ValidationError as e:
        logger.error(f"Failed to validate fallback outline: {e}")
        # Create a minimal valid outline as last resort
        return PaperOutline(
            title=f"Paper on {field.topic}",
            abstract=f"This paper discusses {field.topic}.",
            sections=[
                OutlineSection(
                    title="Introduction",
                    key_points=["Background on " + field.topic, "Research motivation"],
                    subsections=None,
                ),
                OutlineSection(
                    title="Literature Review",
                    key_points=["Recent developments", "Current state of research"],
                    subsections=None,
                ),
                OutlineSection(
                    title="Conclusion",
                    key_points=["Summary of findings", "Future directions"],
                    subsections=None,
                ),
            ],
        )


def generate_outlines(state: State) -> State:
    """
    Generate multiple paper outlines using different models

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with generated outlines
    """
    logger.info("Starting generate_outlines step")
    research_field = state.research_field
    models = OUTLINE_GENERAOR_MODELS
    parser = PydanticOutputParser(pydantic_object=PaperOutline)

    seed_outline = get_seed_outline_template(research_field.paper_type)

    # Create a partial function with common parameters
    outline_generator = partial(
        generate_single_outline,
        field=research_field,
        user_query=state.user_query,
        parser=parser,
        seed_outline=seed_outline,
        max_sections=state.max_sections,
        min_depth=state.min_depth,
    )

    outlines = []
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        future_to_model = {
            executor.submit(outline_generator, model): model for model in models
        }

        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                outline = future.result()
                outlines.append(outline)
                logger.info(f"Successfully generated outline with model: {model}")
            except Exception as e:
                logger.error(f"Failed to generate outline with model {model}: {e}")
                # We continue despite failures to ensure we get at least some outlines

    if not outlines:
        # If all models failed, create a fallback outline
        logger.warning(
            "All models failed to generate outlines. Creating a fallback outline."
        )
        fallback_outline = _generate_fallback_outline(
            DEFAULT_MODEL_FOR_OUTLINE,
            field,
            state.user_query,
            state.max_sections,
            state.min_depth,
        )
        outlines = [fallback_outline]

    logger.info(f"Generated {len(outlines)} outlines")
    return state.update(outlines=outlines)


def synthesize_outlines(state: State) -> State:
    """
    Synthesize multiple outlines into one optimal outline

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with synthesized outline
    """
    logger.info("Starting synthesize_outlines step")
    outlines = state.outlines

    if len(outlines) == 1:
        logger.info("Only one outline available, skipping synthesis step")
        return state.update(synthesized_outline=outlines[0])

    parser = PydanticOutputParser(pydantic_object=PaperOutline)

    # Convert outlines to JSON format for prompt
    outlines_json = json.dumps(
        [o.model_dump() for o in outlines], ensure_ascii=False, indent=2
    )

    prompt = get_outline_synthesis_prompt(
        outlines_json,
        state.research_field.research_field,
        state.research_field.paper_type,
        state.research_field.topic,
        parser.get_format_instructions(),
    )

    chain = prompt | default_llm | parser

    synthesized_outline = safe_invoke(
        chain,
        {},
        outlines[0],  # Fall back to first outline if synthesis fails
        "Failed to synthesize outlines",
        3,
    )

    logger.info(f"Synthesized outline created: {synthesized_outline.title}")
    return state.update(synthesized_outline=synthesized_outline)


def reflection(state: State) -> State:
    """
    Review the synthesized outline and provide feedback

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with reflection results
    """
    logger.info("Starting reflection step")
    synthesized_outline = state.synthesized_outline
    parser = PydanticOutputParser(pydantic_object=ReflectionResult)

    prompt = get_outline_reflection_prompt_v2(
        state.user_query,
        state.research_field.research_field,
        state.research_field.paper_type,
        state.research_field.topic,
        json.dumps(synthesized_outline.model_dump(), ensure_ascii=False, indent=2),
        parser.get_format_instructions(),
    )

    chain = prompt | default_llm | parser

    reflection_result = safe_invoke(
        chain,
        {},
        ReflectionResult(
            meets_requirements=True, reasons=None
        ),  # Default to positive reflection if parsing fails
        "Failed to parse reflection result",
        3,
    )

    logger.info(
        f"Reflection completed: meets_requirements={reflection_result.meets_requirements}"
    )

    return state.update(
        reflection_result=reflection_result, reflection_count=state.reflection_count + 1
    )


def improve_outline(state: State) -> State:
    """
    Improve the outline based on reflection feedback

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with improved outline
    """
    logger.info("Starting improve_outline step")
    reflection_result = state.reflection_result
    current_outline = state.synthesized_outline
    parser = PydanticOutputParser(pydantic_object=PaperOutline)

    reasons = (
        reflection_result.reasons
        if reflection_result.reasons
        else ["Needs further improvement"]
    )

    prompt = get_outline_improve_prompt_v2(
        state.user_query,
        state.research_field.research_field,
        state.research_field.paper_type,
        state.research_field.topic,
        json.dumps(current_outline.model_dump(), ensure_ascii=False, indent=2),
        json.dumps(reasons, ensure_ascii=False, indent=2),
        parser.get_format_instructions(),
    )

    chain = prompt | default_llm | parser

    improved_outline = safe_invoke(
        chain,
        {},
        current_outline,  # Fall back to current outline if improvement fails
        "Failed to improve outline",
        3,
    )

    logger.info(f"Outline improved: {improved_outline.title}")
    return state.update(synthesized_outline=improved_outline)


def should_continue_improvement(state: State) -> str:
    """
    Decision logic: determine if we should continue improvement or finalize

    Args:
        state: Current workflow state

    Returns:
        Decision string ("finalize" or "improve")
    """
    logger.info(
        f"Checking improvement continuation: reflection_count={state.reflection_count}, max_reflections={state.max_reflections}"
    )

    if state.reflection_result.meets_requirements:
        logger.info("Requirements met, finalizing outline")
        return "finalize"
    elif state.reflection_count >= state.max_reflections:
        logger.info("Max reflections reached, finalizing outline")
        return "finalize"
    else:
        logger.info("Continuing with improvement")
        return "improve"


def finalize_outline(state: State) -> State:
    """
    Finalize the outline

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with final outline
    """
    logger.info("Finalizing outline")
    return state.update(final_outline=state.synthesized_outline)


def generate_content_query(
    content_point: str,
    section_title: str,
    paper_title: str,
    max_retries: int = MAX_RETRY_ATTEMPTS,
) -> ContentSearchQuery:
    """
    Generate a search query for a specific content point

    Args:
        content_point: The specific content point
        section_title: Title of the containing section
        paper_title: Title of the paper
        max_retries: Maximum number of retry attempts

    Returns:
        A ContentSearchQuery object
    """
    for attempt in range(max_retries):
        try:
            prompt = get_query_generation_prompt_v2(
                paper_title, section_title, content_point
            )
            chain = prompt | llm_map[MODEL_GEN_QUERY] | StrOutputParser()
            query = chain.invoke({})

            if not query or len(query.strip()) < 10:
                raise ValueError("Generated query is too short or empty")

            logger.info(f"Generated query for content point '{content_point}': {query}")
            return ContentSearchQuery(content_point=content_point, query=query)

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt+1} failed for content point '{content_point}': {str(e)}. Retrying..."
                )
                time.sleep(1)  # Add a small delay between retries
            else:
                logger.error(
                    f"Failed to generate query for '{content_point}': {str(e)}"
                )
                # Fall back to using the content point itself as query
                fallback_query = f"{paper_title} {section_title} {content_point}"
                logger.info(
                    f"Using fallback query for '{content_point}': {fallback_query}"
                )
                return ContentSearchQuery(
                    content_point=content_point, query=fallback_query
                )


def generate_section_queries(
    outline_section: OutlineSection,
    paper_title: str,
    parent_title: str = "",
    max_workers: int = QUERY_GENERATION_CONCURRENCY,
) -> List[SectionSearchQuery]:
    """
    Generate search queries recursively for sections and subsections

    Args:
        outline_section: The section to process
        paper_title: Title of the paper
        parent_title: Title of the parent section (if any)
        max_workers: Maximum number of concurrent workers

    Returns:
        List of section search queries
    """
    logger.info(f"Processing section: {outline_section.title}")
    results = []

    # Check if this is a conclusion section
    is_conclusion = is_conclusion_reference_section(outline_section.title)

    if is_conclusion:
        logger.info(
            f"Skipping query generation for conclusion/reference section: {outline_section.title}"
        )
        results.append(
            SectionSearchQuery(
                section_title=outline_section.title,
                content_queries=[],
                importance=CONCLUSION_IMPORTANCE_SCORE,
                is_conclusion=True,
            )
        )
        return results

    # Process based on section structure
    if outline_section.subsections:
        # If subsections exist, process them
        logger.info(f"Section {outline_section.title} has subsections, processing them")
        for subsection in outline_section.subsections:
            subsection_queries = generate_section_queries(
                subsection,
                paper_title,
                parent_title=outline_section.title,
                max_workers=max_workers,
            )
            results.extend(subsection_queries)
    else:
        # If no subsections, generate queries for current section content
        logger.info(f"Generating queries for leaf section: {outline_section.title}")
        content_queries = []

        if outline_section.key_points:
            # Process key points in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a mapping of futures to key points
                future_to_content = {}
                for key_point in outline_section.key_points:
                    future = executor.submit(
                        generate_content_query,
                        key_point,
                        outline_section.title,
                        paper_title,
                    )
                    future_to_content[future] = key_point

                # Process results as they complete
                for future in as_completed(future_to_content):
                    key_point = future_to_content[future]
                    try:
                        content_query = future.result()
                        content_queries.append(content_query)
                        logger.debug(f"Generated query for content point: {key_point}")
                    except Exception as e:
                        logger.error(
                            f"Error processing content point '{key_point}': {e}"
                        )
                        # Add a fallback query
                        fallback_query = ContentSearchQuery(
                            content_point=key_point,
                            query=f"{paper_title} {outline_section.title} {key_point}",
                        )
                        content_queries.append(fallback_query)

        # Add current section's queries
        results.append(
            SectionSearchQuery(
                section_title=outline_section.title,
                content_queries=content_queries,
                importance=REGULAR_SECTION_IMPORTANCE_SCORE,
                is_conclusion=False,
            )
        )

    return results


def generate_search_queries(state: State) -> State:
    """
    Generate search queries for each section of the final outline

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with search queries
    """
    logger.info("Starting generate_search_queries step")
    outline = state.final_outline
    paper_title = outline.title
    search_queries = {}

    try:
        # Generate queries for each main section
        for section in outline.sections:
            section_queries = generate_section_queries(section, paper_title)
            search_queries[section.title] = section_queries

        logger.info(f"Generated search queries for {len(search_queries)} sections")

    except Exception as e:
        error_msg = f"Error generating search queries: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        # Return state with error, but don't fail the entire workflow
        return state.log_error(error_msg)

    return state.update(outline_with_queries=search_queries)


def _organize_section_hierarchy(
    section_queries: List[SectionSearchQuery],
) -> Dict[str, Dict]:
    """
    组织章节层次结构，识别主章节和子章节的关系

    Args:
        section_queries: 章节查询列表

    Returns:
        章节层次结构映射
    """
    # 获取主章节标题（默认第一个查询的章节）
    main_section_title = section_queries[0].section_title if section_queries else ""

    # 初始化层次结构映射
    hierarchy = {}

    # 为每个章节创建初始结构
    for query in section_queries:
        title = query.section_title
        hierarchy[title] = {
            "parent": None if title == main_section_title else main_section_title,
            "level": 0 if title == main_section_title else 1,
            "content_queries": query.content_queries,
            "importance": query.importance,
            "is_conclusion": query.is_conclusion,
        }

    return hierarchy


def _build_hierarchical_subsections(
    section_info_process: Dict,
    section_hierarchy: Dict = None,
    parent_structure: Dict[str, Dict] = None,
    depth: int = 0,
) -> Dict[str, Dict]:
    """
    递归构建层次化的子章节结构

    Args:
        section_info_process: 处理过的章节信息
        section_hierarchy: 章节层次结构映射
        parent_structure: 父级结构字典（用于递归）
        depth: 当前递归深度

    Returns:
        层次化的子章节结构
    """
    if parent_structure is None:
        parent_structure = {}

    # 处理每个子章节
    for key, value in section_info_process.items():
        subsection_key_points = []

        # 提取子章节关键点
        if (
            "subsection_key_point_and_search_query_pairs" in value
            and value["subsection_key_point_and_search_query_pairs"]
        ):
            subsection_key_points = [
                item["subsection_key_point"]
                for item in value["subsection_key_point_and_search_query_pairs"]
            ]

        # 创建子章节结构
        subsection_info = {
            "section_index": value.get("subsection_index"),
            "section_title": value.get("subsection_title"),
            "is_conclusion": False,  # 子章节通常不是结论
            "key_points": subsection_key_points,
            "subsection_info": {},  # 初始化空的子章节字典
        }

        # 递归处理更深层次的子章节（如果存在）
        if "deeper_subsections" in value and value["deeper_subsections"]:
            subsection_info["subsection_info"] = _build_hierarchical_subsections(
                value["deeper_subsections"], section_hierarchy, {}, depth + 1
            )

        parent_structure[key] = subsection_info

    return parent_structure


def _organize_deeper_subsections(section_info: Dict, section_hierarchy: Dict) -> Dict:
    """
    根据层次结构组织章节的deeper_subsections

    Args:
        section_info: 章节信息字典
        section_hierarchy: 章节层次结构映射

    Returns:
        组织后的章节信息
    """
    # 复制一份，避免修改原始数据
    result = section_info.copy()

    # 识别子章节并将其移动到父章节的deeper_subsections中
    to_remove = []

    for title, info in section_hierarchy.items():
        parent = info.get("parent")
        if parent and parent in result and title in result:
            # 将子章节添加到父章节的deeper_subsections中
            parent_info = result[parent]
            if "deeper_subsections" not in parent_info:
                parent_info["deeper_subsections"] = {}

            parent_info["deeper_subsections"][title] = result[title]
            to_remove.append(title)  # 标记应从顶层移除的子章节

    # 从顶层移除已移动到deeper_subsections的子章节
    for title in to_remove:
        if title in result:
            del result[title]

    return result


def _extract_subsection_info(
    section_queries: List[SectionSearchQuery],
) -> Tuple[Dict, List[str], List[str], float, bool]:
    """
    Extract and organize subsection information from section queries, supporting multiple levels

    Args:
        section_queries: List of section search queries

    Returns:
        Tuple containing:
            - Dictionary of processed section info
            - List of key points
            - List of search queries
            - Importance score
            - Is conclusion flag
    """
    section_info_process = {}
    key_points = []
    search_queries = []
    importance = REGULAR_SECTION_IMPORTANCE_SCORE
    is_conclusion = False

    # 构建章节层次结构映射
    section_hierarchy = _organize_section_hierarchy(section_queries)

    # 获取主章节标题（默认第一个查询的章节）
    main_section_title = section_queries[0].section_title if section_queries else ""

    # 处理每个章节的查询
    for subsection_index, subsection_info in enumerate(section_queries):
        subsection_title = subsection_info.section_title
        temp = []

        # 更新章节标志
        importance = max(importance, subsection_info.importance)
        is_conclusion = is_conclusion or subsection_info.is_conclusion

        # 收集所有内容点和搜索查询
        for content_query in subsection_info.content_queries:
            key_points.append(content_query.content_point)
            search_queries.append(content_query.query)

            # 根据是否为主章节构建不同的数据结构
            if subsection_title != main_section_title:
                temp.append(
                    {
                        "parent_section": subsection_title,
                        "subsection_key_point": content_query.content_point,
                        "subsection_search_query": content_query.query,
                    }
                )

        # 处理章节信息
        if subsection_title == main_section_title:
            # 主章节的内容点分别创建子章节条目
            for idx, content_query in enumerate(subsection_info.content_queries):
                section_info_process[content_query.content_point] = {
                    "subsection_index": idx,
                    "subsection_title": content_query.content_point,
                    "subsection_key_point_and_search_query_pairs": [],
                    "deeper_subsections": {},  # 支持更深层次的子章节
                }
        else:
            # 非主章节，创建包含所有内容点的条目
            section_info_process[subsection_title] = {
                "subsection_index": subsection_index,
                "subsection_title": subsection_title,
                "subsection_key_point_and_search_query_pairs": temp,
                "deeper_subsections": {},  # 支持更深层次的子章节
            }

    # 根据层次结构将章节组织成树状结构
    organized_section_info = _organize_deeper_subsections(
        section_info_process, section_hierarchy
    )

    return organized_section_info, key_points, search_queries, importance, is_conclusion


def prepare_final_result(state: State) -> State:
    """
    Prepare the final result with all components

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with final result
    """
    logger.info("Preparing final result")

    try:
        sections_with_index = {}
        final_sections = {}

        # Process each section
        for i, (section_title, section_queries) in enumerate(
            state.outline_with_queries.items()
        ):
            try:
                # Extract and organize section information
                (
                    section_info_process,
                    key_points,
                    search_queries,
                    importance,
                    is_conclusion,
                ) = _extract_subsection_info(section_queries)

                # Format for section_writer consumption
                processed_section = {
                    "section_index": i,
                    "section_title": section_title,
                    "key_points": key_points,
                    "search_queries": search_queries,
                    "importance": importance,
                    "is_conclusion": is_conclusion,
                    "outline_subsection_info": section_info_process,
                }

                sections_with_index[section_title] = processed_section

                # 使用递归方法构建层次化的子章节结构
                subsections = _build_hierarchical_subsections(
                    section_info_process, _organize_section_hierarchy(section_queries)
                )

                final_sections[section_title] = {
                    "section_index": i,
                    "section_title": section_title,
                    "is_conclusion": is_conclusion,
                    "key_points": key_points,
                    "subsection_info": subsections,
                }

            except Exception as e:
                error_msg = f"Error processing section {section_title}: {str(e)}"
                logger.error(error_msg)
                state = state.log_error(error_msg)

        final_result = {
            "user_query": state.user_query,
            "title": state.final_outline.title,
            "abstract": state.final_outline.abstract,
            "research_field": state.research_field.model_dump(),
            "final_outline": {
                "title": state.final_outline.title,
                "sections": final_sections,
            },
            "reflection_count": state.reflection_count,
            "meets_requirements": state.reflection_result.meets_requirements,
            "outline_with_query": sections_with_index,
            "errors": state.error_log if state.error_log else None,
        }

        return state.update(final_result=final_result)

    except Exception as e:
        error_msg = f"Error preparing final result: {traceback.format_exc()}"
        logger.error(error_msg)

        # Create a minimal valid result structure
        minimal_result = {
            "user_query": state.user_query,
            "title": state.final_outline.title if state.final_outline else "Outline",
            "abstract": state.final_outline.abstract if state.final_outline else "",
            "research_field": (
                state.research_field.model_dump() if state.research_field else {}
            ),
            "final_outline": {
                "title": (
                    state.final_outline.title if state.final_outline else "Outline"
                ),
                "sections": {},
            },
            "error": f"Failed to prepare complete result: {str(e)}",
            "errors": state.error_log + [error_msg] if state.error_log else [error_msg],
        }

        return state.update(final_result=minimal_result)


# ======================================
# Workflow Construction
# ======================================


def build_paper_outline_workflow(determain_intent=True) -> Any:
    """
    Build and compile the workflow graph

    Returns:
        Compiled workflow
    """
    logger.info("Building paper outline workflow")
    workflow = StateGraph(State)

    # Add nodes
    if determain_intent:
        workflow.add_node("determine_research_field", determine_research_field)
    workflow.add_node("generate_outlines", generate_outlines)
    workflow.add_node("synthesize_outlines", synthesize_outlines)
    workflow.add_node("reflection", reflection)
    workflow.add_node("improve_outline", improve_outline)
    workflow.add_node("finalize_outline", finalize_outline)
    workflow.add_node("generate_search_queries", generate_search_queries)
    workflow.add_node("prepare_final_result", prepare_final_result)

    # Define linear flow
    if determain_intent:
        workflow.add_edge("determine_research_field", "generate_outlines")
    workflow.add_edge("generate_outlines", "synthesize_outlines")
    workflow.add_edge("synthesize_outlines", "reflection")

    # Conditional branches based on reflection results
    workflow.add_conditional_edges(
        "reflection",
        should_continue_improvement,
        {"improve": "improve_outline", "finalize": "finalize_outline"},
    )

    workflow.add_edge("improve_outline", "reflection")
    workflow.add_edge("finalize_outline", "generate_search_queries")
    workflow.add_edge("generate_search_queries", "prepare_final_result")
    workflow.add_edge("prepare_final_result", END)

    # Set entry point
    if determain_intent:
        workflow.set_entry_point("determine_research_field")
    else:
        workflow.set_entry_point("generate_outlines")

    logger.info("Workflow built successfully")

    return workflow.compile()


# ======================================
# Public API Functions
# ======================================


def generate_paper_outline(
    user_query: str,
    query_intent: dict = None,
    max_reflections: int = 3,
    max_sections: int = 4,
    min_depth: int = 2,
) -> Dict[str, Any]:
    """
    Generate an academic paper outline based on user query with search queries

    Args:
        user_query: User query
        query_intent: Optional pre-analyzed query intent information
        max_reflections: Maximum number of reflection cycles
        max_sections: Maximum number of top-level sections
        min_depth: Minimum depth of subsection hierarchy

    Returns:
        Result containing the final paper outline and search queries
    """
    logger.info(
        f"Starting paper outline generation for query: {user_query} "
        f"query_intent={query_intent} "
        f"(max_sections={max_sections}, min_depth={min_depth}, max_reflections={max_reflections})"
    )
    try:
        # Build a modified workflow if we have pre-analyzed query intent
        if query_intent:
            logger.info(f"Using pre-analyzed query intent: {query_intent}")
            workflow = build_paper_outline_workflow(False)

            # Create research field from query intent
            research_field = QueryIntent(
                research_field=query_intent.get("research_field", "Computer Science"),
                paper_type=query_intent.get("paper_type", "research"),
                topic=query_intent.get("topic", user_query),
                explanation=query_intent.get("explanation", ""),
            )

            initial_state = State(
                user_query=user_query,
                research_field=research_field,
                max_reflections=max_reflections,
                max_sections=max_sections,
                min_depth=min_depth,
            )
        else:
            # Use the standard workflow if no pre-analyzed intent
            logger.info("do not use pre-analyzed query intent, using default workflow")
            workflow = build_paper_outline_workflow()

            research_field = QueryIntent(
                research_field=query_intent.get("research_field", "Computer Science"),
                paper_type=query_intent.get("paper_type", "research"),
                topic=query_intent.get("topic", user_query),
                explanation=query_intent.get("explanation", ""),
            )

            initial_state = State(
                user_query=user_query,
                max_reflections=max_reflections,
                max_sections=max_sections,
                min_depth=min_depth,
            )

        result = workflow.invoke(initial_state)

        if not result or "final_result" not in result or not result["final_result"]:
            logger.error("Workflow completed but no final result was produced")
            return {
                "user_query": user_query,
                "error": "Failed to generate paper outline",
                "title": f"Outline for {user_query}",
                "abstract": "",
            }

        logger.info("Paper outline generation completed successfully")
        return result["final_result"]

    except Exception as e:
        logger.error(f"Error in generate_paper_outline: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "user_query": user_query,
            "error": f"Failed to generate paper outline: {str(e)}",
            "title": f"Outline for {user_query}",
            "abstract": "",
        }


async def generate_paper_outline_async(
    user_query: str,
    query_intent: dict = None,
    max_reflections: int = 3,
    max_sections: int = 4,
    min_depth: int = 2,
) -> Dict[str, Any]:
    """
    Asynchronous version of paper outline generation based on user query with search queries

    Args:
        user_query: User query
        query_intent: Optional pre-analyzed query intent information
        max_reflections: Maximum number of reflection cycles
        max_sections: Maximum number of top-level sections
        min_depth: Minimum depth of subsection hierarchy

    Returns:
        Result containing the final paper outline and search queries
    """
    logger.info(f"Starting async paper outline generation for query: {user_query}")

    try:
        # Run the synchronous function in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_paper_outline(
                user_query, query_intent, max_reflections, max_sections, min_depth
            ),
        )

        logger.info("Async paper outline generation completed")
        return result

    except Exception as e:
        logger.error(f"Error in generate_paper_outline_async: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "user_query": user_query,
            "error": f"Failed to generate paper outline asynchronously: {str(e)}",
            "title": f"Outline for {user_query}",
            "abstract": "",
        }


async def run_async_example():
    """Example usage of the asynchronous API"""
    user_query_async = "recent advances in natural language processing"
    logger.info(f"Running asynchronous example with query: {user_query_async}")

    result_async = await generate_paper_outline_async(user_query_async)

    logger.info("Writing result to file")
    with open("temp/paper_outline_async.json", "w") as f:
        json.dump(result_async, f, ensure_ascii=False, indent=2)

    logger.info("Async example completed")


# Example usage
if __name__ == "__main__":
    pass
    # Example 1: Synchronous usage
    # user_query_sync = "current research on LLM Agent"
    # logger.info(f"Running synchronous example with query: {user_query_sync}")
    # result_sync = generate_paper_outline(user_query_sync)

    # # Output results
    # with open("temp/paper_outline_sync.json", "w") as f:
    #     json.dump(result_sync, f, ensure_ascii=False, indent=2)

    # Example 2: Asynchronous usage
    # asyncio.run(run_async_example())
