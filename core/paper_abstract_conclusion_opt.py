#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Generation and reflection of abstract and conclusion
# ==================================================================
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
from log import logger
from model_factory import llm_map
import traceback
from utils import prepare_sections_data
from prompt_manager import (
    get_abstract_prompt,
    get_conclusion_prompt,
    get_abstract_conclusion_evaluation_prompt,
    get_conclusion_section_content_prompt,
    get_section_summary_intro_prompt,
)
from configuration import (
    GLOBAL_ABSTRACT_CONCLUSION_MAX_TURNS,
    MODEL_GEN_ABSTRACT_CONCLUSION,
)
from langgraph.graph import StateGraph, END


class AbstractConclusionResult(BaseModel):
    """Model for abstract and conclusion generation results."""

    abstract: str = Field(description="Generated paper abstract")
    conclusion: str = Field(description="Generated paper conclusion")
    meets_requirements: bool = Field(description="Whether both meet requirements")
    feedback: Optional[str] = Field(
        default=None, description="Feedback for improvement if needed"
    )


class AbstractConclusionState(BaseModel):
    """State container for the abstract/conclusion workflow."""

    user_query: str = Field(description="Original user query")
    paper_title: str = Field(description="Paper title")
    outline: Dict[str, Any] = Field(description="Paper outline structure")
    summaries: List[dict] = Field(description="Section summaries")
    abstract: Optional[str] = Field(default=None, description="Generated abstract")
    conclusion: Optional[str] = Field(default=None, description="Generated conclusion")
    reflection_count: int = Field(default=0, description="Current reflection iteration")
    max_reflections: int = Field(default=3, description="Maximum reflection cycles")
    feedback: Optional[str] = Field(default=None, description="Current feedback")
    meets_requirements: bool = Field(
        default=False, description="Whether requirements are met"
    )
    evaluate_required: bool = Field(
        default=True, description="Whether to evaluate abstract and conclusion"
    )


class GenerationError(Exception):
    """Exception raised when generation fails after multiple retries."""

    pass


async def generate_section_introduction_content(
    paper_title: str,
    user_query: str,
    section: Dict[str, Any],
    llm: Any,  # Language model instance
) -> str:
    """Generates an introductory paragraph for a given section based on its key points."""
    section_name = section.get("name", "Unnamed Section")
    key_points = section.get("key_points", [])

    if not key_points:
        logger.info(
            f"No key points found for section '{section_name}', skipping introduction generation."
        )
        return ""

    # Format key_points for the prompt get_section_summary_intro_prompt
    # The prompt expects key_points_info: "A string containing summarized information from the section's key points."
    key_points_list_str = "\n".join([f"- {kp}" for kp in key_points])
    key_points_info_for_prompt = f"""Key points to be covered in section '{section_name}': {key_points_list_str}"""

    logger.info(f"Generating introduction for section: {section_name}")

    try:
        prompt = get_section_summary_intro_prompt(
            paper_title=paper_title,
            user_query=user_query,
            section_name=section_name,
            key_points_info=key_points_info_for_prompt,
        )

        chain = prompt | llm
        result = await chain.ainvoke({})
        introduction_content = result.content.strip()

        if (
            not introduction_content or len(introduction_content) < 20
        ):  # Basic validation
            logger.warning(
                f"Generated introduction for section '{section_name}' is too short or empty ({len(introduction_content)} chars)."
            )
            return ""

        logger.info(
            f"Successfully generated introduction for section: {section_name} ({len(introduction_content)} chars)"
        )
        return introduction_content

    except Exception as e:
        logger.error(
            f"Error generating introduction for section {section_name}: {str(e)}"
        )
        logger.error(traceback.format_exc())
        return ""


async def generate_conclusion_section_content(
    state: dict, conclusion_sections: List[dict]
) -> Dict[str, str]:
    """Generate content for conclusion-type sections based on paper content.

    Args:
        state: Current state containing paper information and summaries
        conclusion_sections: List of conclusion-related sections identified in the outline

    Returns:
        Dictionary mapping section IDs to generated content
    """
    logger.info(
        f"Generating content for {len(conclusion_sections)} conclusion-related sections"
    )
    conclusion_sections_processed = {}

    for section in conclusion_sections:
        try:
            conclusion_sections_processed[section["section_title"]] = section
            section_id = section.get("section_index")
            section_title = section.get("section_title", "")

            logger.info(
                f"Generating content for conclusion section: {section_title} (ID: {section_id})"
            )

            llm = llm_map[MODEL_GEN_ABSTRACT_CONCLUSION]

            prompt = get_conclusion_section_content_prompt(
                paper_title=state["paper_title"],
                user_query=state["user_query"],
                section_title=section_title,
                section_summaries=state["summaries"],
            )

            chain = prompt | llm
            result = await chain.ainvoke({})
            section_content = result.content.strip()

            if not section_content or len(section_content) < 100:
                logger.error(
                    f"Generated content too short or empty ({len(section_content) if section_content else 0} chars)"
                )
                continue

            conclusion_sections_processed[section["section_title"]][
                "section_text"
            ] = section_content
            logger.info(
                f"Successfully generated {len(section_content)} characters for section: {section_title}"
            )

        except Exception as e:
            logger.error(
                f"Error generating content for conclusion section {section.get('title', 'Unknown')}: {str(e)}"
            )
            logger.error(traceback.format_exc())

    return conclusion_sections_processed


async def generate_abstract(state: AbstractConclusionState) -> AbstractConclusionState:
    """Generate a paper abstract based on section summaries and outline with retry logic."""
    logger.info("Generating paper abstract")

    max_retries = 3
    retry_count = 0
    abstract = None
    errors = []

    while retry_count < max_retries and not abstract:
        try:
            llm = llm_map[MODEL_GEN_ABSTRACT_CONCLUSION]

            prompt = get_abstract_prompt(
                paper_title=state.paper_title,
                user_query=state.user_query,
                outline=state.outline,
                summaries=state.summaries,
            )

            logger.info(
                f"Prompt string for abstract generation (attempt {retry_count+1}/{max_retries})"
            )

            chain = prompt | llm
            result = await chain.ainvoke({})
            abstract = result.content.strip()

            # Validate abstract
            if not abstract:
                raise ValueError("Empty abstract returned")

            if len(abstract) < 50:
                logger.warning(
                    f"Generated abstract too short ({len(abstract)} chars), retrying..."
                )
                errors.append(
                    f"Attempt {retry_count+1}: Abstract too short ({len(abstract)} chars)"
                )
                abstract = None
                retry_count += 1
                continue

            logger.info(f"Abstract generated successfully ({len(abstract)} characters)")
            return AbstractConclusionState(
                **{**state.model_dump(), "abstract": abstract}
            )

        except Exception as e:
            retry_count += 1
            error_msg = f"Error generating abstract (attempt {retry_count}/{max_retries}): {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            errors.append(error_msg)

            if retry_count >= max_retries:
                logger.error("Max retries reached for abstract generation")
                # Create fallback abstract
                try:
                    fallback_abstract = create_fallback_abstract(
                        state.paper_title, state.user_query, errors
                    )
                    return AbstractConclusionState(
                        **{**state.model_dump(), "abstract": fallback_abstract}
                    )
                except Exception as fallback_error:
                    logger.error(
                        f"Error creating fallback abstract: {str(fallback_error)}"
                    )
                    raise GenerationError(
                        f"Failed to generate abstract after {max_retries} attempts: {errors}"
                    )

            # Wait before retrying with exponential backoff
            await asyncio.sleep(1 * (2 ** (retry_count - 1)))

    return state


def create_fallback_abstract(
    paper_title: str, user_query: str, errors: List[str]
) -> str:
    """Create a fallback abstract when generation fails."""
    logger.warning(f"Creating fallback abstract for '{paper_title}'")

    fallback_text = (
        f"This paper addresses the topic of '{paper_title}'. "
        f"It explores key aspects related to {user_query} through systematic analysis. "
        f"The research examines theoretical foundations, current methodologies, and practical applications. "
        f"Findings suggest important implications for both theory and practice in this domain."
    )

    return fallback_text


async def generate_conclusion(
    state: AbstractConclusionState,
) -> AbstractConclusionState:
    """Generate a paper conclusion based on section summaries and outline with retry logic."""
    logger.info("Generating paper conclusion")

    max_retries = 3
    retry_count = 0
    conclusion = None
    errors = []

    while retry_count < max_retries and not conclusion:
        try:
            llm = llm_map[MODEL_GEN_ABSTRACT_CONCLUSION]
            prompt = get_conclusion_prompt(
                paper_title=state.paper_title,
                user_query=state.user_query,
                abstract=state.abstract,
                outline=state.outline,
                summaries=state.summaries,
            )

            logger.info(
                f"Prompt string for conclusion generation (attempt {retry_count+1}/{max_retries})"
            )

            chain = prompt | llm
            result = await chain.ainvoke({})
            conclusion = result.content.strip()

            # Validate conclusion
            if not conclusion:
                raise ValueError("Empty conclusion returned")

            if len(conclusion) < 100:
                logger.warning(
                    f"Generated conclusion too short ({len(conclusion)} chars), retrying..."
                )
                errors.append(
                    f"Attempt {retry_count+1}: Conclusion too short ({len(conclusion)} chars)"
                )
                conclusion = None
                retry_count += 1
                continue

            logger.info(
                f"Conclusion generated successfully ({len(conclusion)} characters)"
            )
            return AbstractConclusionState(
                **{**state.model_dump(), "conclusion": conclusion}
            )

        except Exception as e:
            retry_count += 1
            error_msg = f"Error generating conclusion (attempt {retry_count}/{max_retries}): {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            errors.append(error_msg)

            if retry_count >= max_retries:
                logger.error("Max retries reached for conclusion generation")
                # Create fallback conclusion
                try:
                    fallback_conclusion = create_fallback_conclusion(
                        state.paper_title, state.abstract, errors
                    )
                    return AbstractConclusionState(
                        **{**state.model_dump(), "conclusion": fallback_conclusion}
                    )
                except Exception as fallback_error:
                    logger.error(
                        f"Error creating fallback conclusion: {str(fallback_error)}"
                    )
                    raise GenerationError(
                        f"Failed to generate conclusion after {max_retries} attempts: {errors}"
                    )

            # Wait before retrying with exponential backoff
            await asyncio.sleep(1 * (2 ** (retry_count - 1)))

    return state


def create_fallback_conclusion(
    paper_title: str, abstract: str, errors: List[str]
) -> str:
    """Create a fallback conclusion when generation fails."""
    logger.warning(f"Creating fallback conclusion for '{paper_title}'")

    fallback_text = (
        f"This paper has presented a comprehensive analysis of topics related to '{paper_title}'. "
        f"The research has explored key theoretical concepts and practical applications in this domain. "
        f"The findings contribute to the existing literature by providing insights into critical aspects "
        f"of the subject matter. Future research should expand on these foundations by addressing "
        f"limitations identified in this work and exploring additional dimensions of the topic."
    )

    return fallback_text


async def evaluate_abstract_conclusion(
    state: AbstractConclusionState,
) -> AbstractConclusionState:
    """Evaluate the quality of generated abstract and conclusion."""
    # Check if evaluation should be skipped
    if not state.evaluate_required:
        logger.info("Evaluation skipped as per configuration")
        return AbstractConclusionState(
            **{
                **state.model_dump(),
                "meets_requirements": True,  # Assume meets requirements when skipping evaluation
                "feedback": "Evaluation skipped as per configuration",
                "reflection_count": state.max_reflections,  # Set to max to ensure workflow completes
            }
        )

    logger.info("Evaluating abstract and conclusion")

    # Validate input state
    if not state.abstract or not state.conclusion:
        logger.warning("Abstract or conclusion missing, cannot evaluate")
        return AbstractConclusionState(
            **{
                **state.model_dump(),
                "meets_requirements": False,
                "feedback": "Abstract or conclusion is missing and needs to be generated.",
                "reflection_count": state.reflection_count + 1,
            }
        )

    max_retries = 2
    retry_count = 0

    while retry_count < max_retries:
        try:
            llm = llm_map[MODEL_GEN_ABSTRACT_CONCLUSION]
            parser = PydanticOutputParser(pydantic_object=AbstractConclusionResult)

            prompt = get_abstract_conclusion_evaluation_prompt(
                paper_title=state.paper_title,
                user_query=state.user_query,
                abstract=state.abstract,
                conclusion=state.conclusion,
                summaries=state.summaries,
                format_instructions=parser.get_format_instructions(),
            )

            chain = prompt | llm | parser
            result = await chain.ainvoke({})

            logger.info(
                f"Evaluation complete: Meets requirements: {result.meets_requirements}"
            )
            return AbstractConclusionState(
                **{
                    **state.model_dump(),
                    "meets_requirements": result.meets_requirements,
                    "feedback": result.feedback,
                    "reflection_count": state.reflection_count + 1,
                }
            )

        except Exception as e:
            retry_count += 1
            logger.error(
                f"Error evaluating abstract/conclusion (attempt {retry_count}/{max_retries}): {str(e)}"
            )
            logger.error(traceback.format_exc())

            if retry_count >= max_retries:
                # Fallback evaluation logic
                logger.warning("Using fallback evaluation logic")
                fallback_meets_requirements = False
                fallback_feedback = f"Evaluation error: {str(e)}. Please review the abstract and conclusion manually."

                if state.reflection_count >= state.max_reflections - 1:
                    # If we're on the last reflection, consider it acceptable to proceed
                    fallback_meets_requirements = True
                    fallback_feedback += (
                        " Auto-approving due to maximum reflection cycles reached."
                    )

                return AbstractConclusionState(
                    **{
                        **state.model_dump(),
                        "meets_requirements": fallback_meets_requirements,
                        "feedback": fallback_feedback,
                        "reflection_count": state.reflection_count + 1,
                    }
                )

            # Wait briefly before retrying
            await asyncio.sleep(1)

    # This should not be reached due to the retry logic, but including for completeness
    return AbstractConclusionState(
        **{
            **state.model_dump(),
            "meets_requirements": False,
            "feedback": "Evaluation failed after multiple attempts.",
            "reflection_count": state.reflection_count + 1,
        }
    )


def should_continue_reflection(state: AbstractConclusionState) -> str:
    """Decision node to determine if more reflection is needed."""
    # If evaluation is disabled, always return the result
    if not state.evaluate_required:
        logger.info("Evaluation disabled, returning result")
        return "return_result"

    logger.info(
        f"Checking if more reflection needed: meets_requirements={state.meets_requirements}, count={state.reflection_count}/{state.max_reflections}"
    )

    if state.meets_requirements:
        logger.info("Requirements met, concluding workflow")
        return "return_result"
    elif state.reflection_count >= state.max_reflections:
        logger.info("Max reflections reached, concluding workflow")
        return "return_result"
    else:
        logger.info("Continue reflection cycle")
        return "regenerate"


def build_abstract_conclusion_workflow():
    """Build and compile the workflow for abstract and conclusion generation."""

    workflow = StateGraph(AbstractConclusionState)

    # Define nodes
    workflow.add_node("generate_abstract", generate_abstract)
    workflow.add_node("generate_conclusion", generate_conclusion)
    workflow.add_node("evaluate", evaluate_abstract_conclusion)

    # Define edges
    workflow.add_edge("generate_abstract", "generate_conclusion")
    workflow.add_edge("generate_conclusion", "evaluate")

    # Add conditional branching based on evaluation
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_reflection,
        {"regenerate": "generate_abstract", "return_result": END},
    )

    # Set entry point
    workflow.set_entry_point("generate_abstract")

    return workflow.compile()


def validate_input_params(params: Dict[str, Any]) -> None:
    """Validate input parameters and raise appropriate exceptions."""
    required_keys = ["paper_title", "user_query", "outline"]
    missing_keys = [
        key for key in required_keys if key not in params or not params[key]
    ]

    if missing_keys:
        raise ValueError(f"Missing required parameters: {', '.join(missing_keys)}")

    # Validate sections_content existence
    if "sections_content" not in params:
        raise ValueError("Missing 'sections_content' in parameters")

    # Validate outline structure
    if not isinstance(params["outline"], dict):
        raise TypeError("'outline' must be a dictionary")


async def generate_abstract_conclusion(
    params: Dict[str, Any], evaluate_required: bool = False
) -> Dict[str, Any]:
    """Generate abstract and conclusion for a paper.

    Args:
        params: Dictionary containing:
            - paper_title: Title of the paper
            - user_query: Original user query
            - outline: Paper outline structure
            - sections_content: Content of each section
        evaluate_required: Whether to evaluate and potentially regenerate the abstract/conclusion

    Returns:
        Dictionary containing generated abstract and conclusion
    """
    logger.info(
        f"Starting abstract and conclusion generation for paper: {params.get('paper_title', 'Untitled')}"
    )

    try:
        # Validate input parameters
        result = {}
        result["conclusion_sections_processed"] = {}

        validate_input_params(params)

        user_query = params.get("user_query", "")
        paper_title = params.get("paper_title", "")
        outline = params.get("outline", {})
        sections_content = params.get("sections_content", {})
        reflection_num = params.get(
            "reflection_num", GLOBAL_ABSTRACT_CONCLUSION_MAX_TURNS
        )

        # Process sections data
        sections_data, conclusion_sections = prepare_sections_data(
            outline, sections_content
        )

        # Generate introductions for each non-conclusion section in sections_data
        llm_for_intro = llm_map[MODEL_GEN_ABSTRACT_CONCLUSION]
        logger.info(
            f"Attempting to generate introductions for {len(sections_data)} sections."
        )

        conclusion_section_titles = set()
        if conclusion_sections:
            conclusion_section_titles = {
                cs.get("section_title")
                for cs in conclusion_sections
                if cs.get("section_title")
            }
            logger.info(f"conclusion_sections: {conclusion_sections}")

        for section_item in sections_data:
            current_section_title = section_item.get("section_title")
            if current_section_title in conclusion_section_titles:
                logger.info(
                    f"Skipping standard introduction generation for conclusion-type section: {current_section_title}"
                )
                section_item["introduction"] = ""  # Explicitly set empty or placeholder
                continue

            logger.info("Generate section intruduction content")
            intro_content = await generate_section_introduction_content(
                paper_title=paper_title,
                user_query=user_query,
                section=section_item,
                llm=llm_for_intro,
            )

            section_info = {
                "section_name": section_item["name"],
                "section_text": intro_content,
            }
            result["conclusion_sections_processed"].update(
                {section_item["name"]: section_info}
            )

        logger.info(
            f"Finished generating introductions for sections. {len(sections_data)} sections processed."
        )

        logger.info(f"Processed {len(sections_data)} sections for paper")

        # Create initial state
        state = AbstractConclusionState(
            user_query=user_query,
            paper_title=paper_title,
            outline=outline,
            summaries=sections_data,
            max_reflections=reflection_num,
            evaluate_required=evaluate_required,
        )

        # Run the workflow
        workflow = build_abstract_conclusion_workflow()
        result_state = await workflow.ainvoke(state)

        # Extract and return results
        # Access the state values directly from the result dictionary
        final_abstract = result_state.get("abstract", "")
        final_conclusion = result_state.get("conclusion", "")
        final_meets_requirements = result_state.get("meets_requirements", False)
        final_reflection_count = result_state.get("reflection_count", 0)
        final_feedback = result_state.get("feedback", "")

        logger.info(
            f"Abstract and conclusion generation completed successfully. "
            f"Abstract length: {len(final_abstract)}, "
            f"Conclusion length: {len(final_conclusion)}, "
            f"Met requirements: {final_meets_requirements}"
        )

        result.update(
            {
                "Abstract": final_abstract,
                "Conclusion": final_conclusion,
                "meets_requirements": final_meets_requirements,
                "reflection_count": final_reflection_count,
                "feedback": final_feedback,
            }
        )

        if conclusion_sections:
            logger.info(
                "Existing conclusion sections found, generating content based on paper content"
            )
            conclusion_sections_processed = await generate_conclusion_section_content(
                result_state, conclusion_sections
            )
            result["conclusion_sections_processed"].update(
                conclusion_sections_processed
            )

        return result

    except ValueError as e:
        # Handle parameter validation errors
        error_msg = f"Parameter validation error: {str(e)}"
        logger.error(error_msg)
        return {
            "Abstract": f"Error: {error_msg}",
            "Conclusion": f"Error: {error_msg}",
            "meets_requirements": False,
            "reflection_count": 0,
            "feedback": error_msg,
        }

    except GenerationError as e:
        # Handle specific generation errors
        error_msg = f"Generation error: {str(e)}"
        logger.error(error_msg)
        return {
            "Abstract": f"Error: {error_msg}",
            "Conclusion": f"Error: {error_msg}",
            "meets_requirements": False,
            "reflection_count": 0,
            "feedback": error_msg,
        }

    except Exception as e:
        # Handle unexpected errors with detailed logging
        error_msg = f"Unexpected error in abstract/conclusion generation: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return {
            "Abstract": "An unexpected error occurred during generation. Please check the logs for details.",
            "Conclusion": "An unexpected error occurred during generation. Please check the logs for details.",
            "meets_requirements": False,
            "reflection_count": 0,
            "feedback": f"Error: {error_msg}",
        }


if __name__ == "__main__":
    import asyncio

    src_filec = "example_full_data.json"
    try:
        with open(src_filec, "r", encoding="utf-8") as f:
            full_info = json.load(f)

        # Make sure keys exist before accessing
        paper_title = full_info.get("paper_title", "")
        user_query = full_info.get("user_query", "")

        global_reflection_result = full_info.get("global_reflection_result", {})
        sections_content_input = global_reflection_result.get("sections_content", {})

        outline_structure_wo_query = full_info.get("outline_structure_wo_query", {})
        outline_input = outline_structure_wo_query.get("sections", {})

        abstract_conclusion_input = full_info.get("abstract_conclusion", {})

        test_params = {
            "paper_title": paper_title,
            "user_query": user_query,
            "outline": outline_input,
            "sections_content": sections_content_input,
            "rag_service_url": "xxx",
        }

        # result = asyncio.run(generate_abstract_conclusion(test_params))
        # print(result)
        # print("Abstract:")
        # print(result["Abstract"])
        # print("\nConclusion:")
        # print(result["Conclusion"])
        # print(f"\nMeets requirements: {result['meets_requirements']}")

        # result.update(test_params)
        # with open("./temp/paper_abstract_conclusion.json", "w") as f:
        #     json.dump(result, f, indent=2)
    except:
        traceback.print_exc()
        pass
