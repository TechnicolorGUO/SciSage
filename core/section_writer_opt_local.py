#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : RAG-based section writer with optimized async processing
# ==================================================================

import aiohttp
import asyncio
import json
import uuid
import traceback
from functools import lru_cache
from typing import Dict, Any, List, Optional, Union, Tuple
import time

from fastapi import HTTPException
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ValidationError

from log import logger
from models import SectionContent, Reference, Figure
from configuration import (
    global_semaphores,
    DEFAULT_MODEL_FOR_SECTION_WRITER,
    DEFAULT_MODEL_FOR_SECTION_WRITER_IMAGE_EXTRACT,
    DEFAULT_MODEL_FOR_SECTION_WRITER_RERANK
)
from generation.section_writer_actor import run_section_writer_actor


class RagResult(BaseModel):
    """Structured result from RAG service."""

    search_query: str = Field(description="The query sent to the RAG service")
    section_key_point: str = Field(description="The key point of this section")
    section_text: str = Field(description="The generated content from the RAG service")
    main_figure_data: str = Field(
        description="The base64 encoded image data for the main figure", default=""
    )
    main_figure_caption: str = Field(
        description="The caption for the main figure", default=""
    )
    reportIndexList: List[Reference] = Field(
        description="List of reference entities", default_factory=list
    )
    task_id: str = Field(
        description="Unique identifier for the task, used for tracking"
    )
    section_name: str = Field(description="Name of the section")
    section_index: Optional[Any] = Field(description="Index of the section")
    parent_section: Optional[str] = Field(
        default=None, description="Parent section name"
    )


class SectionWriterState(BaseModel):
    """State container for the section writer workflow."""

    section_name: str = Field(description="Name of the section")
    section_index: int = Field(description="Index of the section")
    parent_section: Optional[str] = Field(
        default=None, description="Parent section name"
    )
    user_query: str = Field(description="Original user query")
    query_domain: str=Field(description="Query domain type: [academic or general]")
    section_key_points: List[str] = Field(description="Content key points from outline")
    paper_title: str = Field(description="Paper title")
    search_queries: List[str] = Field(description="Search queries for RAG service")
    generated_content: Dict[str, Any] = Field(
        default_factory=dict, description="Generated section content"
    )
    sub_task_id: str = Field(description="Unique identifier for the section task")
    rag_model: str = Field(description="Model name used for rag section generating")


class RagServiceConfig:
    """Configuration for RAG service connection with resource management."""

    def __init__(
        self, url: str, timeout: int = 1200, max_retries: int = 2, retry_delay: int = 5
    ):
        """
        Initialize RAG service configuration.

        Args:
            url: URL for RAG service API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session = None

    @property
    async def session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp client session with proper timeout."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    def generate_serial_number(self) -> str:
        """Generate a unique request ID using UUID4."""
        return str(uuid.uuid4())

    async def close(self):
        """Close the aiohttp session if it exists."""
        if self._session and not self._session.closed:
            try:
                await self._session.close()
                self._session = None
            except Exception as e:
                logger.warning(f"Error closing aiohttp session: {e}")


def convert_ctx_to_entities(response: Dict) -> List[Reference]:
    """
    Convert RAG response's context to Reference objects.

    Args:
        response: The RAG service response dictionary

    Returns:
        List of structured Reference objects
    """
    report_index_list = response.get("ctx", [])

    # Early return for empty list to avoid unnecessary processing
    if not report_index_list:
        return []

    references = []
    for item in report_index_list:
        try:
            references.append(
                Reference(
                    title=item.get("title", ""),
                    authors=item.get("authors", ""),
                    url=item.get("url", ""),
                    conference=item.get("conference", ""),
                    source=item.get("source", ""),
                    abstract= item.get("abstract", ""),
                )
            )
        except ValidationError as e:
            logger.warning(f"Invalid reference data: {item}. Error: {e}")
            # Add a default reference to avoid failing the entire process
            references.append(
                Reference(
                    title=item.get("title", "Unknown Reference"),
                    authors="",
                    url="",
                    conference="",
                    source="",
                )
            )

    return references


async def run_rag_chat(
    rag_config: RagServiceConfig,
    query: str,
    key_point: str,
    state: SectionWriterState,
) -> Union[RagResult, Exception]:
    """
    Execute a RAG query and process the result with retry logic.

    Args:
        rag_config: Configuration for RAG service
        query: Search query
        key_point: Section key point
        state: Current workflow state

    Returns:
        RagResult or Exception if all retries fail
    """
    retries = 0
    query_summary = query[:50] + "..." if len(query) > 50 else query

    while retries <= rag_config.max_retries:
        try:
            async with global_semaphores.rag_semaphore:
                logger.info(
                    f"Sending RAG request for query: {query_summary} (attempt {retries+1})"
                )

                # Prepare the item data for run_section_writer_actor
                item_data = {
                    "query": query,
                    "query_domain":state.query_domain,
                    "request_id": rag_config.generate_serial_number(),
                    "task_id": state.sub_task_id,
                    "section_name": key_point,
                    "section_index": state.section_index,
                    "model_name": state.rag_model,
                }

                logger.debug(
                    f"[{query_summary}] RAG payload: {json.dumps(item_data, indent=2)}"
                )

                # Call run_section_writer_actor instead of making HTTP request
                section_writer_model_info = {
                    "rag_model":state.rag_model,
                    "image_extraction_model": DEFAULT_MODEL_FOR_SECTION_WRITER_IMAGE_EXTRACT,
                    "reranker_model_name": DEFAULT_MODEL_FOR_SECTION_WRITER_RERANK
                }
                response_data = await run_section_writer_actor(query, state.query_domain, state.sub_task_id,section_writer_model_info)

                # Check if the response is valid
                if not response_data:
                    logger.warning(f"RAG service returned empty response")
                    if retries < rag_config.max_retries:
                        retries += 1
                        await asyncio.sleep(rag_config.retry_delay)
                        continue
                    raise ValueError("RAG service returned empty response")

                # Check response status - use attribute access instead of .get()
                if response_data.status != 200:
                    logger.warning(
                        f"RAG service returned status {response_data.status}"
                    )
                    if retries < rag_config.max_retries:
                        retries += 1
                        await asyncio.sleep(rag_config.retry_delay)
                        continue
                    raise HTTPException(
                        status_code=500,
                        detail=f"RAG service returned error: status={response_data.status}, message={response_data.message}",
                    )

                # Extract data from successful response - use attribute access
                generated_content = response_data.output or ""
                if not generated_content:
                    logger.warning("RAG service returned empty content")
                    if retries < rag_config.max_retries:
                        retries += 1
                        await asyncio.sleep(rag_config.retry_delay)
                        continue
                    raise ValueError("RAG service returned empty content")

                # Convert ctx list to entities - response_data.ctx is already a list
                report_index_list = convert_ctx_to_entities(
                    {"ctx": response_data.ctx or []}
                )
                main_figure_data = response_data.main_figure_base64 or ""
                main_figure_caption = response_data.main_figure_caption or ""

                logger.info(
                    f"[{query_summary}] RAG successfully generated content ({len(generated_content)} chars)"
                )

                return RagResult(
                    section_key_point=key_point,
                    search_query=query,
                    section_name=state.section_name,
                    section_index=state.section_index,
                    parent_section=state.parent_section,
                    section_text=generated_content,
                    main_figure_data=main_figure_data,
                    main_figure_caption=main_figure_caption,
                    reportIndexList=report_index_list,
                    task_id=state.sub_task_id,
                )

        except asyncio.TimeoutError:
            logger.warning(
                f"RAG service call timed out for query '{query_summary}' (attempt {retries+1})"
            )
            if retries < rag_config.max_retries:
                retries += 1
                await asyncio.sleep(rag_config.retry_delay)
                continue
            return HTTPException(
                status_code=504,
                detail=f"RAG service timed out after {retries+1} attempts",
            )

        except Exception as e:
            logger.warning(
                f"RAG service call failed for query '{query_summary}': {e} (attempt {retries+1})"
            )
            if retries < rag_config.max_retries:
                retries += 1
                await asyncio.sleep(rag_config.retry_delay)
                continue
            return e


async def rag_search_async(
    rag_config: RagServiceConfig,
    state: SectionWriterState,
) -> List[RagResult]:
    """
    Process multiple RAG queries concurrently with fallback mechanisms.

    Args:
        rag_config: Configuration for RAG service
        state: Current workflow state

    Returns:
        List of successful RagResults

    Raises:
        HTTPException: If all queries fail
    """
    search_queries = state.search_queries
    section_key_points = state.section_key_points

    # Input validation with fallback
    if len(search_queries) == 0:
        # Generate generic queries if none provided
        search_queries = [
            f"Generate content about {key_point} for {state.section_name} in {state.paper_title}"
            for key_point in section_key_points
        ]
        logger.info(
            f"No search queries provided, generated {len(search_queries)} default queries"
        )

    if len(search_queries) != len(section_key_points):
        logger.warning(
            f"Mismatch between queries ({len(search_queries)}) and key points ({len(section_key_points)})"
        )
        # Make the lengths match by duplicating or truncating
        if len(search_queries) < len(section_key_points):
            # Duplicate the last query for remaining key points
            search_queries.extend(
                [search_queries[-1]] * (len(section_key_points) - len(search_queries))
            )
        else:
            # Truncate extra queries
            search_queries = search_queries[: len(section_key_points)]

        logger.info(
            f"Adjusted queries to match key points: {len(search_queries)} queries"
        )

    # Process queries concurrently
    tasks = [
        run_rag_chat(rag_config, query, key_point, state)
        for query, key_point in zip(search_queries, section_key_points)
    ]
    rag_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process any errors
    failed_queries = [
        (query, result, key_point)
        for query, result, key_point in zip(
            search_queries, rag_results, section_key_points
        )
        if isinstance(result, Exception)
    ]

    successful_results = [
        result for result in rag_results if not isinstance(result, Exception)
    ]

    if failed_queries:
        error_messages = [
            f"Key point '{key_point}' with query '{query[:30]}...': {str(error)}"
            for query, error, key_point in failed_queries
        ]
        error_details = "\n".join(error_messages)
        logger.error(f"RAG searches failed:\n{error_details}")

        # If all queries failed, raise exception with details
        if len(failed_queries) == len(search_queries):
            raise HTTPException(
                status_code=500, detail=f"All RAG queries failed: {error_details}"
            )

        # Otherwise log warnings but continue with successful results
        logger.warning(f"{len(failed_queries)}/{len(search_queries)} queries failed")

    # Generate fallback content for failed queries if needed
    if failed_queries and len(successful_results) > 0:
        # Use content from successful queries as fallback for failed ones
        for query, error, key_point in failed_queries:
            # Create fallback content using the first successful result as a template
            sample_result = successful_results[0]
            fallback_result = RagResult(
                section_key_point=key_point,
                search_query=query,
                section_name=state.section_name,
                section_index=state.section_index,
                parent_section=state.parent_section,
                section_text="",
                main_figure_data="",
                main_figure_caption="",
                reportIndexList=[],  # Empty references list
                task_id=state.sub_task_id,
            )
            successful_results.append(fallback_result)
            logger.info(f"Created fallback content for key point: {key_point}")

    return successful_results


async def section_writer_node(
    state: SectionWriterState, config: Dict[str, Any]
) -> SectionWriterState:
    """
    Process a section by retrieving content from RAG service.

    Args:
        state: Current workflow state
        config: Configuration including RAG service settings

    Returns:
        Updated workflow state with generated content
    """
    rag_config = config.get("rag_config")
    if not rag_config:
        raise ValueError("rag_config not provided in the node configuration")

    logger.info(
        f"Generating content for section: [{state.section_name}] with sub_task_id: [{state.sub_task_id}]"
    )

    try:
        # Get RAG results for all queries
        rag_results = await rag_search_async(rag_config, state)

        if not rag_results:
            raise ValueError(
                f"No content generated for section: [{state.section_name}]"
            )

        # Organize results by key point
        section_content = {result.section_key_point: result for result in rag_results}

        # Return updated state
        return SectionWriterState(
            **{**state.model_dump(), "generated_content": section_content}
        )

    except Exception as e:
        logger.error(f"[{state.section_name}]: Failed to generate section: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@lru_cache(maxsize=10)
def build_section_writer_workflow(
    rag_service_url: str, timeout: int = 1200, max_retries: int = 2
):
    """
    Build and compile the section writer workflow graph.

    Uses LRU cache to avoid rebuilding the workflow for repeated calls
    with the same parameters.

    Args:
        rag_service_url: URL for RAG service API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests

    Returns:
        Compiled workflow
    """
    rag_config = RagServiceConfig(
        url=rag_service_url, timeout=timeout, max_retries=max_retries
    )
    workflow = StateGraph(SectionWriterState)

    # Register the async node
    async def node_wrapper(state):
        try:
            result = await section_writer_node(state, {"rag_config": rag_config})
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in node_wrapper: {str(e)}")
            # Return state with error information instead of raising
            error_state = state.copy()
            error_state.generated_content = {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            return error_state.model_dump()

    workflow.add_node("generate_content", node_wrapper)
    workflow.add_edge("generate_content", END)
    workflow.set_entry_point("generate_content")

    return workflow.compile()


async def section_writer_async(
    params: Dict[str, Any],
    rag_service_url: str,
    timeout: int = 1200,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Asynchronous implementation of section content generation with enhanced error handling.

    Args:
        params: Parameters including section details and queries.
        rag_service_url: URL for the RAG service API.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for failed requests.

    Returns:
        Generated content dictionary for the section.

    Raises:
        HTTPException: If content generation completely fails
    """
    logger.info(
        f"Starting section_writer for section: [{params.get('section_name', 'Unknown')}]"
    )

    # Validate critical parameters
    required_fields = ["section_name", "user_query", "paper_title"]
    missing_fields = [field for field in required_fields if field not in params]
    if missing_fields:
        error_msg = f"Missing required parameters: {', '.join(missing_fields)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    logger.info(f"section_writer_async Parameters: {json.dumps(params, indent=2)}")

    # Create RAG service configuration
    rag_config = RagServiceConfig(
        url=rag_service_url, timeout=timeout, max_retries=max_retries
    )

    start_time = time.time()

    try:
        # Validate and prepare inputs with fallbacks
        search_queries = params.get("search_queries", [])
        section_key_points = params.get("section_key_points", [])

        if not section_key_points:
            # If no key points, create a default one
            section_key_points = [params["section_name"]]
            logger.warning(
                f"No section_key_points provided, using section name as default"
            )

        if not search_queries:
            # Generate queries from section content if not provided
            search_queries = [
                f"Generate detailed content about {content} for a paper titled '{params['paper_title']}'"
                for content in section_key_points
            ]
            logger.info(
                f"Generated {len(search_queries)} search queries from key points"
            )

        # Ensure matching length
        if len(search_queries) != len(section_key_points):
            logger.warning(
                f"Query count ({len(search_queries)}) doesn't match key point count ({len(section_key_points)})"
            )
            # Adjust to make lengths match
            if len(search_queries) < len(section_key_points):
                # Duplicate the last query for remaining key points
                search_queries.extend(
                    [search_queries[-1] if search_queries else ""]
                    * (len(section_key_points) - len(search_queries))
                )
            else:
                # Truncate extra queries
                search_queries = search_queries[: len(section_key_points)]

        # Prepare state
        state = SectionWriterState(
            section_name=params["section_name"],
            section_index=params.get("section_index", 0),
            parent_section=params.get("parent_section"),
            user_query=params["user_query"],
            query_domain=params["query_domain"],
            paper_title=params["paper_title"],
            section_key_points=section_key_points,
            search_queries=search_queries,
            sub_task_id=params.get("sub_task_id", str(uuid.uuid4())),
            rag_model=params.get("rag_model", DEFAULT_MODEL_FOR_SECTION_WRITER),
        )

        # Execute RAG search asynchronously
        rag_results = await rag_search_async(rag_config, state)

        if not rag_results:
            logger.error(
                f"[{params['section_name']}]: No content generated for section"
            )
            raise HTTPException(
                status_code=500, detail="Failed to generate section content"
            )

        # Organize results by key point
        section_content = {result.section_key_point: result for result in rag_results}

        raw_section_content = {
            key: value.model_dump() for key, value in section_content.items()
        }

        elapsed_time = time.time() - start_time
        logger.info(
            f"[{params['section_name']}]: Section writer completed in {elapsed_time:.2f}s with {len(raw_section_content)} key points"
        )
        return raw_section_content

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise

    except Exception as e:
        logger.error(f"Error in section_writer_async: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Section generation failed: {str(e)}"
        )

    finally:
        # Always clean up resources
        await rag_config.close()
        elapsed_time = time.time() - start_time
        logger.debug(f"section_writer_async completed in {elapsed_time:.2f}s")


async def eaxmple_usage():
    """Example usage for testing."""
    params = {
        "sub_task_id": "xf_writer_test",
        "section_name": "Current Approaches",
        "section_index": 0,
        "parent_section": "",
        "user_query": "current research on LLM Agent",
        "section_key_points": ["Modern architectures", "Learning mechanisms"],
        "paper_title": "Current Research on Lifelong Learning Machines (LLM) Agents: A Comprehensive Review and Exploration",
        "search_queries": [
            "Comparison of modern lifelong learning machine architectures and mechanisms in recent research literature",
            "Comparative analysis of modern architectures and learning mechanisms in Lifelong Learning Machines (LLM) Agents in recent academic literature",
        ],
        "rewrite": True,
    }
    url = "http://120.92.91.62:9528/chat"

    try:
        result = await section_writer_async(params, url)
        dest_file = "section_writer_result_opt.json"
        with open(dest_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {dest_file}")
    except Exception as e:
        print(f"Error: {e}")


# if __name__ == "__main__":
#     asyncio.run(eaxmple_usage())
