#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : refer from openscholar
# [Dependencies] : spacy, nltk, requests, tqdm, asyncio, numpy
# ==================================================================

import os
import re
import spacy
import asyncio
import numpy as np
from nltk import sent_tokenize
from typing import Dict, Any, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import requests
import tqdm
import traceback
import time
import json
from collections import Counter

from generation.instruction_pro import *

from generation.retrival import run_requests_parallel
from generation.reranker import rerank_papers_hybrid
from pydantic import BaseModel, ValidationError
import uuid

from log import logger


from generation.generation_config import (
    CHAT_MODEL_NAME,
)

from generation.websearch_scholar import get_doc_info_from_api
from generation.engine import process_ctx, process_paragraph
from generation.utils import keep_letters, flow_information_sync
from generation.extract_main_figure import get_arxiv_main_figure

from generation.relevance import DocumentRelevanceEvaluator

from local_request_v2 import get_from_llm

# Define constants for response delimiters
RESPONSE_START_DELIMITER = "[Response_Start]"
RESPONSE_END_DELIMITER = "[Response_End]"
REFERENCES_HEADER = "References:"
REVISED_ANSWER_HEADER = "Here is the revised answer:\n\n"

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error(
        "spaCy model 'en_core_web_sm' not found. Please download it: python -m spacy download en_core_web_sm"
    )
    # Depending on the application's criticality, you might want to exit or use a fallback
    nlp = None  # Or raise an exception


class OpenScholarAPIError(Exception):
    """Custom exception class for OpenScholarAPI errors."""

    pass


class LLMCommunicationError(OpenScholarAPIError):
    """Exception raised for errors during communication with the LLM."""

    pass


class DataProcessingError(OpenScholarAPIError):
    """Exception raised for errors during data processing."""

    pass


class OpenScholarAPI:
    """
    A class to interact with scientific literature APIs for tasks like retrieval,
    ranking, and response generation. Enhanced with improved error handling,
    data structures, and logical flow.
    """

    def __init__(
        self,
        client=None,  # Consider type hinting if possible
        api_model_name: Optional[str] = "Qwen3-8B",  # Use constant default
        use_contexts: bool = True,
        top_n: int = 20,
        reranker=None,  # Consider type hinting
        min_citation: Optional[int] = None,
        norm_cite: bool = False,
        online_retriver: bool = True,  # Typo: retriever
    ):
        """
        Initialize the OpenScholarAPI.

        Args:
            client: API client for interacting with language models (optional).
            api_model_name: Name of the API model to use.
            use_contexts: Whether to use contextual passages in response generation.
            top_n: Number of top passages to consider after reranking.
            reranker: Reranker model for passage reranking.
            min_citation: Minimum citation count threshold for filtering passages.
            norm_cite: Whether to normalize citation counts during reranking.
            online_retriever: Whether to use semantic search retriever for online search.
        """
        self.client = client
        self.model_name = api_model_name
        self.top_n = top_n
        self.reranker = reranker
        self.min_citation = min_citation
        self.norm_cite = norm_cite
        self.online_retriever = online_retriver  # Corrected parameter name usage
        self.use_contexts = use_contexts
        if nlp is None and (posthoc_at := True):  # Example check if nlp is needed
            logger.warning(
                "spaCy model not loaded, some functionalities like post-hoc attribution might be affected."
            )

    def _call_llm(
        self, messages: List[Dict[str, str]], model_name: Optional[str] = None, **kwargs
    ) -> str:
        """
        Helper function to call the LLM and handle basic errors.

        Args:
            messages: List of message dictionaries for the LLM.
            model_name: Specific model name to use, defaults to instance model_name.
            **kwargs: Additional arguments for get_from_llm.

        Returns:
            The LLM response content.

        Raises:
            LLMCommunicationError: If the LLM call fails or returns None.
        """
        effective_model_name = model_name or self.model_name
        logger.info(f"calling LLM with messages: {messages}")
        logger.info(f"effective_model_name: {effective_model_name}")
        try:
            content = get_from_llm(messages, model_name=effective_model_name, **kwargs)
            if content is None:
                logger.error(
                    f"LLM call returned None for model {effective_model_name}. Messages: {messages}"
                )
                raise LLMCommunicationError(
                    f"LLM call failed for model {effective_model_name}. Received no content."
                )
            return content
        except Exception as e:
            logger.error(
                f"Error during LLM call for model {effective_model_name}: {e}\n{traceback.format_exc()}"
            )
            # Re-raise as a custom exception for potentially more specific handling upstream
            raise LLMCommunicationError(
                f"LLM call failed for model {effective_model_name}: {e}"
            ) from e

    def _parse_llm_output(self, output: str) -> str:
        """
        Helper function to parse LLM output, removing delimiters.
        """
        if RESPONSE_START_DELIMITER in output:
            output = output.split(RESPONSE_START_DELIMITER, 1)[1]
        if RESPONSE_END_DELIMITER in output:
            output = output.split(RESPONSE_END_DELIMITER, 1)[0]
        # Add other common cleanup steps if needed
        output = output.strip()
        return output

    def judge_section_shold_have_figure(self, section_name: str) -> Tuple[bool, Dict]:
        """
        Judge if a section should ideally contain a figure based on its name.

        Args:
            section_name: The name of the section.

        Returns:
            Tuple (bool indicating if a figure is needed, dictionary with details or empty if error).
        """
        logger.info(
            f"Running judge_section_shold_have_figure for section: '{section_name}'"
        )
        if not section_name:
            logger.warning(
                "judge_section_shold_have_figure called with empty section_name."
            )
            return False, {}

        message = [
            {"role": "system", "content": judge_section_should_give_an_system},
            {
                "role": "user",
                "content": judge_section_should_give_an_figure.format(
                    section_name=section_name
                ),
            },
        ]

        try:
            # Using a potentially more capable model for judgment tasks might be beneficial
            content = self._call_llm(
                message, model_name=self.model_name, temperature=0.8
            )  # Consider making model configurable

            # More robust JSON extraction
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                logger.error(
                    f"Could not find JSON in LLM response for judge_section_shold_have_figure: {content}"
                )
                return False, {
                    "error": "JSON not found in response",
                    "raw_response": content,
                }

            json_str = json_match.group(0)
            parsed_result = json.loads(json_str)

            # Validate expected keys
            if "need_image" not in parsed_result:
                logger.error(
                    f"Key 'need_image' missing in parsed JSON: {parsed_result}"
                )
                return False, {
                    "error": "Key 'need_image' missing",
                    "parsed_json": parsed_result,
                }

            needs_image = str(parsed_result.get("need_image", "no")).lower() == "yes"
            logger.info(
                f"Judge result for section '{section_name}': Needs image? {needs_image}"
            )
            return needs_image, parsed_result

        except json.JSONDecodeError as e:
            logger.error(
                f"JSON decoding failed for judge_section_shold_have_figure: {e}. Response content: {content}"
            )
            return False, {"error": "JSON decoding failed", "raw_response": content}
        except LLMCommunicationError as e:
            logger.error(
                f"LLM communication failed for judge_section_shold_have_figure: {e}"
            )
            return False, {"error": f"LLM communication failed: {e}"}
        except Exception as e:  # Catch broader exceptions as a fallback
            logger.error(
                f"Unexpected error in judge_section_shold_have_figure: {e}\n{traceback.format_exc()}"
            )
            return False, {"error": f"Unexpected error: {e}"}

    def _filter_passages_by_citation(self, passages: List[Dict]) -> List[Dict]:
        """Filter passages based on minimum citation count.

        Args:
            passages: List of passage dictionaries.

        Returns:
            Filtered list of passages.
        """
        if self.min_citation is None:
            return passages

        # Improved filtering with explicit check for key existence and type
        filtered = []
        for p in passages:
            citation_count = p.get("citation_counts")  # Use .get for safety
            if citation_count is not None:
                try:
                    if int(citation_count) >= self.min_citation:
                        filtered.append(p)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid citation count format found: {citation_count} in passage {p.get('id', 'N/A')}. Skipping."
                    )

        num_filtered = len(filtered)
        logger.info(
            f"Filtered {len(passages) - num_filtered} passages by citation count (min: {self.min_citation}). Remaining: {num_filtered}"
        )
        # The check `if len(filtered) > self.top_n:` seems misplaced here,
        # as filtering happens before potentially selecting top_n.
        # Consider if this log message is needed or should be adjusted.

        return filtered

    def _format_passages(self, passages: List[Dict], start_index: int = 0) -> str:
        """Format passages into a string for inclusion in prompts.

        Args:
            passages: List of passage dictionaries.
            start_index: Starting index for passage numbering.

        Returns:
            Formatted string of passages.
        """
        formatted_lines = []
        for idx, doc in enumerate(passages):
            doc_idx = start_index + idx
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if not text:  # Skip passages with empty text
                logger.warning(
                    f"Passage at index {idx} (doc_idx {doc_idx}) has empty text. Skipping."
                )
                continue

            if title:
                formatted_lines.append(f"[{doc_idx}] Title: {title}; Text: {text}")
            else:
                formatted_lines.append(f"[{doc_idx}] {text}")
        return "\n".join(formatted_lines)

    def generate_response(
        self,
        item: Dict[str, Any],
        max_tokens: int = 8192,
        llama3_chat: bool = False,  # Parameter seems unused in the provided logic
        task_name: str = "default",
        zero_shot: bool = False,
        model_name: Optional[str] = None,
    ) -> Tuple[Optional[str], str, float]:
        """Generate a response based on the query and passages.

        Args:
            item: Dictionary containing input query and potentially 'ctxs'.
            max_tokens: Maximum tokens for the generated response.
            llama3_chat: Whether to use LLaMA3 chat format (currently unused).
            task_name: Task type (e.g., summarization, single_qa).
            zero_shot: Whether to use zero-shot prompting.
            model_name: Specific model name to use, defaults to instance model_name.


        Returns:
            Tuple of (generated response or None on failure, formatted passages string, API cost (currently 0)).
        """
        logger.info(
            f"Generating response for task '{task_name}'. Zero-shot: {zero_shot}, Use contexts: {self.use_contexts}"
        )

        input_query = item.get("input", "")
        if not input_query:
            logger.error("Input query is missing in the item for generate_response.")
            raise ValueError("Input query ('input') cannot be empty.")

        ctxs_str = ""
        final_prompt = ""
        try:

            logger.info("Generating response with context passages.")
            passages = item.get("ctxs", [])
            if not passages:
                logger.warning(
                    "no 'ctxs' found in the item."
                )
                # Decide fallback: proceed without context or raise error?
                # Option 1: Proceed as if use_contexts was False (could be confusing)
                # Option 2: Raise error
                raise ValueError(
                    "Contexts are required  but 'ctxs' is missing or empty."
                )

            ctxs_str = self._format_passages(passages[: self.top_n])
            item["final_passages"] = ctxs_str  # Store the formatted string used

            # Simplify template selection logic
            template = None

            template = (
                generation_instance_prompts_w_references_zero_shot
                if zero_shot
                else generation_instance_prompts_w_references
            )

            # Format using template if not already constructed
            if template and not final_prompt:
                if not template:
                    raise ValueError(
                        f"Prompt template is missing for task '{task_name}' (zero_shot={zero_shot})."
                    )
                logger.debug(f"Using template for task '{task_name}'")
                final_prompt = template.format_map(
                    {"context": ctxs_str, "input": input_query}
                )
            elif not final_prompt:
                # This case should ideally be covered by the logic above
                raise ValueError(
                    f"Could not determine prompt structure for task '{task_name}' with context."
                )

            logger.debug(
                f"Final generation prompt (first 500 chars): {final_prompt[:500]}..."
            )

            messages = [
                {"role": "system", "content": chat_system},
                {"role": "user", "content": final_prompt},
            ]

            outputs = self._call_llm(
                messages,
                model_name=model_name,
                temperature=0.7,
                max_tokens=max_tokens,  # Pass max_tokens here
            )

            logger.debug(f"Raw generation response: {outputs}")
            raw_output = self._parse_llm_output(outputs)

            # Remove references section if appended by the model
            if REFERENCES_HEADER in raw_output:
                raw_output = raw_output.split(REFERENCES_HEADER, 1)[0].rstrip()

            item["output"] = raw_output  # Store the generated output back into the item
            # Cost calculation needs actual implementation if required
            api_cost = 0.0
            return raw_output, ctxs_str, api_cost

        except LLMCommunicationError as e:
            logger.error(f"LLM communication failed during response generation: {e}")
            return None, ctxs_str, 0.0  # Return None for output on failure
        except (ValueError, KeyError) as e:  # Catch specific data errors
            logger.error(
                f"Data error during response generation: {e}\n{traceback.format_exc()}"
            )
            raise DataProcessingError(f"Data error: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during response generation: {e}\n{traceback.format_exc()}"
            )
            # Re-raise or return None depending on desired handling
            raise OpenScholarAPIError(f"Unexpected generation error: {e}") from e

    def process_feedback(self, response: str) -> List[Tuple[str, str]]:
        """Extract feedback and questions from the response using regex.

        Args:
            response: Raw response string containing feedback.

        Returns:
            List of (feedback, question) tuples. Returns empty list if no matches.
        """
        # Regex to find Feedback and optional Question pairs
        # Made Question part non-capturing within the optional group for clarity
        # Handles potential variations in spacing and line breaks better
        pattern = re.compile(
            r"Feedback:\s*(.*?)(?:\n\s*Question:\s*(.*?))?(?=\nFeedback:|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        matches = pattern.findall(response)

        processed_feedback = []
        for feedback, question in matches:
            fb = feedback.strip()
            q = question.strip() if question else ""
            if fb:  # Only add if feedback is not empty
                processed_feedback.append((fb, q))

        if not matches and response:
            logger.warning(
                f"Could not parse feedback using regex from response: {response[:100]}..."
            )
            # Fallback: maybe return the whole response as feedback? Or log and return empty.
            # return [(response.strip(), "")] # Example fallback

        return processed_feedback

    def get_feedback(
        self,
        item: Dict[str, Any],
        llama3_chat: bool = False,  # Parameter seems unused
        max_tokens: int = 8192,
        model_name: Optional[str] = None,
        retry_attempts: int = 2,  # Added retry mechanism
    ) -> Tuple[List[Tuple[str, str]], float]:
        """Generate feedback on the model's predictions.

        Args:
            item: Dictionary containing input, final_passages, and output.
            llama3_chat: Whether to use LLaMA3 chat format (currently unused).
            max_tokens: Maximum tokens for feedback generation.
            model_name: Specific model name to use, defaults to instance model_name.
            retry_attempts: Number of times to retry on failure.

        Returns:
            Tuple of (feedback list, API cost (currently 0)). Returns empty list on failure.
        """
        logger.info("Running get_feedback...")
        required_keys = ["input", "final_passages", "output"]
        if not all(key in item for key in required_keys):
            logger.error(
                f"Missing required keys for get_feedback. Need: {required_keys}, Have: {list(item.keys())}"
            )
            raise ValueError(
                f"Item dictionary missing required keys for feedback generation: {required_keys}"
            )

        # Validate that required fields are not empty
        if (
            not item["input"]
            or item["final_passages"] is None
            or item["output"] is None
        ):
            logger.error(
                f"One or more required fields are empty/None for get_feedback. Input: '{item['input']}', Passages: '{item['final_passages']}', Output: '{item['output']}'"
            )
            # Decide: return empty feedback or raise error? Returning empty for now.
            return [], 0.0

        input_query = feedback_example_instance_prompt.format_map(
            {
                "question": item["input"],
                "answer": item["output"],
                "references": item.get(
                    "final_passages", ""
                ),  # 添加新的 references 参数
            }
        )
        logger.debug(f"Get_feedback prompt (first 500 chars): {input_query[:500]}...")

        messages = [
            {"role": "system", "content": chat_system},
            {"role": "user", "content": input_query},
        ]

        for attempt in range(retry_attempts + 1):
            try:
                outputs = self._call_llm(
                    messages,
                    model_name=model_name,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )

                logger.debug(f"Raw feedback response: {outputs}")
                # Use the dedicated parsing function
                raw_output = self._parse_llm_output(outputs)
                feedbacks = self.process_feedback(raw_output)

                if not feedbacks and raw_output:
                    logger.warning(
                        f"Feedback parsing returned empty list, but raw output was present. Raw: {raw_output[:100]}..."
                    )
                    # Consider if the raw_output itself should be treated as feedback in this case

                logger.info(f"Generated {len(feedbacks)} feedback items.")
                return feedbacks, 0.0  # Cost is 0

            except LLMCommunicationError as e:
                logger.warning(
                    f"LLM communication failed during get_feedback (Attempt {attempt + 1}/{retry_attempts + 1}): {e}"
                )
                if attempt == retry_attempts:
                    logger.error("Max retries reached for get_feedback.")
                    return [], 0.0  # Fallback after retries
                time.sleep(1 * (attempt + 1))  # Exponential backoff (simple version)
            except Exception as e:
                logger.error(
                    f"Unexpected error in get_feedback: {e}\n{traceback.format_exc()}"
                )
                return [], 0.0  # Fallback on unexpected error

        return (
            [],
            0.0,
        )  # Should not be reached if retry logic is correct, but as a safeguard

    def edit_with_feedback(
        self,
        item: Dict[str, Any],
        feedback: str,
        max_tokens: int = 8192,
        llama3_chat: bool = False,  # Parameter seems unused
        model_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Edit the response based on feedback.

        Args:
            item: Dictionary containing input, final_passages, and output.
            feedback: Feedback string to apply.
            max_tokens: Maximum tokens for the edited response.
            llama3_chat: Whether to use LLaMA3 chat format (currently unused).
            model_name: Specific model name to use, defaults to instance model_name.

        Returns:
            Tuple of (edited response, API cost (currently 0)). Returns original output on failure.
        """
        logger.info("Running edit_with_feedback...")
        required_keys = ["input", "final_passages", "output"]
        if not all(key in item for key in required_keys):
            logger.error(
                f"Missing required keys for edit_with_feedback. Need: {required_keys}, Have: {list(item.keys())}"
            )
            raise ValueError(
                f"Item dictionary missing required keys for editing: {required_keys}"
            )
        if not feedback:
            logger.warning(
                "edit_with_feedback called with empty feedback. Returning original output."
            )
            return item.get("output", ""), 0.0

        input_query = editing_instance_prompt.format_map(
            {
                "question": item["input"],
                "passages": item[
                    "final_passages"
                ],  # Assumes final_passages exists and is formatted
                "answer": item["output"],
                "feedback": feedback,
            }
        )
        logger.debug(
            f"Edit_with_feedback prompt (first 500 chars): {input_query[:500]}..."
        )

        messages = [
            {"role": "system", "content": chat_system},
            {"role": "user", "content": input_query},
        ]

        original_output = item.get("output", "")  # Store original for fallback

        try:
            outputs = self._call_llm(
                messages, model_name=model_name, temperature=0.7, max_tokens=max_tokens
            )

            logger.info(f"Raw edit_with_feedback response: {outputs}")
            edited_output = self._parse_llm_output(outputs)

            # Clean potential model preamble
            if REVISED_ANSWER_HEADER in edited_output:
                edited_output = edited_output.split(REVISED_ANSWER_HEADER, 1)[1]

            logger.info(f"Original answer: {original_output[:100]}...")
            logger.info(f"Feedback: {feedback[:100]}...")
            logger.info(f"Updated answer: {edited_output[:100]}...")
            return edited_output, 0.0  # Cost is 0

        except LLMCommunicationError as e:
            logger.error(
                f"LLM communication failed during edit_with_feedback: {e}. Returning original output."
            )
            return original_output, 0.0  # Fallback
        except (ValueError, KeyError) as e:  # Catch specific data errors
            logger.error(
                f"Data error during edit_with_feedback: {e}\n{traceback.format_exc()}"
            )
            raise DataProcessingError(f"Data error: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error in edit_with_feedback: {e}\n{traceback.format_exc()}. Returning original output."
            )
            return original_output, 0.0  # Fallback

    def edit_with_feedback_retrieval(
        self,
        item: Dict[str, Any],
        feedback: str,
        passages: List[Dict],  # Newly retrieved passages
        passage_start_index: int,
        max_tokens: int = 8192,
        llama3_chat: bool = False,  # Parameter seems unused
        model_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Edit the response using feedback and newly retrieved passages.

        Args:
            item: Dictionary containing input, output.
            feedback: Feedback string to apply.
            passages: Newly retrieved passages (list of dicts).
            passage_start_index: Starting index for numbering new passages.
            max_tokens: Maximum tokens for the edited response.
            llama3_chat: Whether to use LLaMA3 chat format (currently unused).
            model_name: Specific model name to use, defaults to instance model_name.

        Returns:
            Tuple of (edited response, API cost (currently 0)). Returns original output on failure.
        """
        logger.info("Running edit_with_feedback_retrieval...")
        required_keys = [
            "input",
            "output",
        ]  # Doesn't necessarily need 'final_passages' from original item
        if not all(key in item for key in required_keys):
            logger.error(
                f"Missing required keys for edit_with_feedback_retrieval. Need: {required_keys}, Have: {list(item.keys())}"
            )
            raise ValueError(
                f"Item dictionary missing required keys for editing with retrieval: {required_keys}"
            )
        if not feedback:
            logger.warning(
                "edit_with_feedback_retrieval called with empty feedback. Returning original output."
            )
            return item.get("output", ""), 0.0
        if not passages:
            logger.warning(
                "edit_with_feedback_retrieval called with empty new passages. Consider falling back to non-retrieval edit or returning original."
            )
            # Option: Call edit_with_feedback instead?
            # return self.edit_with_feedback(item, feedback, max_tokens, llama3_chat, model_name)
            # Option: Return original
            return item.get("output", ""), 0.0

        logger.debug(f"Number of new passages provided: {len(passages)}")
        # Format only the top N new passages for the prompt
        processed_passages_str = self._format_passages(
            passages[: self.top_n], passage_start_index
        )
        logger.debug(
            f"Formatted new passages for prompt: {processed_passages_str[:200]}..."
        )

        input_query = editing_with_retrieval_instance_prompt.format_map(
            {
                "question": item["input"],
                "retrieved_passages": processed_passages_str,
                "answer": item["output"],
                "feedback": feedback,
            }
        )
        logger.debug(
            f"Edit_with_feedback_retrieval prompt (first 500 chars): {input_query[:500]}..."
        )

        messages = [
            {"role": "system", "content": chat_system},
            {"role": "user", "content": input_query},
        ]

        original_output = item.get("output", "")  # Store original for fallback

        try:
            outputs = self._call_llm(
                messages, model_name=model_name, temperature=0.7, max_tokens=max_tokens
            )

            logger.debug(f"Raw edit_with_feedback_retrieval response: {outputs}")
            edited_output = self._parse_llm_output(outputs)

            # Clean potential model preamble
            if REVISED_ANSWER_HEADER in edited_output:
                edited_output = edited_output.split(REVISED_ANSWER_HEADER, 1)[1]

            logger.info(f"Original answer: {original_output[:100]}...")
            logger.info(f"Feedback: {feedback[:100]}...")
            logger.info(f"Updated answer with retrieval: {edited_output[:100]}...")

            return edited_output, 0.0  # Cost is 0

        except LLMCommunicationError as e:
            logger.error(
                f"LLM communication failed during edit_with_feedback_retrieval: {e}. Returning original output."
            )
            return original_output, 0.0  # Fallback
        except (ValueError, KeyError) as e:  # Catch specific data errors
            logger.error(
                f"Data error during edit_with_feedback_retrieval: {e}\n{traceback.format_exc()}"
            )
            raise DataProcessingError(f"Data error: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error in edit_with_feedback_retrieval: {e}\n{traceback.format_exc()}. Returning original output."
            )
            return original_output, 0.0  # Fallback

    def _process_attributions_output(
        self,
        outputs: List[
            Optional[str]
        ],  # Allow None in case of LLM failure for a specific sentence
        post_hoc_sentence_map: Dict[str, str],  # Map placeholder to original sentence
    ) -> Dict[str, str]:
        """Process the output of attribution insertion, handling potential errors.

        Args:
            outputs: List of raw outputs from the LLM (or None if failed).
            post_hoc_sentence_map: Dictionary mapping placeholders to original sentences.

        Returns:
            Updated dictionary where placeholders map to attributed sentences or original if attribution failed.
        """
        updated_map = post_hoc_sentence_map.copy()  # Work on a copy
        placeholders = list(post_hoc_sentence_map.keys())

        if len(outputs) != len(placeholders):
            logger.error(
                f"Mismatch between number of attribution outputs ({len(outputs)}) and placeholders ({len(placeholders)}). Cannot reliably map results."
            )
            # Fallback: return original map, no attributions applied
            return updated_map

        for i, output in enumerate(outputs):
            placeholder = placeholders[i]
            original_sentence = post_hoc_sentence_map[placeholder]

            if output is None:
                logger.warning(
                    f"Attribution failed for placeholder {placeholder}. Keeping original sentence: '{original_sentence[:50]}...'"
                )
                updated_map[placeholder] = (
                    original_sentence  # Keep original if LLM failed
                )
                continue

            try:
                # Parse the output for the specific sentence
                processed_output = self._parse_llm_output(output)
                # Basic validation: Check if it's not empty and maybe contains the placeholder? (optional)
                if processed_output:
                    updated_map[placeholder] = processed_output
                else:
                    logger.warning(
                        f"Attribution resulted in empty output for placeholder {placeholder}. Keeping original sentence."
                    )
                    updated_map[placeholder] = (
                        original_sentence  # Keep original if parsing fails or yields empty
                    )

            except Exception as e:
                logger.error(
                    f"Error processing attribution output for placeholder {placeholder}: {e}. Output: '{output[:100]}...'. Keeping original sentence."
                )
                updated_map[placeholder] = (
                    original_sentence  # Keep original on unexpected error
                )

        return updated_map

    def _insert_attributions_posthoc_base(
        self,
        item: Dict[str, Any],
        prompt_template: str,
        split_method: str = "paragraph",  # "paragraph" or "sentence"
        max_tokens: int = 8192,
        llama3_chat: bool = False,  # Unused
        model_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Base function for inserting attributions post-hoc (paragraph or sentence).

        Args:
            item: Dictionary containing 'output' and 'ctxs' or 'final_passages'.
            prompt_template: The specific prompt template to use.
            split_method: How to split the text ("paragraph" or "sentence").
            max_tokens: Maximum tokens for attribution generation per segment.
            llama3_chat: Unused parameter.
            model_name: Specific model name to use.

        Returns:
            Tuple of (updated text with attributions, API cost (currently 0)). Returns original text on major failure.
        """
        logger.info(
            f"Running post-hoc attribution insertion (method: {split_method})..."
        )
        text = item.get("output", "")
        if not text:
            logger.warning("Input text ('output') is empty. Skipping attribution.")
            return "", 0.0

        # Get passages, preferring 'final_passages' if available (already formatted)
        passages_str = item.get("final_passages")
        if passages_str is None:
            logger.info("Using 'ctxs' for passages as 'final_passages' not found.")
            passages_list = item.get("ctxs", [])
            if not passages_list:
                logger.warning(
                    "No passages ('ctxs' or 'final_passages') found. Attributions may be inaccurate or skipped."
                )
                # Decide: return original text or proceed with empty passages?
                # Proceeding with empty passages string for now, LLM might handle it.
                passages_str = ""
            else:
                passages_str = self._format_passages(passages_list)  # Format if needed
        elif not passages_str:
            logger.warning(
                "Provided 'final_passages' is empty. Attributions may be inaccurate."
            )

        # Split text into segments (sentences or paragraphs)
        if split_method == "paragraph":
            segments = text.split("\n")
        elif split_method == "sentence":
            if nlp is None:
                logger.error(
                    "spaCy model not loaded, cannot perform sentence splitting for attribution. Falling back to paragraph split."
                )
                segments = text.split("\n")
                split_method = "paragraph"  # Update method for logging consistency
            else:
                try:
                    doc = nlp(text)
                    segments = [sent.text for sent in doc.sents]
                except Exception as e:
                    logger.error(
                        f"spaCy sentence tokenization failed: {e}. Falling back to paragraph split."
                    )
                    segments = text.split("\n")
                    split_method = "paragraph"
        else:
            raise ValueError(f"Invalid split_method: {split_method}")

        updated_segments = []
        post_hoc_segment_map = {}  # Maps placeholder to original segment text
        needs_attribution_indices = []

        # Identify segments needing attribution
        for s_index, segment in enumerate(segments):
            segment_stripped = segment.strip()
            # Basic heuristic: Skip short segments, segments already containing citations "[...]",
            # or segments followed by a citation line. More sophisticated checks possible.
            is_likely_citation = segment_stripped.startswith(
                "["
            ) and segment_stripped.endswith("]")
            has_internal_citation = re.search(r"\[\d+\]", segment_stripped) is not None
            # Check if next segment is purely a citation (simple check)
            next_segment_is_citation = (
                split_method == "paragraph"  # Only makes sense for paragraphs
                and s_index < len(segments) - 1
                and segments[s_index + 1].strip().startswith("[")
                and segments[s_index + 1].strip().endswith("]")
            )

            # Heuristic: Needs attribution if reasonably long and doesn't seem to have citations.
            # Adjust length threshold as needed.
            min_length_threshold = 15 if split_method == "sentence" else 25

            if (
                len(segment_stripped) < min_length_threshold
                or has_internal_citation
                or next_segment_is_citation
            ):
                # Handle cases where a citation might be attached to the previous segment
                if updated_segments and is_likely_citation:
                    joiner = " " if split_method == "sentence" else "\n"
                    updated_segments[-1] += joiner + segment
                else:
                    updated_segments.append(segment)
            else:
                placeholder = f"[replace_{s_index}]"
                updated_segments.append(placeholder)
                post_hoc_segment_map[placeholder] = segment  # Store original
                needs_attribution_indices.append(s_index)

        if not post_hoc_segment_map:
            logger.info("No segments identified as needing post-hoc attribution.")
            return text, 0.0

        logger.info(
            f"{len(post_hoc_segment_map)} {split_method}(s) require attribution, e.g., '{list(post_hoc_segment_map.values())[0][:50]}...'"
        )

        # Prepare prompts for segments needing attribution
        prompts = []
        placeholders_order = list(post_hoc_segment_map.keys())  # Keep order consistent
        for placeholder in placeholders_order:
            original_segment = post_hoc_segment_map[placeholder]
            # Optional: Pre-process segment (e.g., clean residual citations if needed)
            cleaned_segment = (
                process_paragraph(re.sub(r"\[\d+\]", "", original_segment))
                if split_method == "paragraph"
                else original_segment
            )

            prompts.append(
                prompt_template.format_map(
                    {
                        "statement": cleaned_segment,  # Use cleaned/original segment
                        "passages": passages_str,
                    }
                )
            )

        # Call LLM for each prompt (Consider batching if API/model supports it)
        llm_outputs = []
        desc = f"Posthoc Attribution ({split_method})"
        for i, input_query in enumerate(
            tqdm.tqdm(prompts, total=len(prompts), desc=desc)
        ):
            logger.debug(
                f"Inferring attribution for segment {i+1}/{len(prompts)}. Placeholder: {placeholders_order[i]}"
            )
            logger.debug(f"Attribution prompt (first 300): {input_query[:300]}...")
            messages = [
                {"role": "system", "content": chat_system},
                {"role": "user", "content": input_query},
            ]
            try:
                output = self._call_llm(
                    messages,
                    model_name=model_name,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )  # Adjust max_tokens if needed per segment
                llm_outputs.append(output)
                logger.debug(f"Raw attribution response: {output[:100]}...")
            except LLMCommunicationError as e:
                logger.warning(
                    f"LLM call failed for attribution segment {i+1}: {e}. Will keep original segment."
                )
                llm_outputs.append(
                    None
                )  # Append None to indicate failure for this segment

        # Process LLM outputs and substitute back
        processed_segment_map = self._process_attributions_output(
            llm_outputs, post_hoc_segment_map
        )

        # Reconstruct the final text
        final_segments = []
        for segment_template in updated_segments:
            if segment_template in processed_segment_map:
                final_segments.append(processed_segment_map[segment_template])
            else:
                final_segments.append(
                    segment_template
                )  # Keep segments that didn't need attribution

        # Join based on original split method
        joiner = "\n" if split_method == "paragraph" else " "
        final_text = joiner.join(final_segments)

        logger.info(f"Finished post-hoc attribution ({split_method}).")
        return final_text, 0.0  # Cost is 0

    # --- Specific Attribution Methods ---

    def insert_attributions_posthoc_paragraph(
        self,
        item: Dict[str, Any],
        max_tokens: int = 8192,
        llama3_chat: bool = False,  # Unused
        model_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Insert attributions into paragraphs post-hoc."""
        # This method now becomes a simple wrapper around the base method
        return self._insert_attributions_posthoc_base(
            item,
            prompt_template=posthoc_attributions_paragraph,
            split_method="paragraph",
            max_tokens=max_tokens,
            llama3_chat=llama3_chat,
            model_name=model_name,
        )

    def insert_attributions_posthoc(
        self,
        item: Dict[str, Any],
        max_tokens: int = 8192,
        llama3_chat: bool = False,  # Unused
        model_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Insert attributions into sentences post-hoc."""
        # This method now becomes a simple wrapper around the base method
        return self._insert_attributions_posthoc_base(
            item,
            prompt_template=posthoc_attributions,
            split_method="sentence",
            max_tokens=max_tokens,
            llama3_chat=llama3_chat,
            model_name=model_name,
        )

    def refine_markdown_content(
        self,
        markdown_text: str,
        max_tokens: int = 8192,
        model_name: Optional[str] = "Qwen3-8B",
    ) -> str:
        """
        Refine the given markdown content using an LLM.

        Args:
            markdown_text: The markdown content to refine.
            max_tokens: Maximum tokens for the LLM response.
            model_name: The name of the model to use for refinement.

        Returns:
            The refined markdown content as a string. Returns the original content if refinement fails.
        """
        if not markdown_text:
            logger.warning("refine_markdown_content called with empty markdown_text.")
            return markdown_text  # Return original content if empty

        effective_model_name = model_name or self.model_name
        try:
            # Prepare the prompt
            message = context_refine_prompt.format(markdown_text=markdown_text)
            logger.debug(f"refine_markdown_content prompt: {message[:200]}...")

            # Call the LLM
            response = self._call_llm(
                [{"role": "user", "content": message}],
                model_name=effective_model_name,
                temperature=0.3,  # Low temperature for deterministic output
                max_tokens=max_tokens,
            )
            logger.debug(f"refine_markdown_content raw response: {response[:200]}...")

            # Parse and clean the output
            refined_content = self._parse_llm_output(response)
            if not refined_content:
                logger.warning(
                    "refine_markdown_content returned empty content. Using original markdown_text."
                )
                return markdown_text  # Fallback to original content if refinement fails

            logger.info("refine_markdown_content completed successfully.")
            return refined_content

        except LLMCommunicationError as e:
            logger.error(f"LLM communication error during markdown refinement: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error during markdown refinement: {e}\n{traceback.format_exc()}"
            )

        # Fallback to original content in case of any error
        return markdown_text

    def insert_attributions_posthoc_paragraph_all(
        self,
        item: Dict[str, Any],
        max_tokens: int = 8192,
        llama3_chat: bool = False,  # Unused
        model_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Insert attributions into all paragraphs post-hoc (using specific prompt)."""
        # This method uses a different prompt but the same base logic
        # Note: The name implies "all" paragraphs, ensure the prompt reflects this intent.
        # The base method's heuristic might still skip some paragraphs.
        # If truly *all* paragraphs need processing regardless of content,
        # the heuristic logic in the base method needs adjustment for this specific call.
        # For now, assuming the prompt `posthoc_attributions_paragraph_all` handles the "all" aspect.

        # Ensure 'ctxs_refered' is used for passages if available and intended
        passages_list = item.get("ctxs_refered")
        if passages_list:
            logger.info(
                "Using 'ctxs_refered' for passages in insert_attributions_posthoc_paragraph_all."
            )
            item["final_passages"] = self._format_passages(passages_list)
        elif "ctxs" in item:
            logger.warning(
                "'ctxs_refered' not found, falling back to 'ctxs' for insert_attributions_posthoc_paragraph_all."
            )
            item["final_passages"] = self._format_passages(item["ctxs"])
        else:
            logger.warning(
                "No context ('ctxs_refered' or 'ctxs') found for insert_attributions_posthoc_paragraph_all."
            )
            item["final_passages"] = ""

        return self._insert_attributions_posthoc_base(
            item,
            prompt_template=posthoc_attributions_paragraph_all,
            split_method="paragraph",  # Still paragraph-based
            max_tokens=max_tokens,
            llama3_chat=llama3_chat,
            model_name=model_name,
        )


    def _run_ctx_rerank(self, query, ctxs, reranker_model_name):
        """Run context reranking using the hybrid approach."""
        reranked_ctxs = rerank_papers_hybrid(
            ctxs, query, model_name=reranker_model_name
        )
        logger.info(
            f"reranked_ctxs sample key: {list(reranked_ctxs[0].keys()) if reranked_ctxs else 'No contexts found'}"
        )
        return reranked_ctxs

    async def get_rag_qa(
        self,
        user_query: str,
        item: Dict[str, Any],
        request_id: str,
        task_id: str,
        model_name: str = "openscholar",  # Use instance default if None
        reranker_model_name: str = "Qwen3-8B",  # Use instance default if None
        image_extraction_model: str = "Qwen3-8B",  # Use instance default if None
        ranking_ce: bool = True,  # Default to True?
        use_feedback: bool = False,
        skip_generation: bool = False,
        posthoc_at: bool = True,  # Default to True?
        format_reflection: bool = True,  # <-- 新增参数
        llama3_chat: bool = False,  # Unused
        task_name: str = "default",
        zero_shot: bool = False,
        use_abstract: bool = False,
        max_tokens: int = 8192,
        new_feedback_docs: int = 5,
        feedback_num: int = 3,
        # online_retriver: bool = True, # Use self.online_retriever
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Run the Retrieval-Augmented Generation (RAG) QA pipeline with improved error handling and flow.

        Args:
             user_query: The original user query triggering the process.
             item: Dictionary containing initial data, typically 'input' and 'ctxs'.
                   It will be updated throughout the pipeline.
             request_id: Unique identifier for the request.
             task_id: Unique identifier for the task (used for logging/flow tracking).
             model_name: Model name for generation/feedback/editing. Defaults to instance model.
             ranking_ce: Whether to rerank passages using a cross-encoder.
             use_feedback: Whether to use feedback for iterative improvement.
             skip_generation: Whether to skip the initial response generation (e.g., if output already provided).
             posthoc_at: Whether to insert attributions post-hoc.
             llama3_chat: Unused parameter.
             task_name: Task type (e.g., summarization, single_qa).
             zero_shot: Whether to use zero-shot prompting for generation.
             max_per_paper: Maximum passages per paper (currently unused).
             use_abstract: Whether to use abstracts in reranking.
             max_tokens: Maximum tokens for response generation/editing.
             new_feedback_docs: Number of new documents to retrieve per feedback iteration.
             feedback_num: Maximum number of feedback iterations to perform.
             **kwargs: Additional arguments (e.g., section_index, section_name).

        Returns:
            Updated item dictionary with results ('output', 'initial_result', 'feedbacks', etc.)
            or None if a critical error occurs that prevents meaningful output.
        """

        self.task_id = task_id  # Store task_id for potential use in methods
        section_index = kwargs.get("section_index", "SectionFake")
        section_name = kwargs.get(
            "section_name", ""
        )  # Get section name for figure judgment
        effective_model_name = (
            model_name or self.model_name
        )  # Determine model name to use

        logger.info(
            f"Starting RAG pipeline for task_id: {task_id}, request_id: {request_id}"
        )
        logger.info(
            f"Parameters: ranking_ce={ranking_ce}, use_feedback={use_feedback}, posthoc_at={posthoc_at}, task_name='{task_name}', model='{effective_model_name}'"
        )

        already_retrival_papers = {}

        # --- Main Pipeline ---
        try:
            # 1. Initial Context Processing & Reranking
            initial_ctxs = item.get("ctxs", [])
            for one in initial_ctxs:
                already_retrival_papers[one["title"].strip()] = one

            if not initial_ctxs:
                logger.warning(
                    f"{task_id}: Initial context list 'ctxs' is empty or missing."
                )
                # Decide: Fail, or proceed assuming generation doesn't need context?
                # Proceeding, but generation might fail if it expects context.
                item["ctxs"] = []
            else:
                logger.info(
                    f"{task_id}: Processing {len(initial_ctxs)} initial contexts."
                )
                item["ctxs"] = process_ctx(
                    initial_ctxs
                )  # Assuming process_ctx handles its errors

            if ranking_ce and item["ctxs"]:

                try:

                    reranked_ctxs = self._run_ctx_rerank(
                        item["input"], item["ctxs"], reranker_model_name
                    )

                    item["ctxs"] = reranked_ctxs  # Update item with reranked contexts
                    logger.info(
                        f"{task_id}: Reranking complete. {len(item['ctxs'])} contexts remaining."
                    )
                except (DataProcessingError, Exception) as e:
                    logger.error(
                        f"{task_id}: Reranking failed: {e}. Proceeding with potentially unranked contexts."
                    )
                    # Continue with original (but processed) contexts
            elif not item["ctxs"]:
                logger.warning(
                    f"{task_id}: Skipping reranking as there are no contexts."
                )

            # 2. Initial Generation
            item["output"] = item.get(
                "output", ""
            )  # Ensure output key exists, might be pre-filled
            item["initial_result"] = item.get("initial_result", "")  # Ensure key exists
            item["final_passages"] = item.get("final_passages", "")  # Ensure key exists

            try:
                generated_result, passages_str, _ = self.generate_response(
                    item,  # Pass the whole item, generate_response extracts needed fields
                    max_tokens=max_tokens,
                    llama3_chat=llama3_chat,  # Unused
                    task_name=task_name,
                    zero_shot=zero_shot,
                    model_name=effective_model_name,
                )

                if generated_result is None:
                    # Handle generation failure more gracefully
                    logger.error(
                        f"{task_id}: Initial generation failed (LLM returned None or error)."
                    )
                    # Decide: Stop pipeline, or try feedback/posthoc on empty string? Stop seems safer.
                    raise OpenScholarAPIError(
                        "Initial generation failed, cannot proceed."
                    )
                else:
                    # Clean and store results
                    item["output"] = self._clean_final_output(generated_result)
                    item["initial_result"] = item[
                        "output"
                    ]  # Store the first successful generation
                    item["final_passages"] = (
                        passages_str  # Store formatted passages used
                    )
                    logger.info(
                        f"{task_id}: Initial generation successful. Length: {len(item['output'])}"
                    )

            except (
                LLMCommunicationError,
                DataProcessingError,
                ValueError,
                OpenScholarAPIError,
            ) as e:
                logger.error(
                    f"{task_id}: Initial generation step failed critically: {e}"
                )
                raise  # Re-raise to be caught by the main try-except block

            # 3. Feedback Loop
            item["feedbacks"] = item.get("feedbacks", [])  # Ensure key exists
            if (
                use_feedback and item["output"]
            ):  # Only run feedback if there is an output to critique

                try:
                    feedbacks, _ = self.get_feedback(
                        item,  # Pass item containing 'input', 'output', 'final_passages'
                        llama3_chat=llama3_chat,  # Unused
                        max_tokens=max_tokens,
                        model_name=effective_model_name,
                    )
                    item["feedbacks"] = feedbacks
                    logger.info(
                        f"{task_id}: Generated {len(feedbacks)} feedback items."
                    )

                    if feedbacks:
                        num_iterations = min(len(feedbacks), feedback_num)
                        logger.info(
                            f"{task_id}: Starting feedback iteration loop for {num_iterations} items."
                        )
                        for feedback_idx, (feedback, new_query) in tqdm.tqdm(
                            enumerate(feedbacks[:num_iterations]),
                            total=num_iterations,
                            desc=f"{task_id} Feedback Loop",
                        ):

                            iteration_content_base = f"主题：[{user_query}]\n任务：论文写作 - 反思({feedback_idx+1}/{num_iterations})"

                            # Stylistic/organizational feedback -> Edit
                            if not new_query:

                                edited_answer = self._clean_final_output(edited_answer)

                                # Basic check to prevent replacing with much shorter/empty output
                                if (
                                    len(edited_answer) > len(item["output"]) * 0.5
                                ):  # Threshold: 50% of original length
                                    logger.info(
                                        f"{task_id}: Applying edit from feedback {feedback_idx+1}. Length: {len(edited_answer)}"
                                    )
                                    item["output"] = edited_answer
                                    item[f"edited_answer_{feedback_idx}"] = (
                                        edited_answer  # Store specific edit result
                                    )
                                else:
                                    logger.warning(
                                        f"{task_id}: Edit from feedback {feedback_idx+1} resulted in significantly shorter output ({len(edited_answer)} vs {len(item['output'])}). Edit rejected."
                                    )

                            else:  # Feedback requires new retrieval -> Retrieve, Rerank, Edit

                                logger.info(
                                    f"{task_id}: Feedback {feedback_idx+1} requires new retrieval for query: '{new_query[:50]}...'"
                                )

                                try:
                                    request_info = {
                                        "query": new_query,
                                        "task_id": task_id,
                                    }

                                    new_retriaval_info = await run_requests_parallel(
                                        request_info
                                    )
                                    new_papers = new_retriaval_info["ctxs"]

                                    already_retrival_papers = {}
                                    new_papers_deduped = []
                                    for one in new_papers:
                                        if (
                                            one["title"].strip()
                                            not in already_retrival_papers
                                        ):
                                            new_papers_deduped.append(one)
                                            already_retrival_papers[
                                                one["title"].strip()
                                            ] = one

                                    logger.info(
                                        "get new_papers: {}--{}".format(
                                            len(new_papers), len(new_papers_deduped)
                                        )
                                    )
                                    new_papers = relevance_evaluator.evaluate_and_rank(
                                        new_query,
                                        new_papers_deduped,
                                        top_n=new_feedback_docs,
                                    )
                                    logger.info(
                                        f"After relevance filter, rest doc num is :{len(new_papers)}"
                                    )

                                    # new_papers = await self.retrieve_papers_parallel(request_info)

                                    if new_papers:
                                        # Rerank new papers relative to the original query/input
                                        current_ctx_len = len(
                                            item.get("ctxs", [])
                                        )  # Get current length before extending

                                        new_passages_reranked = self._run_ctx_rerank(
                                            item["input"],
                                            new_papers,
                                            reranker_model_name,
                                        )

                                        new_passages_processed = process_ctx(
                                            new_passages_reranked
                                        )

                                        if new_passages_processed:
                                            edited_answer, _ = (
                                                self.edit_with_feedback_retrieval(
                                                    item,  # Contains current 'output'
                                                    feedback,
                                                    new_passages_processed,  # Pass processed new passages
                                                    current_ctx_len,  # Pass correct start index
                                                    max_tokens=max_tokens,
                                                    model_name=effective_model_name,
                                                )
                                            )
                                            edited_answer = self._clean_final_output(
                                                edited_answer
                                            )

                                            # Basic check for validity
                                            if (
                                                len(edited_answer)
                                                > len(item["output"]) * 0.5
                                            ):
                                                logger.info(
                                                    f"{task_id}: Applying edit with retrieval from feedback {feedback_idx+1}. Length: {len(edited_answer)}"
                                                )
                                                item["output"] = edited_answer
                                                item[
                                                    f"edited_answer_{feedback_idx}"
                                                ] = edited_answer
                                                # Add new contexts (top N) to the main context list
                                                item["ctxs"].extend(
                                                    new_passages_processed[: self.top_n]
                                                )
                                                # Update final_passages string to reflect added contexts (optional, could be expensive)
                                                # item["final_passages"] = self._format_passages(item["ctxs"][:self.top_n]) # Re-format all
                                            else:
                                                logger.warning(
                                                    f"{task_id}: Edit with retrieval from feedback {feedback_idx+1} resulted in significantly shorter output ({len(edited_answer)} vs {len(item['output'])}). Edit rejected."
                                                )
                                        else:
                                            logger.warning(
                                                f"{task_id}: Reranking/processing of new papers for feedback {feedback_idx+1} yielded no usable passages. Skipping edit."
                                            )

                                    else:
                                        logger.warning(
                                            f"{task_id}: New retrieval for feedback {feedback_idx+1} yielded no papers. Skipping edit."
                                        )

                                except (
                                    requests.exceptions.RequestException,
                                    DataProcessingError,
                                    Exception,
                                ) as e:
                                    logger.error(
                                        f"{task_id}: Error during retrieval/reranking step for feedback {feedback_idx+1}: {e}"
                                    )
                                    # Continue to next feedback item

                except (
                    LLMCommunicationError,
                    DataProcessingError,
                    ValueError,
                    Exception,
                ) as e:
                    logger.error(
                        f"{task_id}: Feedback generation/processing failed: {e}. Skipping remaining feedback steps."
                    )
                    # Continue pipeline without feedback improvements

            elif not item["output"]:
                logger.warning(
                    f"{task_id}: Skipping feedback loop because 'output' is empty."
                )

            # 4. Extract References and Identify Most Common
            logger.info(
                f"{task_id}: Extracting referenced context indices from final output."
            )
            item["ctxs_refered"] = []
            item["match_counts"] = {}
            most_common_paper_refered = None

            if item["output"] and item.get(
                "ctxs"
            ):  # Need output and contexts to map references
                all_matches = re.findall(r"\[(\d+)\]", item["output"])
                valid_indices = []
                max_ctx_index = len(item["ctxs"]) - 1

                for match in all_matches:
                    try:
                        idx = int(match)
                        if 0 <= idx <= max_ctx_index:
                            valid_indices.append(idx)
                        else:
                            logger.warning(
                                f"{task_id}: Found out-of-bounds reference index [{idx}] in output (max index: {max_ctx_index}). Ignoring."
                            )
                    except ValueError:
                        logger.warning(
                            f"{task_id}: Found non-integer reference format '[{match}]' in output. Ignoring."
                        )

                if valid_indices:
                    match_counts = Counter(valid_indices)
                    item["match_counts"] = dict(match_counts)
                    # Get unique indices, sorted for consistency
                    unique_indices = sorted(list(set(valid_indices)))
                    item["ctxs_refered"] = [item["ctxs"][idx] for idx in unique_indices]

                    # Find most common referenced paper
                    if match_counts:
                        most_common_idx, most_common_count = match_counts.most_common(
                            1
                        )[0]
                        most_common_paper_refered = item["ctxs"][most_common_idx]
                        logger.info(
                            f"{task_id}: Identified {len(unique_indices)} unique referenced contexts. Most common: index [{most_common_idx}] (count: {most_common_count})."
                        )
                else:
                    logger.info(
                        f"{task_id}: No valid reference indices found in the output."
                    )
            else:
                logger.info(
                    f"{task_id}: Skipping reference extraction due to empty output or contexts."
                )

            # ADD STEP: 分析优化数据的格式，是否包含冗余，重复的内容，数据中可以包含 有序/无序编号，不可以包含段落标题

            try:
                markdown_text = item.get("output", "")
                if not markdown_text:
                    logger.warning(
                        f"{task_id}: Output is empty. Skipping markdown refinement."
                    )
                else:
                    refined_text = self.refine_markdown_content(markdown_text,model_name=self.model_name)
                    if refined_text:
                        logger.info("refine text done.")
                        item["output"] = refined_text
                    else:
                        logger.warning(
                            f"{task_id}: Markdown refinement returned empty content."
                        )
            except (
                LLMCommunicationError,
                DataProcessingError,
                ValueError,
                Exception,
            ) as e:
                logger.error(
                    f"{task_id}: refine markdown content failed: {e}. Proceeding with output before attribution."
                )

            # 5. Post-Hoc Attribution
            if (
                posthoc_at and item["output"] and item["ctxs_refered"]
            ):  # Only run if output and referred contexts exist

                try:
                    # Use the specific method that uses ctxs_refered
                    attributed_results, _ = (
                        self.insert_attributions_posthoc_paragraph_all(
                            item,  # Pass item, method expects 'output' and 'ctxs_refered'
                            max_tokens=int(
                                max_tokens * 1.5
                            ),  # Allow more tokens for attribution
                            llama3_chat=llama3_chat,  # Unused
                            model_name=effective_model_name,
                        )
                    )
                    item["output"] = self._clean_final_output(
                        attributed_results
                    )  # Clean after attribution
                    logger.info(
                        f"{task_id}: Post-hoc attribution complete. Final length: {len(item['output'])}"
                    )

                except (
                    LLMCommunicationError,
                    DataProcessingError,
                    ValueError,
                    Exception,
                ) as e:
                    logger.error(
                        f"{task_id}: Post-hoc attribution failed: {e}. Proceeding with output before attribution."
                    )
                    # Output remains as it was before this step

            elif posthoc_at:
                logger.warning(
                    f"{task_id}: Skipping post-hoc attribution because output or referenced contexts are missing."
                )

            # 6. Final Output Cleaning (already done after generation/edit/attribution)
            # item["output"] = self._clean_final_output(item["output"]) # Ensure one final clean
            logger.debug(f"{task_id}: Final output preview: {item['output'][:200]}...")

            # 7. Figure Generation Logic
            item["gen_main_figure_judgement_info"] = {}
            item["main_figure_base64"] = ""
            item["main_figure_caption"] = ""

            if section_name and most_common_paper_refered:
                logger.info(
                    f"{task_id}: Judging if section '{section_name}' needs a figure."
                )
                try:
                    needs_figure, judge_info = self.judge_section_shold_have_figure(
                        section_name
                    )
                    item["gen_main_figure_judgement_info"] = judge_info
                    needs_figure = False ## NOTE: CLOSE THE FIGURE INFO

                    if needs_figure:
                        logger.info(
                            f"{task_id}: Section judged to need a figure. Attempting extraction from most referenced paper (index {most_common_idx})."
                        )

                        try:
                            # Ensure most_common_paper_refered is the dict, not just index
                            figure_info = get_arxiv_main_figure(
                                most_common_paper_refered,
                                llm_provider="local",
                                image_extraction_model=image_extraction_model,
                            )  # Consider config for provider
                            if figure_info:
                                ## check figure_info["main_figure_caption"] valid
                                caption_valid = True
                                if (
                                    "main_figure_caption" in figure_info
                                    and figure_info["main_figure_caption"]
                                ):
                                    try:
                                        prompt = validate_figure_caption_prompt.format(
                                            caption=figure_info["main_figure_caption"],
                                            section_name=section_name,
                                        )

                                        validation_response = self._call_llm(
                                            [{"role": "user", "content": prompt}],
                                            model_name=self.model_name,
                                            temperature=0.1,
                                            max_tokens=100,
                                        )
                                        # 解析验证结果
                                        parsed_response = (
                                            self._parse_llm_output(validation_response)
                                            .strip()
                                            .lower()
                                        )
                                        caption_valid = (
                                            "yes" in parsed_response
                                            and "no" not in parsed_response
                                        )

                                        logger.info(
                                            f"{task_id}: Figure caption validation result: {caption_valid}. Response: {parsed_response}"
                                        )
                                    except Exception as val_e:
                                        logger.warning(
                                            f"{task_id}: Error validating figure caption: {val_e}. Proceeding with original caption."
                                        )
                                else:
                                    caption_valid = False
                                    logger.warning(
                                        f"{task_id}: No figure caption found in extracted figure info."
                                    )

                                if caption_valid:
                                    item.update(
                                        figure_info
                                    )  # Adds 'main_figure_url', 'main_figure_caption' etc.
                                    logger.info(
                                        f"{task_id}: Successfully extracted main figure information."
                                    )
                            else:
                                logger.warning(
                                    f"{task_id}: Figure extraction attempt did not yield a figure URL."
                                )
                        except Exception as fig_e:
                            logger.error(
                                f"{task_id}: Error during figure extraction: {fig_e}\n{traceback.format_exc()}"
                            )
                    else:
                        logger.info(f"{task_id}: Section judged not to need a figure.")
                except Exception as judge_e:
                    logger.error(
                        f"{task_id}: Error during figure judgment: {judge_e}\n{traceback.format_exc()}"
                    )
            else:
                logger.info(
                    f"{task_id}: Skipping figure judgment/extraction (no section name or most common paper)."
                )

            # 8. Format Reflection and Reformatting (New Step)
            if format_reflection and item["output"]:

                try:
                    reformatted_output, _ = self._reflect_and_reformat_section(
                        item,
                        model_name=effective_model_name,  # Use the same model? Or a specific one?
                        max_tokens=max_tokens,  # Reuse max_tokens or use a different value?
                    )
                    logger.info(f"reformatted_output: {reformatted_output}")
                    # Check if the output changed significantly (optional, basic length check)
                    if len(reformatted_output) > 0.5 * len(
                        item["output"]
                    ):  # Avoid replacing with empty/very short
                        item["output"] = (
                            reformatted_output  # Already cleaned by the method
                        )
                        logger.info(
                            f"{task_id}: Format reflection and potential reformatting applied."
                        )
                    else:
                        logger.warning(
                            f"{task_id}: Format reflection resulted in significantly shorter output. Change rejected."
                        )

                except Exception as reflect_e:
                    logger.error(
                        f"{task_id}: Format reflection step failed: {reflect_e}. Proceeding with previous output."
                    )
                    logger.error(traceback.format_exc())
                    # Output remains as it was before this step
            elif format_reflection:
                logger.warning(
                    f"{task_id}: Skipping format reflection because output is empty."
                )

            # Pipeline finished successfully

            logger.info(f"{task_id}: RAG pipeline finished successfully.")
            return item

        except Exception as e:
            # --- Fallback Logic ---
            logger.error(
                f"{task_id}: Critical error encountered in RAG pipeline: {e}\n{traceback.format_exc()}"
            )

            # Attempt to provide some fallback output if possible
            fallback_output = item.get("output") or item.get(
                "initial_result"
            )  # Use last known good output
            if fallback_output:
                logger.warning(
                    f"{task_id}: Attempting to return last known output as fallback."
                )
                try:
                    # Basic cleaning of the fallback output
                    cleaned_fallback = self._clean_final_output(fallback_output)
                    item["output"] = cleaned_fallback
                    # Ensure essential keys exist, even if empty/default
                    item["ctxs_refered"] = item.get("ctxs_refered", [])
                    item["feedbacks"] = item.get("feedbacks", [])
                    item["match_counts"] = item.get("match_counts", {})
                    # Add an error flag/message to the item
                    item["error_info"] = f"Pipeline failed with error: {e}"

                    return item
                except Exception as fallback_e:
                    logger.error(
                        f"{task_id}: Error during fallback processing: {fallback_e}\n{traceback.format_exc()}"
                    )

                    return None  # Indicate complete failure
            else:
                logger.error(f"{task_id}: No usable output available for fallback.")

                return None  # Indicate complete failure

    def _reflect_and_reformat_section(
        self,
        item: Dict[str, Any],
        model_name: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> Tuple[str, float]:
        """
        Analyze the section content's format and rewrite if necessary, preserving citations.

        Args:
            item: Dictionary containing the current 'output'.
            model_name: Specific model name to use.
            max_tokens: Maximum tokens for the reformatted response.

        Returns:
            Tuple of (potentially reformatted text, API cost (currently 0)).
            Returns original text on failure.
        """
        task_id = getattr(self, "task_id", "unknown_task")  # Get task_id if available
        logger.info(
            f"{task_id}: Running format reflection and potential reformatting..."
        )
        original_content = item.get("output", "")
        if not original_content:
            logger.warning(
                f"{task_id}: Skipping format reflection as content is empty."
            )
            return "", 0.0

        effective_model_name = model_name or self.model_name
        prompt = format_reflection_prompt.format(
            section_content=original_content,
            RESPONSE_START_DELIMITER=RESPONSE_START_DELIMITER,
            RESPONSE_END_DELIMITER=RESPONSE_END_DELIMITER,
        )

        messages = [
            {
                "role": "system",
                "content": chat_system,
            },  # Or a more specific system prompt for editing?
            {"role": "user", "content": prompt},
        ]

        try:
            outputs = self._call_llm(
                messages,
                model_name=effective_model_name,
                temperature=0.5,  # Lower temperature for more deterministic reformatting
                max_tokens=max_tokens,
            )

            logger.debug(
                f"{task_id}: Raw format reflection response: {outputs[:200]}..."
            )
            reformatted_content = self._parse_llm_output(outputs)

            # Basic check: Ensure output is not empty if original wasn't
            if not reformatted_content and original_content:
                logger.warning(
                    f"{task_id}: Format reflection resulted in empty output. Returning original content."
                )
                return original_content, 0.0

            # Optional: Add a check to see if citations were drastically changed (complex)
            # For now, we rely on the prompt constraint.

            logger.info(f"{task_id}: Format reflection complete.")
            # Clean the potentially reformatted output one last time
            return self._clean_final_output(reformatted_content), 0.0

        except LLMCommunicationError as e:
            logger.error(
                f"{task_id}: LLM communication failed during format reflection: {e}. Returning original content."
            )
            return original_content, 0.0
        except Exception as e:
            logger.error(
                f"{task_id}: Unexpected error during format reflection: {e}\n{traceback.format_exc()}. Returning original content."
            )
            return original_content, 0.0

    def _clean_final_output(self, text: Optional[str]) -> str:
        """Standard cleaning for final output text."""
        if text is None:
            return ""
        text = text.replace(RESPONSE_START_DELIMITER, "").replace(
            RESPONSE_END_DELIMITER, ""
        )
        text = text.replace(REVISED_ANSWER_HEADER, "")  # Remove edit header if present
        # Remove references section more robustly
        text = re.split(r"\n#{1,3}\s*" + REFERENCES_HEADER, text, maxsplit=1)[0]
        # Remove potential citation-only lines at the end? (Optional, might be too aggressive)
        # lines = text.strip().split('\n')
        # if lines and re.fullmatch(r"^\s*\[\d+(?:,\s*\d+)*\]\s*$", lines[-1]):
        #     lines.pop()
        # text = "\n".join(lines)
        return text.strip()

    def _get_flowchart_data(
        self, section_index: str, stage: str, iter_index: Optional[int] = None
    ) -> Optional[Dict]:
        """Generates flowchart data for flow_information_sync based on stage."""
        # This function centralizes the complex flowchart dictionary creation.
        # It's simplified here; you'd need to map each stage string to the correct nested structure.
        # Example for a couple of stages:
        base = {"name": f"{section_index}: Essay Writing Start"}
        search = {"name": f"{section_index}: Academic paper search"}
        generate = {"name": f"{section_index}: Paper content generation"}
        rerank = {"name": f"{section_index}: Recall content rerank"}
        first_draft = {"name": f"{section_index}: Generate first draft content"}
        reflect = {"name": f"{section_index}: Reflect on the generated content"}
        research_adjust = {"name": f"{section_index}: Re-search and content adjustment"}
        format_adjust = {"name": f"{section_index}: Format Adjustment"}
        format_reflect = {
            "name": f"{section_index}: Format Reflection & Adjustment"
        }  # <-- 新节点
        figure_extract = {"name": f"{section_index}: Core Figure Extraction"}  # Added
        finish = f"{section_index}: Section Finished"  # String for final node

        try:
            if stage == "rerank_start":
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            elif stage == "generate_start":
                rerank["children"] = [first_draft]
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            elif stage == "feedback_start":
                first_draft["children"] = [reflect]
                rerank["children"] = [first_draft]
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            elif stage == "feedback_iter_start" or stage == "feedback_retrieval_start":
                reflect["children"] = [research_adjust]
                first_draft["children"] = [reflect]
                rerank["children"] = [first_draft]
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            elif stage == "posthoc_start":
                research_adjust["children"] = [format_adjust]
                reflect["children"] = [research_adjust]
                first_draft["children"] = [reflect]
                rerank["children"] = [first_draft]
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            elif (
                stage == "figure_extract_start"
            ):  # Assuming figure happens after format adjust
                format_adjust["children"] = [figure_extract]
                research_adjust["children"] = [format_adjust]
                reflect["children"] = [research_adjust]
                first_draft["children"] = [reflect]
                rerank["children"] = [first_draft]
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            elif stage == "format_reflect_start":  # <-- 新 stage 处理
                # Assume it happens after figure extraction if that exists, otherwise after format_adjust/posthoc
                prev_node = (
                    figure_extract if "figure_extract" in locals() else format_adjust
                )  # Determine previous node
                prev_node["children"] = [format_reflect]
                # Reconstruct the tree up to this point (similar logic as 'finish')
                format_adjust["children"] = (
                    [figure_extract]
                    if "figure_extract" in locals()
                    else [format_reflect]
                )
                research_adjust["children"] = [format_adjust]
                reflect["children"] = [research_adjust]
                first_draft["children"] = [reflect]
                rerank["children"] = [first_draft]
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            elif stage == "finish":
                # Point to last known node, which could now be format_reflect
                last_node = (
                    format_reflect
                    if "format_reflect" in locals()
                    else (
                        figure_extract
                        if "figure_extract" in locals()
                        else format_adjust
                    )
                )
                last_node["children"] = [finish]
                # Reconstruct the full tree (ensure all potential paths are covered)
                figure_extract["children"] = (
                    [format_reflect] if "format_reflect" in locals() else [finish]
                )
                format_adjust["children"] = (
                    [figure_extract]
                    if "figure_extract" in locals()
                    else (
                        [format_reflect] if "format_reflect" in locals() else [finish]
                    )
                )
                research_adjust["children"] = [format_adjust]
                reflect["children"] = [research_adjust]
                first_draft["children"] = [reflect]
                rerank["children"] = [first_draft]
                generate["children"] = [rerank]
                search["children"] = [generate]
                base["children"] = [search]
                return base
            else:
                # Default or unknown stage - return None or a basic structure
                return None
        except Exception as e:
            logger.warning(f"Error generating flowchart data for stage '{stage}': {e}")
            return None  # Return None on error

    async def run(
        self, user_query: str, item: Dict[str, Any], **input_kwargs
    ) -> Optional[Dict[str, Any]]:
        """Entry point to run the RAG QA pipeline.

        Args:
            user_query: The original user query.
            item: Dictionary containing input query and contexts.
            **input_kwargs: Additional keyword arguments for the pipeline (request_id, task_id, etc.).

        Returns:
            Updated item dictionary or None if a critical error occurs.
        """
        # Validate required kwargs like request_id and task_id
        request_id = input_kwargs.get("request_id")
        task_id = input_kwargs.get("task_id")
        if not request_id or not task_id:
            logger.error(
                "Missing required arguments 'request_id' or 'task_id' in run method."
            )
            # Consider raising ValueError or returning None
            raise ValueError("Missing required arguments 'request_id' or 'task_id'")

        # Add any other pre-run checks here

        return await self.get_rag_qa(user_query, item, **input_kwargs)


class ResponseRagChat(BaseModel):
    request_id: str
    query: str
    status: int
    message: str = None
    ctx: List[Dict[str, Any]] = None
    output: str = None
    time_cost: Optional[Dict[str, Any]] = None
    main_figure_base64: str = None
    main_figure_caption: str = None


async def run_section_writer_actor(query,query_domain, task_id, model_info):
    try:
        request_id = f"SectionWriterActor-{uuid.uuid4().hex}"
        time_cost = {}
        start_time = time.time()
        data = {
            "query": query,
            "query_domain":query_domain,
            "task_id": task_id
        }
        for _ in range(5):
            retrival_result = await run_requests_parallel(data)
            if "ctxs" in retrival_result:
                break

        model_name = model_info.get("rag_model","Qwen3-8B")
        image_extraction_model = model_info.get("image_extraction_model", "Qwen3-8B")
        reranker_model_name = model_info.get("reranker_model_name", "Qwen3-8B")
        keyword_extraiction_model = image_extraction_model

        relevance_evaluator = DocumentRelevanceEvaluator(model_name=keyword_extraiction_model, max_workers=4)

        top_n = 20
        ctx_relevance_filtered = relevance_evaluator.evaluate_and_rank(
            query, retrival_result["ctxs"], top_n=top_n
        )

        retrival_result["ctxs"] = ctx_relevance_filtered
        retrieval_time = time.time()
        retrival_result["input"] = retrival_result["query"]
        retrival_result["answer"] = ""
        if "ctxs" not in retrival_result:
            use_contexts = False
        else:
            use_contexts = True


        # Initialize API (consider adding reranker model path if needed)
        api = OpenScholarAPI(
            online_retriver=True,
            use_contexts=use_contexts,
            top_n=top_n,
            api_model_name=model_name
        )

        for _ in range(4):
            try:
                # Run the pipeline
                result_item = await api.run(
                    user_query=retrival_result["query"],  # Pass the original query
                    item=retrival_result,
                    model_name=model_name,
                    image_extraction_model=image_extraction_model,
                    reranker_model_name=reranker_model_name,
                    request_id="example_req_001",
                    task_id=data["task_id"],
                    ranking_ce=False,  # Enable reranking
                    use_feedback=True,  # Enable feedback loop
                    posthoc_at=True,  # Enable post-hoc attribution
                    task_name="default",  # Or specify task like "summarization"
                    feedback_num=2,  # Limit feedback iterations
                    section_index="S1",  # Example section index
                    section_name="Challenges Section",  # Example section name for figure judgment
                )
                if result_item:
                    end_time = time.time()
                    time_cost.update(
                        {
                            "pipe_retrieval": retrieval_time - start_time,
                            "pipe_chat": end_time - retrieval_time,
                            "total": end_time - start_time,
                        }
                    )
                    result = ResponseRagChat(
                        query=query,
                        request_id=request_id,
                        status=200,
                        message="Success",
                        ctx=result_item["ctxs_refered"],
                        output=result_item["output"],
                        main_figure_base64=result_item.get("main_figure_base64", ""),
                        main_figure_caption=result_item.get("main_figure_caption", ""),
                        time_cost=time_cost,
                    )

                    return result
            except:
                logger.error(traceback.format_exc())
                pass
        return ResponseRagChat(
            query=query,
            request_id=request_id,
            status=502,
            message=f"Failed: Response generation failed",
        )
    except:
        return ResponseRagChat(
            query=query,
            request_id=request_id,
            status=500,
            message=f"Server error: {traceback.format_exc()}",
        )


# --- Example Usage ---
async def main_example():
    """Async example usage function."""
    logger.info("Starting OpenScholarAPI example...")
    from retrival import run_requests_parallel

    data = {
        "query": "What are the main challenges in applying reinforcement learning to robotics?",
        "task_id": "example_id",
    }
    query = data["query"]
    retrival_result = await run_requests_parallel(data)
    logger.info(f"Retrieval result: {retrival_result}")

    top_n = 20
    ctx_relevance_filtered = relevance_evaluator.evaluate_and_rank(
        query, retrival_result["ctxs"], top_n=top_n
    )

    retrival_result["ctxs"] = ctx_relevance_filtered

    retrival_result["input"] = retrival_result["query"]
    retrival_result["answer"] = ""
    try:
        # Initialize API (consider adding reranker model path if needed)

        model_name = "Qwen3-8B"
        image_extraction_model = "Qwen3-8B"
        reranker_model_name = "Qwen3-8B"
        api = OpenScholarAPI(
            online_retriver=True,
            use_contexts=use_contexts,
            top_n=top_n,
            api_model_name=model_name
        )
        # Run the pipeline
        result_item = await api.run(
            user_query=retrival_result["query"],  # Pass the original query
            item=retrival_result,
            model_name=model_name,
            image_extraction_model=image_extraction_model,
            reranker_model_name=reranker_model_name,
            request_id="example_req_001",
            task_id=data["task_id"],
            ranking_ce=True,  # Enable reranking
            use_feedback=True,  # Enable feedback loop
            posthoc_at=True,  # Enable post-hoc attribution
            task_name="default",  # Or specify task like "summarization"
            feedback_num=2,  # Limit feedback iterations
            section_index="S1",  # Example section index
            section_name="Challenges Section",  # Example section name for figure judgment
        )

        # Print results
        if result_item:
            logger.info("\n--- Pipeline Results ---")
            logger.info(f"all info: {result_item}")
            logger.info(f"Final Output:\n{result_item.get('output', 'N/A')}")
            logger.info(
                f"\nInitial Result:\n{result_item.get('initial_result', 'N/A')}"
            )
            logger.info(
                f"\nFeedbacks Generated: {len(result_item.get('feedbacks', []))}"
            )
            for i, (fb, q) in enumerate(result_item.get("feedbacks", [])):
                logger.info(
                    f"  Feedback {i+1}: {fb}" + (f" (New Query: {q})" if q else "")
                )
            logger.info(
                f"\nReferenced Contexts: {len(result_item.get('ctxs_refered', []))}"
            )
            logger.info(f"Reference Counts: {result_item.get('match_counts', {})}")
            logger.info(
                f"Needs Figure Judgment: {result_item.get('gen_main_figure_judgement_info', {})}"
            )
            logger.info(f"Main Figure URL: {result_item.get('main_figure_url', 'N/A')}")
            logger.info(
                f"Main Figure Caption: {result_item.get('main_figure_caption', 'N/A')}"
            )
            if "error_info" in result_item:
                logger.warning(
                    f"Pipeline finished with error: {result_item['error_info']}"
                )

        else:
            logger.error("Pipeline execution failed critically.")

    except Exception as e:
        logger.error(
            f"An error occurred during the example execution: {e}\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    # Run the async example function
    # asyncio.run(main_example())
    pass
