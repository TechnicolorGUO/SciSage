# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import asyncio
import configuration
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import traceback
import json
import re

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from log import logger
from model_factory import llm_map
from configuration import DEFAULT_MODEL_FOR_QUERY_INTENT
from prompt_manager import (
    get_intent_classification_prompt,
    get_language_detection_prompt,
    get_query_rewrite_prompt,
    QUERY_TYPE_CLASSIFICATION_TEMPLATE
)
from utils import safe_invoke
import asyncio
from models import QueryIntent
from local_request_v2 import get_from_llm



class QueryRewrite(BaseModel):
    original_query: str = Field(description="The original user query")
    rewritten_query: str = Field(description="The rewritten query with improved clarity")
    needs_rewrite: bool = Field(description="Whether the query needed rewriting")
    explanation: str = Field(description="Explanation of changes made or why no changes were needed")


class QueryTranslation(BaseModel):
    original_query: str = Field(description="The original user query")
    is_english: bool = Field(description="Whether the original query is in English")
    translated_query: str = Field(description="The query translated to English if needed")
    detected_language: str = Field(description="The detected language of the original query")



class QueryTypeClassification(BaseModel):
    query_type: str = Field(description="The type of query: 'academic' or 'general'")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation for the classification decision")


class QueryProcessingState(BaseModel):
    user_query: str = Field(description="The original user query")
    intent: Optional[QueryIntent] = Field(default=None, description="The detected intent of the query")
    rewrite: Optional[QueryRewrite] = Field(default=None, description="The query rewrite information")
    translation: Optional[QueryTranslation] = Field(default=None, description="The query translation information")
    final_query: str = Field(default="", description="The final processed query ready for use")
    classification: Optional[QueryTypeClassification] = Field(default=None, description="The query type classification")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during processing")
    retry_count: Dict[str, int] = Field(
        default_factory=lambda: {
            "detect_intent": 0,
            "rewrite_query": 0,
            "translate_query": 0,
        },
        description="Count of retries for each processing step",
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for each step")



def classify_query_type(state: QueryProcessingState) -> QueryProcessingState:
    """
    使用LLM判断查询类型：学术(academic)或通用(general)
    """
    logger.info("classify_query_type ...")

    try:
        prompt = QUERY_TYPE_CLASSIFICATION_TEMPLATE.format(query=state.user_query)

        for attempt in range(state.max_retries):
            try:

                response = get_from_llm(prompt, model_name="Qwen3-8B", temperature=0.3)
                # 提取JSON
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    result_dict = json.loads(json_match.group(0))

                    # 验证必需字段
                    if ("query_type" in result_dict and
                        result_dict["query_type"] in ["academic", "general"] and
                        "confidence" in result_dict and
                        "reasoning" in result_dict):

                        classification = QueryTypeClassification(
                            query_type=result_dict["query_type"],
                            confidence=float(result_dict["confidence"]),
                            reasoning=result_dict["reasoning"]
                        )

                        state.classification = classification
                        logger.info(f"Query classified as: {classification.query_type} (confidence: {classification.confidence})")
                        logger.info(f"Classification reasoning: {classification.reasoning}")
                        return state

            except Exception as e:
                logger.warning(f"Query classification attempt {attempt + 1} failed: {e}")
                state.retry_count["classify_query_type"] += 1
                continue

        # 默认fallback：如果包含学术关键词则判断为academic，否则为general
        academic_keywords = [
            "algorithm", "model", "method", "research", "analysis", "学习", "算法",
            "模型", "方法", "研究", "分析", "技术", "理论", "框架", "deep learning",
            "machine learning", "neural network", "artificial intelligence", "AI",
            "自然语言处理", "计算机视觉", "数据挖掘", "论文", "paper", "transformer",
            "CNN", "RNN", "LSTM", "GAN", "optimization", "classification", "regression"
        ]

        query_lower = state.user_query.lower()
        is_academic = any(keyword.lower() in query_lower for keyword in academic_keywords)

        fallback_classification = QueryTypeClassification(
            query_type="academic" if is_academic else "general",
            confidence=0.7,
            reasoning=f"Fallback classification based on keyword matching. Academic keywords found: {is_academic}"
        )

        state.classification = fallback_classification
        logger.info(f"Using fallback classification: {fallback_classification.query_type}")

    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        # 最终fallback：默认为学术查询
        state.classification = QueryTypeClassification(
            query_type="academic",
            confidence=0.5,
            reasoning="Classification failed, defaulting to academic"
        )
        state.errors.append(f"Failed to classify query type: {str(e)}")

    return state

def detect_language_and_translate(state: QueryProcessingState) -> QueryProcessingState:
    logger.info("detect_language_and_translate ...")
    llm = llm_map[DEFAULT_MODEL_FOR_QUERY_INTENT]
    translation_parser = PydanticOutputParser(pydantic_object=QueryTranslation)
    translation_prompt = get_language_detection_prompt(
        query=state.user_query,
        format_instructions=translation_parser.get_format_instructions(),
    )

    chain = translation_prompt | llm | translation_parser
    result = safe_invoke(
        chain_func=chain,
        inputs={},
        default_value=None,
        error_msg="Error in language detection and translation",
        max_retries=state.max_retries,
    )

    if result is None:
        state.errors.append("Failed to detect language and translate query")
        return state

    state.translation = result
    if not result.is_english:
        logger.info(f"Translated from {result.detected_language} to English: {result.translated_query}")
        state.user_query = result.translated_query

    return state


def detect_query_intent(state: QueryProcessingState) -> QueryProcessingState:
    logger.info("detect_query_intent ...")
    llm = llm_map[DEFAULT_MODEL_FOR_QUERY_INTENT]
    intent_parser = PydanticOutputParser(pydantic_object=QueryIntent)
    intent_prompt = get_intent_classification_prompt(
        query=state.user_query,
        format_instructions=intent_parser.get_format_instructions(),
    )

    chain = intent_prompt | llm | intent_parser
    result = safe_invoke(
        chain_func=chain,
        inputs={},
        default_value=None,
        error_msg="Error in intent detection",
        max_retries=state.max_retries,
    )

    if result is None:
        state.errors.append("Failed to detect query intent")
        return state

    state.intent = result
    logger.info(f"Detected intent: {result.research_field}, type: {result.paper_type}")

    return state


def rewrite_query(state: QueryProcessingState) -> QueryProcessingState:
    logger.info("rewrite_query")
    llm = llm_map[DEFAULT_MODEL_FOR_QUERY_INTENT]
    rewrite_parser = PydanticOutputParser(pydantic_object=QueryRewrite)
    rewrite_prompt = get_query_rewrite_prompt(
        query=state.user_query,
        research_domain=state.intent.research_field,
        query_type=state.intent.paper_type,
        format_instructions=rewrite_parser.get_format_instructions(),
    )

    chain = rewrite_prompt | llm | rewrite_parser

    result = safe_invoke(
        chain_func=chain,
        inputs={},
        default_value=None,
        error_msg="Error in query rewriting",
        max_retries=state.max_retries,
    )

    if result is None:
        state.errors.append("Failed to rewrite query")
        return state

    state.rewrite = result
    if result.needs_rewrite:
        logger.info(f"Query rewritten: {result.rewritten_query}")
        logger.info(f"Rewrite explanation: {result.explanation}")
    else:
        logger.info("Query did not need rewriting")

    return state


def finalize_query(state: QueryProcessingState) -> QueryProcessingState:
    # Prioritize the rewritten query if available
    if state.rewrite and state.rewrite.needs_rewrite:
        state.final_query = state.rewrite.rewritten_query
    # Otherwise use the translated query if translation was needed
    elif state.translation and not state.translation.is_english:
        state.final_query = state.translation.translated_query
    # Otherwise use the original query
    else:
        state.final_query = state.user_query

    logger.info(f"Final processed query: {state.final_query}")
    return state


def create_query_processor_graph():
    workflow = StateGraph(QueryProcessingState)

    # Add nodes
    workflow.add_node("detect_language_and_translate", detect_language_and_translate)
    workflow.add_node("detect_query_intent", detect_query_intent)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("finalize_query", finalize_query)

    # Add edges (sequential workflow)
    workflow.set_entry_point("detect_language_and_translate")
    workflow.add_edge("detect_language_and_translate", "detect_query_intent")
    workflow.add_edge("detect_query_intent", "rewrite_query")
    workflow.add_edge("rewrite_query", "finalize_query")
    workflow.add_edge("finalize_query", END)

    # Compile the workflow
    return workflow.compile()


# Main function to process a query
def process_query(user_query: str, max_retries: int = 3) -> Dict[str, Any]:
    try:
        logger.info(f"Processing new query: {user_query}")

        # Initialize state
        initial_state = QueryProcessingState(user_query=user_query, max_retries=max_retries)

        # Create and run graph
        workflow = create_query_processor_graph()

        # Execute the graph
        start_time = time.time()
        result = workflow.invoke(initial_state)  # result is likely a dictionary
        end_time = time.time()

        logger.info(f"Query processing completed in {end_time - start_time:.2f} seconds")

        logger.info(f"result: {result}")
        # Format the result for return
        output = {
            "original_query": user_query,
            "final_query": result["final_query"],  # Access as a dictionary
            "processing_time": f"{end_time - start_time:.2f}s",
            "errors": result["errors"] if result["errors"] else None,
        }

        if "intent" in result and result["intent"]:
            output["intent"] = {
                "research_field": result["intent"].research_field,
                "paper_type": result["intent"].paper_type,
                "topic": result["intent"].topic,
                "explanation": result["intent"].explanation,
            }

        if "translation" in result and result["translation"]:
            output["translation"] = {
                "detected_language": result["translation"].detected_language,
                "was_translated": not result["translation"].is_english,
                "translated_query": (result["translation"].translated_query if not result["translation"].is_english else None),
            }

        if "rewrite" in result and result["rewrite"]:
            output["rewrite"] = {
                "was_rewritten": result["rewrite"].needs_rewrite,
                "rewritten_query": (result["rewrite"].rewritten_query if result["rewrite"].needs_rewrite else None),
                "explanation": result["rewrite"].explanation,
            }

        return output

    except Exception as e:
        logger.error(f"Critical error in query processing: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "original_query": user_query,
            "final_query": user_query,  # Return original query as fallback
            "error": f"Critical error: {str(e)}",
        }


async def process_query_async(user_query: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Asynchronous version of the query processing function.

    Args:
        user_query: The user's original query
        max_retries: Maximum number of retries for each processing step

    Returns:
        A dictionary containing the processed query and analysis results
    """
    try:
        logger.info(f"Processing query asynchronously: {user_query}")
        # Run the synchronous function in a separate thread to avoid blocking
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: process_query(user_query, max_retries))

        logger.info("Async query processing completed")
        return result

    except Exception as e:
        logger.error(f"Critical error in async query processing: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "original_query": user_query,
            "final_query": user_query,  # Return original query as fallback
            "error": f"Critical error in async processing: {str(e)}",
        }


def test():
    # Example queries to test
    test_queries = [
        "What are the latest advances in transformer models?",
        "我想了解人工智能在医疗领域的应用",  # Chinese: "I want to learn about AI applications in healthcare"
        "what is better, lstm or transformers for nlp tasks",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"PROCESSING QUERY: {query}")
        print("=" * 80)

        result = process_query(query)

        print("\nRESULT:")

        print(json.dumps(result, indent=2, ensure_ascii=False))


async def test_aysnc():
    # Example queries to test
    test_queries = [
        "What are the latest advances in transformer models?",
        "我想了解人工智能在医疗领域的应用",  # Chinese: "I want to learn about AI applications in healthcare"
        "what is better, lstm or transformers for nlp tasks",
    ]

    # Process all queries concurrently
    tasks = [process_query_async(query) for query in test_queries]
    results = await asyncio.gather(*tasks)

    # Display the results
    for i, (query, result) in enumerate(zip(test_queries, results)):
        print("\n" + "=" * 80)
        print(f"PROCESSING QUERY {i+1}: {query}")
        print("=" * 80)
        print("\nRESULT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))


# Example usage
if __name__ == "__main__":
    # test()
    # asyncio.run(test_aysnc())
    pass
