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
    get_query_type_classification_prompt,
    get_general_query_rewrite_prompt
)

from utils import safe_invoke
import asyncio
from models import QueryIntent
from local_request_v2 import get_from_llm


class QueryRewrite(BaseModel):
    original_query: str = Field(description="The original user query")
    rewritten_query: str = Field(
        description="The rewritten query with improved clarity"
    )
    needs_rewrite: bool = Field(description="Whether the query needed rewriting")
    explanation: str = Field(
        description="Explanation of changes made or why no changes were needed"
    )


class QueryTranslation(BaseModel):
    original_query: str = Field(description="The original user query")
    is_english: bool = Field(description="Whether the original query is in English")
    translated_query: str = Field(
        description="The query translated to English if needed"
    )
    detected_language: str = Field(
        description="The detected language of the original query"
    )


class QueryTypeClassification(BaseModel):
    query_type: str = Field(description="The type of query: 'academic' or 'general'")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation for the classification decision")


class QueryProcessingState(BaseModel):
    user_query: str = Field(description="The original user query")
    intent: Optional[QueryIntent] = Field(
        default=None, description="The detected intent of the query"
    )
    rewrite: Optional[QueryRewrite] = Field(
        default=None, description="The query rewrite information"
    )
    translation: Optional[QueryTranslation] = Field(
        default=None, description="The query translation information"
    )
    classification: Optional[QueryTypeClassification] = Field(
        default=None, description="The query type classification"
    )
    final_query: str = Field(
        default="", description="The final processed query ready for use"
    )
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered during processing"
    )
    retry_count: Dict[str, int] = Field(
        default_factory=lambda: {
            "detect_intent": 0,
            "rewrite_query": 0,
            "translate_query": 0,
            "classify_query_type": 0,
        },
        description="Count of retries for each processing step",
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for each step"
    )


def classify_query_type(state: QueryProcessingState) -> QueryProcessingState:
    """
    使用LLM判断查询类型：学术(academic)或通用(general)
    """
    logger.info("classify_query_type ...")

    try:
        llm = llm_map[DEFAULT_MODEL_FOR_QUERY_INTENT]
        classification_parser = PydanticOutputParser(
            pydantic_object=QueryTypeClassification
        )
        classification_prompt = get_query_type_classification_prompt(
            query=state.user_query,
            format_instructions=classification_parser.get_format_instructions(),
        )

        chain = classification_prompt | llm | classification_parser

        result = safe_invoke(
            chain_func=chain,
            inputs={},
            default_value=None,
            error_msg="Error in query type classification",
            max_retries=state.max_retries,
        )

        if result is not None:
            state.classification = result
            logger.info(
                f"Query classified as: {result.query_type} (confidence: {result.confidence})"
            )
            logger.info(f"Classification reasoning: {result.reasoning}")
            return state
        else:
            # 如果chain失败，使用fallback逻辑
            logger.warning("Chain failed, using fallback classification")
            raise Exception("Chain classification failed")

    except Exception as e:
        logger.warning(f"Query classification with chain failed: {e}")

        # 默认fallback：如果包含学术关键词则判断为academic，否则为general
        academic_keywords = [
            "algorithm",
            "model",
            "method",
            "research",
            "analysis",
            "学习",
            "算法",
            "模型",
            "方法",
            "研究",
            "分析",
            "技术",
            "理论",
            "框架",
            "deep learning",
            "machine learning",
            "neural network",
            "artificial intelligence",
            "AI",
            "自然语言处理",
            "计算机视觉",
            "数据挖掘",
            "论文",
            "paper",
            "transformer",
            "CNN",
            "RNN",
            "LSTM",
            "GAN",
            "optimization",
            "classification",
            "regression",
        ]

        query_lower = state.user_query.lower()
        is_academic = any(
            keyword.lower() in query_lower for keyword in academic_keywords
        )

        fallback_classification = QueryTypeClassification(
            query_type="academic" if is_academic else "general",
            confidence=0.7,
            reasoning=f"Fallback classification based on keyword matching. Academic keywords found: {is_academic}",
        )

        state.classification = fallback_classification
        logger.info(
            f"Using fallback classification: {fallback_classification.query_type}"
        )

        if result is None:
            state.errors.append(f"Failed to classify query type: {str(e)}")

    return state


def classify_query_type_standalone(query: str) -> Dict[str, Any]:
    """
    独立的查询类型分类函数，可以单独调用

    Args:
        query: 用户查询

    Returns:
        分类结果字典
    """
    try:
        llm = llm_map[DEFAULT_MODEL_FOR_QUERY_INTENT]
        classification_parser = PydanticOutputParser(
            pydantic_object=QueryTypeClassification
        )
        classification_prompt = get_query_type_classification_prompt(
            query=query,
            format_instructions=classification_parser.get_format_instructions(),
        )

        chain = classification_prompt | llm | classification_parser

        result = safe_invoke(
            chain_func=chain,
            inputs={},
            default_value=None,
            error_msg="Error in standalone query type classification",
            max_retries=3,
        )

        if result is not None:
            logger.info(f"Standalone query classification result: {result}")
            return {
                "query_type": result.query_type,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            }

    except Exception as e:
        logger.warning(f"Standalone query classification failed: {e}")

    # 默认fallback
    academic_keywords = [
        "algorithm",
        "model",
        "method",
        "research",
        "analysis",
        "学习",
        "算法",
        "模型",
        "方法",
        "研究",
        "分析",
        "技术",
        "理论",
        "框架",
        "deep learning",
        "machine learning",
        "neural network",
        "artificial intelligence",
        "AI",
        "自然语言处理",
        "计算机视觉",
        "数据挖掘",
        "论文",
        "paper",
    ]

    query_lower = query.lower()
    is_academic = any(keyword in query_lower for keyword in academic_keywords)

    fallback_result = {
        "query_type": "academic" if is_academic else "general",
        "confidence": 0.7,
        "reasoning": f"Fallback classification based on keyword matching",
    }

    logger.info(f"Using fallback classification: {fallback_result}")
    return fallback_result


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
        logger.info(
            f"Translated from {result.detected_language} to English: {result.translated_query}"
        )
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


def rewrite_general_query(state: QueryProcessingState) -> QueryProcessingState:
    """
    对一般类型查询进行重写，保持原语种不变
    """
    logger.info("rewrite_general_query")
    llm = llm_map[DEFAULT_MODEL_FOR_QUERY_INTENT]
    rewrite_parser = PydanticOutputParser(pydantic_object=QueryRewrite)

    # 为一般查询使用简化的重写提示，不涉及研究领域
    rewrite_prompt = get_general_query_rewrite_prompt(
        query=state.user_query,
        format_instructions=rewrite_parser.get_format_instructions(),
    )

    chain = rewrite_prompt | llm | rewrite_parser

    result = safe_invoke(
        chain_func=chain,
        inputs={},
        default_value=None,
        error_msg="Error in general query rewriting",
        max_retries=state.max_retries,
    )

    if result is None:
        # 如果重写失败，保持原查询
        state.rewrite = QueryRewrite(
            original_query=state.user_query,
            rewritten_query=state.user_query,
            needs_rewrite=False,
            explanation="Rewrite failed, keeping original query",
        )
        state.errors.append("Failed to rewrite general query")
    else:
        state.rewrite = result
        if result.needs_rewrite:
            logger.info(f"General query rewritten: {result.rewritten_query}")
            logger.info(f"Rewrite explanation: {result.explanation}")
        else:
            logger.info("General query did not need rewriting")

    return state


def create_query_processor_graph():
    """
    创建查询处理图，默认包含查询类型分类
    根据分类结果决定后续处理流程
    """
    workflow = StateGraph(QueryProcessingState)

    # Add nodes
    workflow.add_node("classify_query_type", classify_query_type)
    workflow.add_node("detect_language_and_translate", detect_language_and_translate)
    workflow.add_node("detect_query_intent", detect_query_intent)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("rewrite_general_query", rewrite_general_query)
    workflow.add_node("finalize_query", finalize_query)

    # Set entry point
    workflow.set_entry_point("classify_query_type")

    # Add conditional edges based on query type
    def route_after_classification(state: QueryProcessingState) -> str:
        if state.classification and state.classification.query_type == "academic":
            return "detect_language_and_translate"
        else:
            return "rewrite_general_query"

    workflow.add_conditional_edges(
        "classify_query_type",
        route_after_classification,
        {
            "detect_language_and_translate": "detect_language_and_translate",
            "rewrite_general_query": "rewrite_general_query",
        },
    )

    # Academic path
    workflow.add_edge("detect_language_and_translate", "detect_query_intent")
    workflow.add_edge("detect_query_intent", "rewrite_query")
    workflow.add_edge("rewrite_query", "finalize_query")

    # General path
    workflow.add_edge("rewrite_general_query", "finalize_query")

    workflow.add_edge("finalize_query", END)

    # Compile the workflow
    return workflow.compile()


def process_query(user_query: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    处理查询的主函数，优化后的逻辑：
    1. 首先进行查询类型分类
    2. 如果是academic类型：语种识别→翻译→意图检测→重写
    3. 如果是general类型：直接重写（保持原语种）

    Args:
        user_query: 用户原始查询
        max_retries: 每个步骤的最大重试次数
    """
    try:
        logger.info(f"Processing new query: {user_query}")

        # Initialize state
        initial_state = QueryProcessingState(
            user_query=user_query, max_retries=max_retries
        )

        # Create and run graph
        workflow = create_query_processor_graph()

        # Execute the graph
        start_time = time.time()
        result = workflow.invoke(initial_state)
        end_time = time.time()

        logger.info(
            f"Query processing completed in {end_time - start_time:.2f} seconds"
        )

        # Format the result for return
        output = {
            "original_query": user_query,
            "final_query": result["final_query"],
            "processing_time": f"{end_time - start_time:.2f}s",
            "errors": result["errors"] if result["errors"] else None,
        }

        # 查询类型分类结果（始终包含）
        if "classification" in result and result["classification"]:
            output["classification"] = {
                "query_type": result["classification"].query_type,
                "confidence": result["classification"].confidence,
                "reasoning": result["classification"].reasoning,
            }

        # 意图检测结果（仅academic类型查询有）
        if "intent" in result and result["intent"]:
            output["intent"] = {
                "research_field": result["intent"].research_field,
                "paper_type": result["intent"].paper_type,
                "topic": result["intent"].topic,
                "explanation": result["intent"].explanation,
            }

        # 翻译结果（仅academic类型查询有）
        if "translation" in result and result["translation"]:
            output["translation"] = {
                "detected_language": result["translation"].detected_language,
                "was_translated": not result["translation"].is_english,
                "translated_query": (
                    result["translation"].translated_query
                    if not result["translation"].is_english
                    else None
                ),
            }

        # 重写结果（所有查询都有）
        if "rewrite" in result and result["rewrite"]:
            output["rewrite"] = {
                "was_rewritten": result["rewrite"].needs_rewrite,
                "rewritten_query": (
                    result["rewrite"].rewritten_query
                    if result["rewrite"].needs_rewrite
                    else None
                ),
                "explanation": result["rewrite"].explanation,
            }

        return output

    except Exception as e:
        logger.error(f"Critical error in query processing: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "original_query": user_query,
            "final_query": user_query,
            "error": f"Critical error: {str(e)}",
        }


async def process_query_async(user_query: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    查询处理的异步版本

    Args:
        user_query: 用户原始查询
        max_retries: 每个步骤的最大重试次数

    Returns:
        包含处理结果和分析结果的字典
    """
    try:
        logger.info(f"Processing query asynchronously: {user_query}")
        # Run the synchronous function in a separate thread to avoid blocking
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: process_query(user_query, max_retries)
        )

        logger.info("Async query processing completed")
        return result

    except Exception as e:
        logger.error(f"Critical error in async query processing: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "original_query": user_query,
            "final_query": user_query,
            "error": f"Critical error in async processing: {str(e)}",
        }


def test():
    """测试查询处理的同步版本"""
    # Example queries to test
    test_queries = [
        "What are the latest advances in transformer models?",  # Academic
        "我想了解人工智能在医疗领域的应用",  # Academic (Chinese)
        "what is better, lstm or transformers for nlp tasks",  # Academic
        "2024年中国经济政策最新变化",  # General (Chinese)
        "今日股市行情分析",  # General
        "COVID-19 vaccine policy updates",  # General
        "深度学习在计算机视觉中的应用研究",  # Academic (Chinese)
        "苹果公司最新财报分析",  # General
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"PROCESSING QUERY: {query}")
        print("=" * 80)

        result = process_query(query)

        print("\nRESULT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 特别显示分类结果
        if "classification" in result:
            print(
                f"\n>>> Classification: {result['classification']['query_type']} "
                f"(confidence: {result['classification']['confidence']:.2f})"
            )
            print(f">>> Reasoning: {result['classification']['reasoning']}")


async def test_async():
    # Example queries to test
    test_queries = [
        "What are the latest advances in transformer models?",
        "我想了解人工智能在医疗领域的应用",
        "what is better, lstm or transformers for nlp tasks",
        "2024年中国经济政策最新变化",
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
    # pass
    # 测试完整流程
    # test()

    # 测试异步版本
    # asyncio.run(test_async())

    # 测试独立分类功能
    # standalone_result = classify_query_type_standalone("饮食和减肥的关系")
    # print(f"\nStandalone classification result: {standalone_result}")
