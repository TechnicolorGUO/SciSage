# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : LLM-based document relevance evaluation and ranking
# ==================================================================

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
try:
    from generation.local_request_v2 import get_from_llm
    from log import logger
except:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document structure with title and text content"""
    title: str
    text: str
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = str(hash(self.title + self.text))


@dataclass
class RelevanceResult:
    """Result of relevance evaluation"""
    document: Document
    relevance_score: float
    reasoning: Optional[str] = None
    rank: Optional[int] = None


class DocumentRelevanceEvaluator:
    """LLM-based document relevance evaluator"""

    def __init__(self, model_name: str = "Qwen3-32B", batch_size: int = 5, max_workers: int = 3):
        """
        Initialize the evaluator

        Args:
            model_name: Name of the LLM model to use
            batch_size: Number of documents to process in parallel
            max_workers: Maximum number of worker threads
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _create_relevance_prompt(self, query: str, document: dict) -> str:
        """Create prompt for LLM to evaluate relevance"""
        prompt = f"""You are an expert at evaluating document relevance. Given a query and a document, evaluate how relevant the document is to the query.

Please provide:
1. A relevance score between 0.0 and 1.0 (where 0.0 means completely irrelevant and 1.0 means perfectly relevant)
2. A brief reasoning for your score

Query: "{query}"

Document:
Title: {document['title']}
Content: {document['text']}

Please respond in the following JSON format:
{{
    "relevance_score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your scoring>"
}}

Focus on:
- Semantic similarity between query and document content
- Whether the document answers or addresses the query
- Topical overlap and conceptual relevance
- Quality and depth of information provided

Be precise and objective in your evaluation."""

        return prompt

    def _parse_llm_response(self, response: str) -> Tuple[float, str]:
        """Parse LLM response to extract score and reasoning"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                score = float(result.get('relevance_score', 0.0))
                reasoning = result.get('reasoning', 'No reasoning provided')

                # Ensure score is in valid range
                score = max(0.0, min(1.0, score))
                return score, reasoning

            # Fallback: try to extract score from text
            score_match = re.search(r'(?:score|relevance)[:\s]*([0-9]*\.?[0-9]+)', response.lower())
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
                return score, "Score extracted from text response"

            logger.warning(f"Could not parse LLM response: {response[:200]}...")
            return 0.0, "Failed to parse response"

        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return 0.0, f"Parse error: {str(e)}"

    def _evaluate_single_document(self, query: str, document: dict) -> RelevanceResult:
        """Evaluate relevance for a single document"""
        try:
            prompt = self._create_relevance_prompt(query, document)
            response = get_from_llm(prompt, model_name=self.model_name)
            score, reasoning = self._parse_llm_response(response)
            document["relevance_score_rude"] = score
            document["relevance_score_reasoning"] = reasoning
            return document
        except Exception as e:
            logger.error(f"Error evaluating document {document.doc_id}: {str(e)}")
            document["relevance_score_rude"] = 0
            document["relevance_score_reasoning"] = f"Error: {str(e)}"
            return document

    def evaluate_relevance_batch(self, query: str, documents: List[Dict]) -> List[RelevanceResult]:
        """Evaluate relevance for multiple documents using parallel processing"""
        logger.info(f"Evaluating relevance for {len(documents)} documents with query: '{query[:100]}...'")

        results = []

        # Process documents in parallel batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_doc = {
                executor.submit(self._evaluate_single_document, query, doc): doc
                for doc in documents
            }

            # Collect results as they complete
            for future in as_completed(future_to_doc):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per request
                    results.append(result)
                    logger.debug(f"Evaluated document {result['title']}: score={result['relevance_score_rude']:.3f}")
                except Exception as e:
                    logger.error(f"Failed to evaluate document {traceback.forma_exc()}")

        return results

    def rank_and_filter(self, results: List[RelevanceResult], top_n: int = 10) -> List[RelevanceResult]:
        """Sort results by relevance score and return top N"""
        # Sort by relevance score in descending order
        sorted_results = sorted(results, key=lambda x: x["relevance_score_rude"], reverse=True)

        # Add rank information
        for i, result in enumerate(sorted_results):
            result["reevance_rerank"] = i + 1

        # Return top N results
        top_results = sorted_results[:top_n]

        logger.info(f"Ranked {len(results)} documents, returning top {len(top_results)}")
        if top_results:
            logger.info(f"Score range: {top_results[0]['relevance_score_rude']:.3f} - {top_results[-1]['relevance_score_rude']:.3f}")

        return top_results

    def evaluate_and_rank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """Complete pipeline: evaluate relevance, rank, and filter top N documents"""
        start_time = time.time()

        # Evaluate relevance for all documents
        relevance_results = self.evaluate_relevance_batch(query, documents)

        # Rank and filter top N
        top_results = self.rank_and_filter(relevance_results, top_n)

        elapsed_time = time.time() - start_time
        logger.info(f"Completed relevance evaluation in {elapsed_time:.2f} seconds")

        return top_results


# Example usage
if __name__ == "__main__":
    # Sample documents
    sample_docs = [
        {
            "title": "Introduction to Machine Learning",
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
            "doc_id": "doc1"
        },
        {
            "title": "Deep Learning Fundamentals",
            "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "doc_id": "doc2"
        },
        {
            "title": "Natural Language Processing",
            "text": "NLP is a field of AI that focuses on the interaction between computers and humans using natural language.",
            "doc_id": "doc3"
        }
    ]

    # # Initialize evaluator
    # relevance_evaluator = DocumentRelevanceEvaluator(model_name="Qwen3-8B", max_workers=4)

    # # Evaluate and rank - now returns List[Dict]
    # query = "What is machine learning?"
    # results = relevance_evaluator.evaluate_and_rank(query, sample_docs, top_n=15)

    # # Print results
    # print(f"\nTop results for query: '{query}'")
    # print("=" * 60)
    # for doc in results:
    #     print(f"Rank {doc['reevance_rerank']}: {doc['title']}")
    #     print(f"Score: {doc['relevance_score_rude']:.3f}")
    #     print(f"Reasoning: {doc.get('relevance_score_reasoning', 'N/A')}")
    #     print("-" * 40)