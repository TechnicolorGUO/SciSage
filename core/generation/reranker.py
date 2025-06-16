# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : LLM-based paper reranking system
# ==================================================================

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback
from log import logger
from generation.local_request_v2 import get_from_llm

# 重排序提示模板
RERANK_TEMPLATE = """You are an expert academic researcher tasked with ranking research papers based on their relevance, authority, and timeliness for a given query.

Query: "{query}"

Papers to rank (ordered by original retrieval score):
{papers}

Please evaluate each paper based on the following criteria:
1. **Relevance (40%)**: How well does the paper's title, abstract, and content match the query?
2. **Authority (35%)**: Consider citation count, venue prestige, and author reputation.
3. **Timeliness (25%)**: More recent papers are generally preferred, but groundbreaking older papers may rank higher.

For each paper, provide ONLY the individual scores (1-10) for each criterion:

Output format for each paper:
```
Paper ID: id
Relevance: score/10
Authority: score/10
Timeliness: score/10
Justification: brief explanation
```

DO NOT calculate Final Score - this will be computed automatically.

Then provide the ranking based on your overall assessment:
```
FINAL RANKING:
1. Paper ID: id
2. Paper ID: id
3. Paper ID: id
...
```

Please be objective and consider the academic quality and impact of each paper.
"""


def calculate_final_score(relevance: float, authority: float, timeliness: float) -> float:
    """
    Calculate final score using predefined weights.

    Args:
        relevance: Relevance score (1-10)
        authority: Authority score (1-10)
        timeliness: Timeliness score (1-10)

    Returns:
        Final weighted score (1-10)
    """
    # Weights for different criteria
    RELEVANCE_WEIGHT = 0.4
    AUTHORITY_WEIGHT = 0.35
    TIMELINESS_WEIGHT = 0.25

    final_score = relevance * RELEVANCE_WEIGHT + authority * AUTHORITY_WEIGHT + timeliness * TIMELINESS_WEIGHT

    return round(final_score, 2)


def extract_scores_from_response_robust(response: str, num_papers: int) -> List[Tuple[int, float]]:
    """
    More robust version of extract_scores_from_response with multiple fallback patterns.
    Now extracts individual scores and calculates final score locally.

    Args:
        response: LLM response text
        num_papers: Expected number of papers

    Returns:
        List of (paper_index, final_score) tuples
    """
    scores = []

    try:
        # Method 1: Extract individual scores and calculate final score locally
        paper_scores = {}

        # Pattern to extract individual scores for each paper
        paper_pattern = r"Paper ID:\s*(\d+).*?Relevance:\s*([\d.]+).*?Authority:\s*([\d.]+).*?Timeliness:\s*([\d.]+)"
        matches = re.findall(paper_pattern, response, re.DOTALL)

        if matches and len(matches) == num_papers:
            logger.info("Extracting individual scores and calculating final score locally")

            for paper_id_str, relevance_str, authority_str, timeliness_str in matches:
                paper_id = int(paper_id_str) - 1  # Convert to 0-based index

                if 0 <= paper_id < num_papers:
                    try:
                        relevance = float(relevance_str.split("/")[0])  # Handle "8/10" format
                        authority = float(authority_str.split("/")[0])
                        timeliness = float(timeliness_str.split("/")[0])

                        # Calculate final score locally
                        final_score = calculate_final_score(relevance, authority, timeliness)
                        paper_scores[paper_id] = {
                            "final_score": final_score,
                            "relevance": relevance,
                            "authority": authority,
                            "timeliness": timeliness,
                        }

                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing scores for paper {paper_id}: {e}")
                        continue

            if len(paper_scores) == num_papers:
                # Sort by final score descending
                scores = sorted(
                    paper_scores.items(),
                    key=lambda x: x[1]["final_score"],
                    reverse=True,
                )
                # Convert to expected format
                scores = [(paper_id, score_data["final_score"]) for paper_id, score_data in scores]
                logger.info("Successfully extracted individual scores and calculated final scores")
                return scores

        # Method 2: Try to extract from FINAL RANKING section and use ranking order
        ranking_pattern = r"FINAL RANKING:(.*?)(?:\n\n|\Z)"
        ranking_match = re.search(ranking_pattern, response, re.DOTALL)

        if ranking_match:
            ranking_section = ranking_match.group(1)
            # Pattern: "1. Paper ID: 3"
            rank_pattern = r"(\d+)\.\s*Paper ID:\s*(\d+)"
            matches = re.findall(rank_pattern, ranking_section)

            if matches and len(matches) == num_papers:
                logger.info("Using FINAL RANKING order to assign scores")
                scores = []
                for rank_str, paper_id_str in matches:
                    paper_id = int(paper_id_str) - 1  # Convert to 0-based index
                    rank = int(rank_str)

                    if 0 <= paper_id < num_papers:
                        # Assign score based on ranking (higher rank = lower score)
                        # Top ranked paper gets 10, last gets 1
                        score = 10.0 - ((rank - 1) * 9.0 / (num_papers - 1))
                        scores.append((paper_id, round(score, 2)))

                if len(scores) == num_papers:
                    logger.info("Successfully extracted ranking-based scores")
                    return scores

        # Method 3: Fallback - try to extract any numerical scores
        logger.warning("Trying fallback score extraction methods")

        # Look for any Paper ID followed by numerical values
        fallback_pattern = r"Paper ID:\s*(\d+).*?(\d+(?:\.\d+)?)"
        matches = re.findall(fallback_pattern, response, re.DOTALL)

        if matches and len(matches) >= num_papers:
            paper_scores = {}
            for paper_id_str, score_str in matches[:num_papers]:
                paper_id = int(paper_id_str) - 1
                score = float(score_str)

                if 0 <= paper_id < num_papers:
                    paper_scores[paper_id] = score

            if len(paper_scores) == num_papers:
                scores = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
                logger.info("Extracted scores using fallback method")
                return scores

        logger.warning("All extraction methods failed")
        return []

    except Exception as e:
        logger.error(f"Error in robust score extraction: {e}")
        return []


def rerank_papers_with_llm(
    papers: List[Dict[str, Any]],
    query: str,
    model_name: str = "Qwen3-32B-long-ctx",
    max_papers: int = 20,
    max_retries: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank papers using LLM based on relevance, authority, and timeliness.
    Final score is calculated locally, not by LLM.

    Args:
        papers: List of paper dictionaries
        query: The search query
        model_name: Name of the LLM model to use
        max_papers: Maximum number of papers to rerank (for efficiency)
        max_retries: Maximum number of retries for LLM calls

    Returns:
        List of reranked papers
    """
    logger.info("rerank_papers_with_llm")

    if not papers:
        logger.warning("No papers to rerank")
        return []

    # Limit number of papers for efficiency
    papers_to_rank = papers[:max_papers]
    logger.info(f"Reranking {len(papers_to_rank)} papers with {model_name}")

    try:
        # Format papers for LLM
        formatted_papers = format_papers_for_ranking(papers_to_rank)
        logger.info(f"Formatted {len(papers_to_rank)} papers for ranking")

        # Create prompt
        prompt = RERANK_TEMPLATE.format(query=query, papers=formatted_papers)
        logger.info(f"Sending rerank request to {model_name}")

        # Get LLM response with retry mechanism
        response = None
        scores = []
        individual_scores = {}  # Store individual scores for debugging

        for attempt in range(max_retries):
            try:
                logger.info(f"LLM rerank attempt {attempt + 1}/{max_retries}")

                # Get LLM response
                response = get_from_llm(prompt, model_name=model_name)

                if not response:
                    logger.warning(f"Attempt {attempt + 1}: Empty response from LLM")
                    continue

                logger.info(f"Received response from {model_name} on attempt {attempt + 1}")
                logger.debug(f"LLM Response: {response[:1000]}...")

                # Try to extract individual scores first
                paper_pattern = r"Paper ID:\s*(\d+).*?Relevance:\s*([\d.]+).*?Authority:\s*([\d.]+).*?Timeliness:\s*([\d.]+)"
                matches = re.findall(paper_pattern, response, re.DOTALL)

                if matches and len(matches) == len(papers_to_rank):
                    logger.info("Successfully extracted individual scores")

                    calculated_scores = []
                    for (
                        paper_id_str,
                        relevance_str,
                        authority_str,
                        timeliness_str,
                    ) in matches:
                        paper_id = int(paper_id_str) - 1

                        if 0 <= paper_id < len(papers_to_rank):
                            try:
                                # Parse scores (handle formats like "8/10" or "8")
                                relevance = float(relevance_str.split("/")[0])
                                authority = float(authority_str.split("/")[0])
                                timeliness = float(timeliness_str.split("/")[0])

                                # Calculate final score locally
                                final_score = calculate_final_score(relevance, authority, timeliness)

                                calculated_scores.append((paper_id, final_score))
                                individual_scores[paper_id] = {
                                    "relevance": relevance,
                                    "authority": authority,
                                    "timeliness": timeliness,
                                    "final_score": final_score,
                                }

                                logger.debug(f"Paper {paper_id}: R={relevance}, A={authority}, T={timeliness}, Final={final_score}")

                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error parsing scores for paper {paper_id}: {e}")
                                continue

                    if len(calculated_scores) == len(papers_to_rank):
                        # Sort by final score descending
                        scores = sorted(calculated_scores, key=lambda x: x[1], reverse=True)
                        logger.info(f"Successfully calculated final scores for {len(scores)} papers")
                        break

                # Fallback: try the robust extraction method
                scores = extract_scores_from_response_robust(response, len(papers_to_rank))

                if scores and len(scores) == len(papers_to_rank):
                    logger.info(f"Successfully parsed {len(scores)} scores on attempt {attempt + 1}")
                    break
                elif scores:
                    logger.warning(f"Attempt {attempt + 1}: Parsed {len(scores)} scores but expected {len(papers_to_rank)}")
                else:
                    logger.warning(f"Attempt {attempt + 1}: Could not extract valid scores from response")

                # Add delay between retries (exponential backoff)
                if attempt < max_retries - 1:
                    import time

                    delay = min(2**attempt, 10)  # Max 10 seconds delay
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Add delay between retries
                if attempt < max_retries - 1:
                    import time

                    delay = min(2**attempt, 10)
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)

        # Check if we successfully got valid scores
        if not scores or len(scores) != len(papers_to_rank):
            logger.error(f"Failed to get valid scores after {max_retries} attempts")

            # Try with a simpler prompt as last resort
            logger.info("Trying with simplified prompt as last resort...")
            scores = try_simplified_rerank(papers_to_rank, query, model_name)

            if not scores:
                logger.warning("All attempts failed, returning original order")
                return papers

        # Reorder papers based on calculated final scores
        reranked_papers = []

        # Add reranked papers with their calculated scores
        for paper_idx, final_score in scores:
            paper = papers_to_rank[paper_idx].copy()
            paper["rerank_score"] = final_score
            paper["original_rank"] = paper_idx + 1

            # Add individual scores if available
            if paper_idx in individual_scores:
                paper["relevance_score"] = individual_scores[paper_idx]["relevance"]
                paper["authority_score"] = individual_scores[paper_idx]["authority"]
                paper["timeliness_score"] = individual_scores[paper_idx]["timeliness"]

            reranked_papers.append(paper)

        # Add any remaining papers that weren't reranked
        remaining_papers = papers[len(papers_to_rank) :]
        for i, paper in enumerate(remaining_papers):
            paper = paper.copy()
            paper["rerank_score"] = 0.0
            paper["original_rank"] = len(papers_to_rank) + i + 1
            reranked_papers.append(paper)

        logger.info(f"Successfully reranked {len(scores)} papers")

        # Log ranking changes with individual scores
        for i, (paper_idx, final_score) in enumerate(scores[:5]):  # Show top 5
            original_rank = paper_idx + 1
            new_rank = i + 1
            title = papers_to_rank[paper_idx].get("title", "")[:50]

            if paper_idx in individual_scores:
                ind_scores = individual_scores[paper_idx]
                logger.info(f"Rank {original_rank} -> {new_rank}: {title}... " f"(Final: {final_score}, R: {ind_scores['relevance']}, " f"A: {ind_scores['authority']}, T: {ind_scores['timeliness']})")
            else:
                logger.info(f"Rank {original_rank} -> {new_rank}: {title}... (Score: {final_score})")

        return reranked_papers

    except Exception as e:
        logger.error(f"Critical error in reranking: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return papers


def format_papers_for_ranking(papers: List[Dict[str, Any]]) -> str:
    """Format papers for LLM input"""
    formatted_papers = []

    for i, paper in enumerate(papers):
        # Extract basic information
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", "")
        if isinstance(authors, list):
            authors = "; ".join([(author.get("name", str(author)) if isinstance(author, dict) else str(author)) for author in authors])

        abstract = paper.get("text", paper.get("abstract", ""))[:2000]  # Limit abstract length
        year = paper.get("year", paper.get("publicationYear", "Unknown"))
        venue = paper.get("venue", "")
        citation_count = paper.get("citationCount", "Unknown")

        paper_info = f"""
Paper ID: {i+1}
Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Citation Count: {citation_count}
Abstract: {abstract}...
---"""
        formatted_papers.append(paper_info)

    return "\n".join(formatted_papers)


def extract_scores_from_response(response: str, num_papers: int) -> List[Tuple[int, float]]:
    """Extract paper IDs and scores from LLM response"""
    scores = []

    try:
        # Try to extract from FINAL RANKING section
        ranking_pattern = r"FINAL RANKING:(.*?)(?:\n\n|\Z)"
        ranking_match = re.search(ranking_pattern, response, re.DOTALL)

        if ranking_match:
            ranking_section = ranking_match.group(1)
            # Pattern: "1. Paper ID: 3 - Score: 8.5"
            rank_pattern = r"\d+\.\s*Paper ID:\s*(\d+)\s*-\s*Score:\s*([\d.]+)"
            matches = re.findall(rank_pattern, ranking_section)

            if matches:
                for paper_id_str, score_str in matches:
                    paper_id = int(paper_id_str) - 1  # Convert to 0-based index
                    score = float(score_str)
                    if 0 <= paper_id < num_papers:
                        scores.append((paper_id, score))

                if len(scores) == num_papers:
                    return scores

        # Fallback: extract scores from individual paper evaluations
        logger.warning("Could not parse final ranking, trying individual scores")

        # Pattern: "Paper ID: 1" followed by scores
        paper_pattern = r"Paper ID:\s*(\d+).*?Final Score:\s*([\d.]+)"
        matches = re.findall(paper_pattern, response, re.DOTALL)

        if matches:
            paper_scores = {}
            for paper_id_str, score_str in matches:
                paper_id = int(paper_id_str) - 1  # Convert to 0-based index
                score = float(score_str)
                if 0 <= paper_id < num_papers:
                    paper_scores[paper_id] = score

            # Sort by score descending
            scores = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)

            if len(scores) == num_papers:
                return scores

        logger.warning("Could not extract scores from LLM response")
        return []

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.error(f"Response: {response[:500]}...")
        return []


def try_simplified_rerank(papers: List[Dict[str, Any]], query: str, model_name: str) -> List[Tuple[int, float]]:
    """
    Try a simplified reranking approach when the main method fails.

    Args:
        papers: List of paper dictionaries
        query: The search query
        model_name: Name of the LLM model to use

    Returns:
        List of (paper_index, score) tuples
    """
    try:
        logger.info("Attempting simplified rerank approach")

        # Create a much simpler prompt
        simplified_prompt = f"""
Query: "{query}"

Papers:
{format_papers_for_ranking(papers)}

Please rank these papers from 1 to {len(papers)} based on relevance to the query.
Output ONLY the ranking in this exact format:
1. Paper ID: X
2. Paper ID: Y
3. Paper ID: Z
...
"""
        response = get_from_llm(simplified_prompt, model_name=model_name)

        if not response:
            logger.error("Simplified rerank also failed to get response")
            return []

        logger.info("Got response from simplified rerank")

        # Extract ranking with simpler pattern
        rank_pattern = r"\d+\.\s*Paper ID:\s*(\d+)"
        matches = re.findall(rank_pattern, response)

        if matches and len(matches) == len(papers):
            scores = []
            # Assign scores based on ranking (higher rank = higher score)
            for rank, paper_id_str in enumerate(matches):
                paper_id = int(paper_id_str) - 1  # Convert to 0-based index
                if 0 <= paper_id < len(papers):
                    # Score decreases with rank (first gets highest score)
                    score = 10.0 - (rank * 9.0 / (len(papers) - 1))
                    scores.append((paper_id, score))

            if len(scores) == len(papers):
                logger.info(f"Simplified rerank successful with {len(scores)} scores")
                return scores

        logger.warning("Simplified rerank could not parse ranking")
        return []

    except Exception as e:
        logger.error(f"Simplified rerank failed: {e}")
        return []


# Update the main function to use the robust extraction
def extract_scores_from_response(response: str, num_papers: int) -> List[Tuple[int, float]]:
    """Extract paper IDs and scores from LLM response with fallback methods"""
    return extract_scores_from_response_robust(response, num_papers)


def rerank_papers_batch(
    papers: List[Dict[str, Any]],
    query: str,
    batch_size: int = 10,
    model_name: str = "Qwen3-32B",
) -> List[Dict[str, Any]]:
    """
    Rerank papers in batches for better performance with large lists.

    Args:
        papers: List of paper dictionaries
        query: The search query
        batch_size: Number of papers to rank in each batch
        model_name: Name of the LLM model to use

    Returns:
        List of reranked papers
    """
    if len(papers) <= batch_size:
        return rerank_papers_with_llm(papers, query, model_name)

    logger.info(f"Reranking {len(papers)} papers in batches of {batch_size}")

    all_reranked = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(papers)-1)//batch_size + 1}")
        try:
            reranked_batch = rerank_papers_with_llm(batch, query, model_name)
            all_reranked.extend(reranked_batch)
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add original batch if reranking fails
            for paper in batch:
                paper = paper.copy()
                paper["rerank_score"] = 0.0
                paper["original_rank"] = i + batch.index(paper) + 1
                all_reranked.append(paper)

    # Final sort by rerank_score
    all_reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    logger.info(f"Completed batch reranking for {len(all_reranked)} papers")
    return all_reranked


def simple_rule_based_rerank(papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Simple rule-based reranking as fallback when LLM fails.

    Args:
        papers: List of paper dictionaries
        query: The search query

    Returns:
        List of reranked papers
    """
    logger.info("Using rule-based reranking as fallback")

    current_year = datetime.now().year
    query_terms = set(query.lower().split())

    for paper in papers:
        score = 0.0
        # Relevance score based on term matching
        title = paper.get("title", "").lower()
        abstract = paper.get("text", paper.get("abstract", "")).lower()

        title_matches = sum(1 for term in query_terms if term in title)
        abstract_matches = sum(1 for term in query_terms if term in abstract)

        relevance_score = (title_matches * 2 + abstract_matches) / max(len(query_terms), 1)
        # Authority score based on citation count and venue
        citation_count = paper.get("citationCount", 0)
        authority_score = min(citation_count / 100, 1.0)  # Normalize to 0-1

        venue = paper.get("venue", "").lower()
        if any(term in venue for term in ["nature", "science", "cell", "nips", "icml", "iclr"]):
            authority_score += 0.3
        # Timeliness score
        try:
            year = int(paper.get("year", paper.get("publicationYear", current_year - 10)))
            years_old = current_year - year
            timeliness_score = max(0, 1 - years_old / 20)  # Decay over 20 years
        except:
            timeliness_score = 0.5

        # Weighted final score
        final_score = relevance_score * 0.4 + authority_score * 0.35 + timeliness_score * 0.25

        paper["rerank_score"] = final_score
        paper["relevance_score"] = relevance_score
        paper["authority_score"] = authority_score
        paper["timeliness_score"] = timeliness_score

    logger.info(f"paers with rerank: {papers}")
    # Sort by final score
    papers.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    return papers


import time


def rerank_papers_hybrid(
    papers: List[Dict[str, Any]],
    query: str,
    model_name: str = "Qwen3-14B",
    use_llm: bool = True,
    max_papers_for_llm: int = 20,
) -> List[Dict[str, Any]]:
    """
    Hybrid reranking: use LLM for top papers, rule-based for the rest.

    Args:
        papers: List of paper dictionaries
        query: The search query
        model_name: Name of the LLM model to use
        use_llm: Whether to use LLM reranking
        max_papers_for_llm: Maximum papers to send to LLM

    Returns:
        List of reranked papers
    """
    if not papers:
        return []
    ts = time.time()
    logger.info(f"Hybrid reranking for {len(papers)} papers, {use_llm}")

    try:
        if use_llm and len(papers) > 0:
            # Use LLM for top papers
            top_papers = papers[:max_papers_for_llm]
            remaining_papers = papers[max_papers_for_llm:]
            logger.info(f"remaining_papers: {remaining_papers}")
            # LLM reranking for top papers
            reranked_top = rerank_papers_with_llm(top_papers, query, model_name)
            # Rule-based reranking for remaining papers
            if remaining_papers:
                reranked_remaining = simple_rule_based_rerank(remaining_papers, query)
                # Adjust scores to be lower than LLM results
                for paper in reranked_remaining:
                    paper["rerank_score"] = paper.get("rerank_score", 0) * 0.8
            else:
                reranked_remaining = []

            # Combine results
            final_result = reranked_top + reranked_remaining

        else:
            # Use only rule-based reranking
            final_result = simple_rule_based_rerank(papers, query)
        te = time.time()
        logger.info(f"Hybrid reranking completed for {len(final_result)} papers, time cost is :{te-ts:.2f} seconds")

        return final_result

    except Exception as e:
        logger.error(f"Error in hybrid reranking: {e}")
        # Fallback to rule-based
        return simple_rule_based_rerank(papers, query)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample papers
    sample_papers = [
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani, Ashish; Shazeer, Noam",
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            "year": "2017",
            "venue": "NIPS",
            "citationCount": 50000,
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": "Devlin, Jacob; Chang, Ming-Wei",
            "text": "We introduce a new language representation model called BERT...",
            "year": "2018",
            "venue": "NAACL",
            "citationCount": 40000,
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "authors": "Brown, Tom B.; Mann, Benjamin",
            "text": "Recent work has demonstrated substantial gains on many NLP tasks...",
            "year": "2020",
            "venue": "NeurIPS",
            "citationCount": 30000,
        },
    ]

    query = "transformer language models"

    # Test hybrid reranking

    # reranked = rerank_papers_hybrid(sample_papers, query, use_llm=True)

    # logger.info(f"reranked: {reranked}")

    # print("Reranked papers:")
    # for i, paper in enumerate(reranked):
    #     print(f"{i+1}. {paper['title']} (Score: {paper.get('rerank_score', 0):.3f})")
