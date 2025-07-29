# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

import json
import asyncio
import nest_asyncio
import time
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
import re
import sys
from bs4 import BeautifulSoup
from request_warp import RequestWrapper
from typing import List
from generation.instruction_pro import PAGE_REFINE_PROMPT, SIMILARITY_PROMPT

try:
    from log import logger
except:
    import logging

    logger = logging.getLogger(__name__)

nest_asyncio.apply()

proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}


class GeneralCrawer:
    # Configuration constants
    MAX_CONCURRENT_CRAWLS = 10
    MAX_CONCURRENT_PROCESSES = 10

    # Document processing constants
    DEFAULT_SIMILARITY_THRESHOLD = 80
    DEFAULT_MIN_LENGTH = 350
    DEFAULT_MAX_LENGTH = 20000

    def __init__(self, model="Qwen3-8B", infer_type="local"):
        """
        Initialize the AsyncCrawler.

        Args:
            model (str): Model identifier for text processing
            infer_type (str): Inference type, e.g., "OpenAI"
        """
        self.request_pool = RequestWrapper(model=model, infer_type=infer_type)

    async def run(
        self,
        topic: str,
        url_list: List[str],
        crawl_output_file_path: str = None,
        top_n: int = 10,
        save_to_file: bool = False,
    ):
        """
        Asynchronously crawls a list of URLs, processes the crawled data, and saves the results.
        The process is split into four stages:
        1. URL crawling
        2. Content filtering and title generation
        3. Similarity scoring
        4. Result processing and saving (optional)

        Args:
            topic (str): The topic or theme associated with the URLs
            url_list (List[str]): A list of URLs to crawl
            crawl_output_file_path (str, optional): The file path where the final processed results will be saved (required if save_to_file=True)
            top_n (int, optional): Maximum number of top results to save. Defaults to 80
            save_to_file (bool, optional): Whether to save results to file. If False, returns processed data directly. Defaults to True

        Returns:
            dict or None: If save_to_file=False, returns processed results dict. If save_to_file=True, returns None
        """
        process_start_time = time.time()
        stage_time = process_start_time
        logger.info(f"Starting crawling process for {len(url_list)} URLs")

        # Validate parameters
        if save_to_file and not crawl_output_file_path:
            raise ValueError(
                "crawl_output_file_path is required when save_to_file=True"
            )

        # Stage 1: Concurrent URL crawling
        results = await self._crawl_urls(topic, url_list)
        logger.info(
            f"Stage 1 - Crawling completed in {time.time() - stage_time:.2f} seconds, with {len(results)} results"
        )
        stage_time = time.time()

        # Stage 2: Concurrent content filtering and title generation
        results = await self._process_filter_and_titles(results)
        logger.info(
            f"Stage 2 - Content filtering and title generation completed in {time.time() - stage_time:.2f} seconds, with {len(results)} results"
        )
        stage_time = time.time()

        # Stage 3: Concurrent similarity scoring
        results = await self._process_similarity_scores(results)
        logger.info(
            f"Stage 3 - Similarity scoring completed in {time.time() - stage_time:.2f} seconds, with {len(results)} results"
        )
        stage_time = time.time()

        # Stage 4: Result processing
        processed_results = self._process_results_internal(
            results,
            top_n=top_n,
            save_to_file=save_to_file,
            output_path=crawl_output_file_path,
        )

        logger.info(
            f"Stage 4 - Results processing completed in {time.time() - stage_time:.2f} seconds"
        )
        logger.info(
            f"Total processing completed in {time.time() - process_start_time:.2f} seconds"
        )

        # Return results if not saving to file
        if not save_to_file:
            return processed_results

        return None

    def _process_results_internal(
        self,
        results,
        top_n=80,
        similarity_threshold=None,
        min_length=None,
        max_length=None,
        save_to_file=True,
        output_path=None,
    ):
        """
        Internal method to process crawling results and optionally save to file.

        Args:
            results: Raw crawling results
            top_n: Maximum number of documents to keep for each topic
            similarity_threshold: Similarity threshold
            min_length: Minimum document length
            max_length: Maximum document length
            save_to_file: Whether to save to file
            output_path: Output file path (required if save_to_file=True)

        Returns:
            dict: Processed results if save_to_file=False, None otherwise
        """
        # Use default values if not provided
        if similarity_threshold is None:
            similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD
        if min_length is None:
            min_length = self.DEFAULT_MIN_LENGTH
        if max_length is None:
            max_length = self.DEFAULT_MAX_LENGTH

        # Step 1: Process each paper data serially
        processed_data = []
        for data in results:
            try:
                # Build paper data
                paper_data = {
                    "title": data["title"],
                    "url": data["url"],
                    "txt": data["filtered"],
                    "similarity": data.get("similarity", 0),
                }
                processed_data.append((data["topic"], paper_data))
            except Exception as e:
                logger.error(f"Failed to process paper data: {e}")
                continue

        # Step 2: Organize data by topic
        topics = {}
        for topic, paper_data in processed_data:
            topics.setdefault(topic, []).append(paper_data)

        # Step 3: Filter papers for each topic
        final_results = {}
        for topic, papers in topics.items():
            filtered_papers = self._filter_papers(
                papers,
                similarity_threshold,
                min_length,
                max_length,
                top_n,
            )
            final_results[topic] = {
                "title": topic,
                "papers": filtered_papers,
                "total_papers": len(papers),
                "filtered_papers_count": len(filtered_papers),
            }

        # Step 4: Save to file or return results
        if save_to_file:
            if not output_path:
                raise ValueError("output_path is required when save_to_file=True")

            with open(output_path, "w", encoding="utf-8") as outfile:
                for topic, topic_data in final_results.items():
                    output_data = {
                        "title": topic_data["title"],
                        "papers": topic_data["papers"],
                    }
                    json.dump(output_data, outfile, ensure_ascii=False)
                    outfile.write("\n")

            logger.info(f"Processed data has been saved to {output_path}")
            return None
        else:
            logger.info(f"Returning processed data for {len(final_results)} topics")
            return final_results

    # Update the original _process_results method to maintain backward compatibility
    def _process_results(
        self,
        results,
        output_path,
        top_n=5,
        similarity_threshold=None,
        min_length=None,
        max_length=None,
    ):
        """
        Process crawling results and save to file (backward compatibility).
        """
        return self._process_results_internal(
            results=results,
            top_n=top_n,
            similarity_threshold=similarity_threshold
            or self.DEFAULT_SIMILARITY_THRESHOLD,
            min_length=min_length or self.DEFAULT_MIN_LENGTH,
            max_length=max_length or self.DEFAULT_MAX_LENGTH,
            save_to_file=True,
            output_path=output_path,
        )

    async def _process_similarity_score(self, data):
        """
        Calculate similarity score for a single piece of data.
        """
        try:
            # Calculate similarity score using SIMILARITY_PROMPT
            prompt = SIMILARITY_PROMPT.format(
                topic=data["topic"], content=data["filtered"]
            )
            res = self.request_pool.completion(prompt)

            # 修改正则表达式以支持浮点数
            score = re.search(r"<SCORE>(\d+(?:\.\d+)?)</SCORE>", res)
            if not score:
                raise ValueError("Invalid similarity score format")

            # 将分数转换为float，然后根据需要转换为int或保持float
            data["similarity"] = float(score.group(1).strip())

        except Exception as e:
            logger.info(f"Failed to process similarity score: {e}")
            data["error"] = True
            data["similarity"] = -1
        return data

    async def _process_filter_and_title(self, data):
        """
        Process title generation and content filtering for a single piece of data.
        """
        try:
            # Generate title and filter content using PAGE_REFINE_PROMPT
            prompt = PAGE_REFINE_PROMPT.format(
                topic=data["topic"], raw_content=data["raw_content"]
            )
            res = self.request_pool.completion(prompt)
            title = re.search(r"<TITLE>(.*?)</TITLE>", res, re.DOTALL)
            content = re.search(r"<CONTENT>(.*?)</CONTENT>", res, re.DOTALL)

            if not title or not content:
                raise ValueError(f"Invalid response format, response: {res}")

            data["title"] = title.group(1).strip()
            data["filtered"] = content.group(1).strip()
        except Exception as e:
            logger.error(f"Failed to process filter and title: {e}")
            data["title"] = "Error in filtering"
            data["filtered"] = f"Error in filtering ({e})"
            data["error"] = True
        return data

    async def _process_similarity_scores(self, results: List[dict]) -> List[dict]:
        """
        Calculate similarity scores for filtered results using pure producer-consumer pattern.
        """
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()
        total_items = len(results)

        # Producer: Add tasks to queue
        for data in results:
            await input_queue.put(data)

        async def consumer():
            while True:
                try:
                    data = input_queue.get_nowait()
                    try:
                        result = await self._process_similarity_score(data)
                        await output_queue.put(result)
                        logger.info(
                            f"Processed similarity score, remaining: {input_queue.qsize()}/{total_items}, URL: {data.get('url', 'N/A')}"
                        )
                    finally:
                        input_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        # Create and start consumers
        consumers = [
            asyncio.create_task(consumer())
            for _ in range(self.MAX_CONCURRENT_PROCESSES)
        ]

        # Wait for all tasks to be processed
        await input_queue.join()

        # Collect results
        results = []
        while not output_queue.empty():
            data = await output_queue.get()
            if data["error"]:
                logger.error(f"Error in processing data, skip: {data}")
            else:
                results.append(data)

        return results

    async def _process_filter_and_titles(self, results: List[dict]) -> List[dict]:
        """
        Process title generation and content filtering using pure producer-consumer pattern.
        """
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()
        total_items = len(results)

        # Producer: Add tasks to queue
        for data in results:
            await input_queue.put(data)

        async def consumer():
            while True:
                try:
                    data = input_queue.get_nowait()
                    try:
                        result = await self._process_filter_and_title(data)
                        await output_queue.put(result)
                        logger.info(
                            f"Title and filter processing completed, remaining: {input_queue.qsize()}/{total_items}, URL: {data.get('url', 'N/A')}"
                        )
                    finally:
                        input_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        # Create and start consumers
        consumers = [
            asyncio.create_task(consumer())
            for _ in range(self.MAX_CONCURRENT_PROCESSES)
        ]

        # Wait for all tasks to be processed
        await input_queue.join()

        # Collect results
        results = []
        while not output_queue.empty():
            data = await output_queue.get()
            if data["error"]:
                logger.error(f"Error in processing data, skip: {data}")
            else:
                results.append(data)

        return results

    async def _crawl_urls(self, topic: str, url_list: List[str]) -> List[dict]:
        """
        Crawl URLs using pure producer-consumer pattern.
        """
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()
        total_items = len(url_list)

        # Producer: Add URLs to queue
        for url in url_list:
            await input_queue.put((url, topic))

        async def consumer():
            while True:
                try:
                    url, topic = input_queue.get_nowait()
                    try:
                        result = await self._crawl_and_collect(url, topic)
                        await output_queue.put(result)
                        logger.info(
                            f"URL crawling completed, remaining: {input_queue.qsize()}/{total_items}, URL: {url}"
                        )
                    finally:
                        input_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        # Create and start consumers
        consumers = [
            asyncio.create_task(consumer()) for _ in range(self.MAX_CONCURRENT_CRAWLS)
        ]

        # Wait for all tasks to be processed
        await input_queue.join()

        # Collect results
        results = []
        while not output_queue.empty():
            data = await output_queue.get()
            if data["error"]:
                logger.error(f"Error in processing data, skip: {data}")
            else:
                results.append(data)

        return results

    async def _crawl_and_collect(self, url: str, topic: str) -> dict:
        """
        Crawl a single URL and collect its content.

        Args:
            url (str): URL to crawl
            topic (str): Associated topic for the URL

        Returns:
            dict: Dictionary containing crawled data and metadata
        """
        try:
            raw_content = await self._simple_crawl(url)
            data = {
                "topic": topic,
                "url": url,
                "raw_content": raw_content,
                "error": False,
            }
        except Exception as e:
            logger.error(f"Crawling failed for URL={url}: {e}")
            data = {
                "topic": topic,
                "url": url,
                "raw_content": f"Error: Crawling failed({e})",
                "error": True,
            }

        return data

    async def _simple_crawl(self, url: str) -> str:
        """
        Perform a simple crawl of a URL using AsyncWebCrawler.

        Args:
            url (str): URL to crawl

        Returns:
            str: Raw markdown content from the webpage
        """
        crawler_run_config = CrawlerRunConfig(
            page_timeout=180000, cache_mode=CacheMode.BYPASS  # 18s timeout
        )

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=crawler_run_config)
            raw_markdown = result.markdown.raw_markdown
            logger.info(f"Content length={len(raw_markdown)} for URL={url}")
            logger.info(f"raw_markdown: {raw_markdown}")
            return raw_markdown

    def _filter_papers(
        self,
        papers,
        similarity_threshold,
        min_length,
        max_length,
        top_n,
    ):
        """
        Filter papers based on criteria.

        Args:
            papers: List of papers to filter
            similarity_threshold: Minimum similarity score required
            min_length: Minimum document length
            max_length: Maximum document length
            minimal_length: Absolute minimum length allowed
            top_n: Maximum number of papers to return

        Returns:
            List of filtered papers
        """
        # Sort by similarity and length
        sorted_papers = sorted(papers, key=lambda x: (-x["similarity"], -len(x["txt"])))

        # Filter documents that are too short
        valid_length_papers = [
            p for p in sorted_papers if min_length <= len(p["txt"]) <= max_length
        ]

        # Filter documents by similarity and length requirements
        valid_similarity_papers = [
            p for p in valid_length_papers if p["similarity"] >= similarity_threshold
        ]

        # Add additional documents if needed to reach top_n
        if len(valid_similarity_papers) < top_n:
            remaining_papers = [
                p for p in valid_length_papers if p not in valid_similarity_papers
            ]
            valid_similarity_papers.extend(
                remaining_papers[: top_n - len(valid_similarity_papers)]
            )

        return valid_similarity_papers


import requests


def searxng_search(query, instance_url="https://searx.namejeff.xyz", num_results=5):
    search_url = f"{instance_url}/search"
    params = {
        "q": query,
        "format": "json",
        "language": "zh-CN",
        "categories": "news",
        "safesearch": 1,
    }

    try:
        response = requests.get(search_url, params=params, timeout=20)
        response.raise_for_status()
        results = response.json().get("results", [])
        return [r["url"] for r in results[:num_results] if "url" in r]
    except Exception as e:
        print("搜索错误:", e)
        return []


import random


def get_random_headers():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/117.0",
        "Mozilla/5.0",
        # 更多 user-agent 可添加
    ]
    return {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.bing.com/",
        "Connection": "keep-alive",
    }


def web_search(query, engine="auto", api_key=None, num_results=5, use_proxy=False):
    """
    统一的网络搜索接口，优先使用 Bing 搜索，失败时使用 SerpAPI

    Args:
        query (str): 搜索查询
        engine (str): 搜索引擎 ("auto", "bing", "serpapi")
        api_key (str, optional): SerpAPI 密钥
        num_results (int): 返回结果数量
        use_proxy (bool): 是否使用代理

    Returns:
        list: URL 列表
    """

    def bing_search_internal(query, num_results=5, use_proxy=False):
        """内部 Bing 搜索函数"""
        try:
            headers = get_random_headers()
            url = f"https://www.bing.com/search?q={query}"

            # 根据需要使用代理
            proxies_config = proxies if use_proxy else None

            res = requests.get(url, headers=headers, proxies=proxies_config, timeout=10)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")
            links = []

            for item in soup.select(".b_algo")[:num_results]:
                a_tag = item.find("a")
                if a_tag and a_tag.get("href"):
                    links.append(a_tag["href"])

            if not links:
                raise Exception("No results found from Bing search")

            return links

        except Exception as e:
            print(f"Bing 搜索失败: {e}")
            return None

    def serpapi_search_internal(
        query, api_key, engine="bing", num_results=5, use_proxy=False
    ):
        """内部 SerpAPI 搜索函数"""
        if not api_key:
            raise ValueError("SerpAPI 需要 API key，但未提供")

        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": api_key,
                "engine": engine,
                "num": num_results,
                "hl": "zh-cn",
            }

            # 根据需要使用代理
            proxies_config = proxies if use_proxy else None

            response = requests.get(
                url, params=params, timeout=10, proxies=proxies_config
            )
            response.raise_for_status()

            results = response.json().get("organic_results", [])
            urls = [r.get("link") for r in results if "link" in r]

            if not urls:
                raise Exception("No results found from SerpAPI")

            return urls[:num_results]

        except Exception as e:
            print(f"SerpAPI 搜索失败: {e}")
            return None

    # 自动模式：优先 Bing，失败时使用 SerpAPI
    print("尝试 Bing 搜索...")
    result = bing_search_internal(query, num_results, use_proxy)

    if result is not None:
        print(f"Bing 搜索成功，获得 {len(result)} 个结果")
        return result

    # Bing 失败，尝试 SerpAPI
    print("Bing 搜索失败，尝试 SerpAPI...")
    if not api_key:
        print("错误: SerpAPI 需要 API key，但未提供。搜索失败。")
        return []

    try:
        result = serpapi_search_internal(query, api_key, "bing", num_results, use_proxy)
        if result is not None:
            print(f"SerpAPI 搜索成功，获得 {len(result)} 个结果")
            return result
        else:
            print("SerpAPI 搜索也失败了")
            return []
    except ValueError as e:
        print(f"错误: {e}")
        return []

import os
async def search_and_crawl(
    query: str,
    output_file_path: str = None,
    url_list: List = None,
    top_n: int = 10,
    num_search_results: int = 10,
    search_engine: str = "auto",
    api_key: str = None,
    use_proxy: bool = False,
    crawler_model: str = "Qwen3-8B",
    crawler_infer_type: str = "local",
    similarity_threshold: int = 85,
    min_length: int = 350,
    max_length: int = 20000,
    save_to_file: bool = True,  # 新增参数
):
    """
    统一的搜索和爬虫函数，先进行网络搜索获取URL列表，然后使用GeneralCrawer解析内容

    Args:
        query (str): 搜索查询词
        output_file_path (str, optional): 输出文件路径（当save_to_file=True时必需）
        url_list (List, optional): 预设的URL列表，如果提供则跳过搜索步骤
        top_n (int): 最终保存的最佳结果数量，默认10
        num_search_results (int): 搜索返回的URL数量，默认10
        search_engine (str): 搜索引擎类型 ("auto", "bing", "serpapi")，默认"auto"
        api_key (str, optional): SerpAPI 密钥，当使用serpapi时需要
        use_proxy (bool): 是否使用代理，默认False
        crawler_model (str): 爬虫使用的模型，默认"Qwen3-8B"
        crawler_infer_type (str): 推理类型，默认"local"
        similarity_threshold (int): 相似度阈值，默认85
        min_length (int): 最小文档长度，默认350
        max_length (int): 最大文档长度，默认20000
        save_to_file (bool): 是否保存到文件，默认True

    Returns:
        dict: 包含处理结果的字典
    """
    start_time = time.time()

    if url_list is None:
        url_list = []

    try:
        logger.info(f"开始搜索和爬虫流程，查询: '{query}'")
        api_key = os.getenv("SERPAPI_API_KEY","xxx")

        # 第一步：网络搜索获取URL列表（如果没有提供url_list）
        if not url_list:
            logger.info(
                f"步骤1: 使用 {search_engine} 引擎搜索，目标获取 {num_search_results} 个URL"
            )
            url_list = web_search(
                query=query,
                engine=search_engine,
                api_key=api_key,
                num_results=num_search_results,
                use_proxy=use_proxy,
            )
            logger.info(f"搜索成功，获得 {len(url_list)} 个URL")
        else:
            logger.info(f"使用提供的URL列表: {len(url_list)} 个URL")

        if not url_list:
            error_msg = f"搜索失败：未找到任何URL结果，查询词: '{query}'"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "search_results_count": 0,
                "crawl_results_count": 0,
                "output_file": "",
                "query": query,
                "url_list": [],
                "processed_data": None,
            }

        # 第二步：使用GeneralCrawer爬取和处理内容
        logger.info(f"步骤2: 开始爬取和处理 {len(url_list)} 个URL")

        crawler = GeneralCrawer(model=crawler_model, infer_type=crawler_infer_type)

        # 运行爬虫和内容处理流程
        processed_data = await crawler.run(
            topic=query,
            url_list=url_list,
            crawl_output_file_path=output_file_path,
            top_n=top_n,
            save_to_file=save_to_file,
        )

        # 计算结果数量
        crawl_results_count = 0
        if save_to_file:
            # 如果保存到文件，从文件读取统计
            import os

            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                crawl_results_count += len(data.get("papers", []))
                except Exception as e:
                    logger.warning(f"无法统计爬虫结果数量: {e}")
                    crawl_results_count = -1
        else:
            # 如果不保存到文件，直接从返回的数据统计
            if processed_data:
                for topic_data in processed_data.values():
                    crawl_results_count += len(topic_data.get("papers", []))

        success_msg = f"搜索和爬虫流程成功完成！耗时: {time.time() - start_time:.2f}秒"
        logger.info(success_msg)

        return {
            "success": True,
            "message": success_msg,
            "search_results_count": len(url_list),
            "crawl_results_count": crawl_results_count,
            "output_file": output_file_path if save_to_file else "",
            "query": query,
            "url_list": url_list,
            "processed_data": processed_data if not save_to_file else None,  # 新增字段
        }

    except Exception as e:
        error_msg = f"搜索和爬虫流程发生错误: {str(e)}"
        logger.error(f"{error_msg}\n{e}")
        import traceback

        logger.error(traceback.format_exc())

        return {
            "success": False,
            "message": error_msg,
            "search_results_count": len(url_list) if "url_list" in locals() else 0,
            "crawl_results_count": 0,
            "output_file": "",
            "query": query,
            "url_list": url_list if "url_list" in locals() else [],
            "processed_data": None,
        }


# 新增便捷函数：只返回数据不保存文件
async def search_and_crawl_data_only(
    query: str,
    url_list: List = None,
    top_n: int = 10,
    num_search_results: int = 10,
    **kwargs,
):
    """
    搜索和爬虫函数，只返回处理后的数据，不保存到文件

    Args:
        query (str): 搜索查询词
        url_list (List, optional): 预设的URL列表
        top_n (int): 最终结果数量
        num_search_results (int): 搜索URL数量
        **kwargs: 其他参数传递给search_and_crawl

    Returns:
        dict: 处理结果，包含processed_data字段
    """
    return await search_and_crawl(
        query=query,
        output_file_path=None,
        url_list=url_list,
        top_n=top_n,
        num_search_results=num_search_results,
        save_to_file=False,
        **kwargs,
    )



# 将示例代码包装在异步函数中
async def main():
    url_list = [
        "https://www.zhihu.com/tardis/zm/art/678176440",
        "https://www.zhihu.com/question/35066454",
        "https://www.zhihu.com/question/41597534",
        "https://www.zhihu.com/tardis/zm/art/474721641",
        "https://www.zhihu.com/question/28289825",
    ]

    # 示例1：保存到文件（原有方式）
    result1 = await search_and_crawl(
        query="咖啡",
        output_file_path="./ai_results.jsonl",
        url_list=url_list,
        save_to_file=True
    )
    print(f"示例1结果: {result1}")

    # 示例2：不保存文件，直接返回数据
    result2 = await search_and_crawl(
        query="咖啡",
        url_list=url_list,
        save_to_file=False
    )
    # 访问处理后的数据
    processed_data2 = result2["processed_data"]
    print(f"示例2处理后的数据: {processed_data2}")

    # 示例3：使用便捷函数
    result3 = await search_and_crawl_data_only(
        query="咖啡",
        url_list=url_list,
        top_n=15
    )
    processed_data3 = result3["processed_data"]
    print(f"示例3处理后的数据: {processed_data3}")

# 运行示例
# if __name__ == "__main__":
#     # 方法1：使用 asyncio.run()
#     import asyncio
#     asyncio.run(main())

    # 方法2：在已有事件循环中运行（如果已经在异步环境中）
    # await main()