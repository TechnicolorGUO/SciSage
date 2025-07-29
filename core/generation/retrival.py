# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, List, Any
from generation.utils import flow_information_sync, keep_letters
from log import logger
from generation.websearch_scholar import (
    get_doc_info_from_api,
    search_paper_via_query_from_openalex,
)
import requests
from generation.global_config import recall_server_url
import traceback
from generation.generation_instructions import template_extract_keywords_source_aware
from generation.websearch_general import search_and_crawl_data_only

from local_request_v2 import get_from_llm
import re


proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}

# 初始化 FastAPI 应用
app = FastAPI()

# 使用何种retrival方式
RETRIVAL_TYPE = "bm25"  # bm25 emb
RETRIVAL_TYPE = "emb"  # bm25 emb
RECALL_DOCS = 5


# 模拟的请求处理函数 1（可以替换为实际的 API 调用或计算任务）
def retrival_offline(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {}
        query = data["query"]
        task_id = data.get("task_id", "example_id")
        retrival_inp = {
            "query": query,
            "domains": "local",
            "n_docs": RECALL_DOCS,
            "search_type": RETRIVAL_TYPE,
        }
        flow_information_sync(task_id, content="[本地数据库检索], 开始")

        retrival_response = requests.post(recall_server_url, json=retrival_inp).json()

        logger.debug(f"{task_id}: retrival_response: {retrival_response}")

        flow_information_sync(task_id, content="[本地数据库检索], 完成")

        if int(retrival_response["status"]) != 200:
            flow_information_sync(task_id, content="[本地数据库检索], 失败")
            return {}
        else:
            # 每个ctx中的数据格式为：
            """
            one = {
                "text": doc.get("text", ""),
                "title": doc.get("title", ""),
                "id": doc.get("id", ""),
                "source": doc.get("source", "Local Database Retrival"),
                "authors": doc.get("authors", ""),
                "abstract": doc.get("abstract", ""),
                "keywords": doc.get("Keywords", ""),
                "meta": {
                    "title": doc.get("title", ""),
                    "authors": doc.get("authors", ""),
                },
            }
            """
            output = {}
            for one in retrival_response["ctxs"]:
                if not one.get("title", "") or not one.get("abstract", ""):
                    logger.info(f"local retrival doc empty, skip: {one}")
                    continue
                one["text"] = one["abstract"]
                one["url"] = ""
                one["arxivUrl"] = ""
                one["arxivId"] = ""
                one["source"] = one.get("source", "Search From Local Database")
                one.pop("meta")
                output[keep_letters(one["title"])] = one
            return output
    except:
        traceback.print_exc()
        return {}


# 在线请求
def retrival_online(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        query = data["query"]
        task_id = data.get("task_id", "example_id")
        response = get_doc_info_from_api(query)
        if response:
            output = {}
            for _id, paper in response.items():
                paper["authors"] = ";".join(one["name"] for one in paper["authors"])
                paper["url"] = paper["arxivUrl"]
                paper["text"] = paper["abstract"]
                paper["year"] = paper.get("publicationYear", "Unknown")
                paper["venue"] = paper.get("journal_ref", "Unknown")
                paper["citationCount"] = paper.get("citationCount", "Unknown")
                output[keep_letters(paper["title"])] = paper
            return output
        else:
            return {}
    except:
        traceback.print_exc()
        return {}


def extract_keywords(query: str, source: str = "semantic") -> List[str]:
    """Extract keywords from query optimized for a specific source."""
    # LLM_MODEL_NAME = "Qwen3-32B"
    LLM_MODEL_NAME = "Qwen3-14B"
    query = query.lower()
    model_inp = template_extract_keywords_source_aware.format(
        user_query=query, source=source
    )
    for _ in range(10):
        try:
            response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
            pattern = r"\[Start\](.*?)\[End\]"
            match = re.search(pattern, response)
            if match:
                keywords = match.group(1).strip()
                logger.info(f"Extracted keywords for {source}: {keywords}")
                return [kw.strip() for kw in keywords.split(",") if kw.strip()]
        except:
            logger.error(f"Failed to extract keywords: {traceback.format_exc()}")
    return []


def retrival_online_openalex(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        logger.info(f"retrival online openalex: {data}")
        query = data["query"]
        task_id = data.get("task_id", "example_id")
        query2kwds = extract_keywords(query, source="openalex")
        logger.info(f"Extracted keywords for OpenAlex: {query2kwds}")
        output = {}
        for kwd in query2kwds[:3]:
            if not kwd.strip():
                logger.info(f"Keyword is empty, skip: {kwd}")
                continue
            response = search_paper_via_query_from_openalex(kwd)
            if response:
                for _id, paper in response.items():
                    paper["authors"] = ";".join(one["name"] for one in paper["authors"])
                    paper["url"] = paper["arxivUrl"]
                    paper["text"] = paper["abstract"]
                    paper["year"] = paper.get("publicationYear", "Unknown")
                    paper["venue"] = paper.get("venue", "Unknown")
                    paper["citationCount"] = paper.get("citationCount", "Unknown")
                    output[keep_letters(paper["title"])] = paper
        return output
    except:
        traceback.print_exc()
        return {}


def convert_crawl_results_to_ctxs(crawl_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将通用爬虫结果转换为标准的ctxs格式

    Args:
        crawl_results: search_and_crawl_data_only的返回结果

    Returns:
        标准格式的ctxs列表
    """
    ctxs = []

    if not crawl_results or not crawl_results.get("success"):
        return ctxs

    processed_data = crawl_results.get("processed_data", {})

    for topic, topic_data in processed_data.items():
        papers = topic_data.get("papers", [])

        for paper in papers:
            ctx = {
                "title": paper.get("title", ""),
                "text": paper.get("txt", ""),  # 爬虫结果中是txt字段
                "abstract": paper.get("txt", "")[:500] + "..." if len(paper.get("txt", "")) > 500 else paper.get("txt", ""),
                "url": paper.get("url", ""),
                "arxivUrl": "",  # 通用爬虫结果没有arxiv信息
                "arxivId": "",
                "authors": "",  # 通用爬虫结果通常没有作者信息
                "year": "Unknown",
                "venue": "Web Content",
                "source": "General Web Search",
                "citationCount": "Unknown",
                "similarity": paper.get("similarity", 0),
                "keywords": "",
                "id": paper.get("url", "").replace("/", "_").replace(":", "")  # 生成简单ID
            }
            ctxs.append(ctx)

    return ctxs

async def retrival_general_web(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用通用网络搜索和爬虫进行检索

    Args:
        data: 包含query和task_id的字典

    Returns:
        标准格式的检索结果
    """
    try:
        query = data["query"]
        task_id = data.get("task_id", "example_id")

        logger.info(f"Using general web search for query: {query}")

        # 调用通用搜索和爬虫
        crawl_result = await search_and_crawl_data_only(
            query=query,
            top_n=20,  # 获取更多结果
            num_search_results=15,  # 搜索更多URL
            search_engine="auto",
            use_proxy=False,
            crawler_model="Qwen3-8B",
            similarity_threshold=70  # 降低相似度阈值以获取更多相关内容
        )

        if not crawl_result.get("success"):
            logger.warning(f"General web search failed: {crawl_result.get('message', 'Unknown error')}")
            return {}

        # 转换为标准ctxs格式
        ctxs = convert_crawl_results_to_ctxs(crawl_result)

        # 转换为与学术检索相同的格式
        output = {}
        for ctx in ctxs:
            if ctx.get("title") and ctx.get("text"):
                key = keep_letters(ctx["title"])
                output[key] = ctx

        logger.info(f"General web search returned {len(output)} results")
        return output

    except Exception as e:
        logger.error(f"General web search failed: {traceback.format_exc()}")
        return {}

async def run_academic_retrieval(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行学术检索（原有逻辑）
    """
    try:
        t0 = time.time()
        loop = asyncio.get_event_loop()

        # 首先并行执行offline检索
        with ThreadPoolExecutor() as executor:
            future_offline = loop.run_in_executor(executor, retrival_offline, data)
            result_offline = await future_offline

        # 在线检索结果
        result_online = {}
        result_online_openalex = {}

        # 尝试retrival_online，最多5次
        online_success = False
        for attempt in range(10):
            try:
                logger.info(f"Attempting retrival_online, attempt {attempt + 1}/10")
                with ThreadPoolExecutor() as executor:
                    future_online = loop.run_in_executor(executor, retrival_online, data)
                    result_online = await future_online

                # 检查是否成功获取到数据
                if result_online and len(result_online) > 0:
                    logger.info(f"retrival_online succeeded on attempt {attempt + 1}")
                    online_success = True
                    break
                else:
                    logger.warning(f"retrival_online attempt {attempt + 1} returned empty result")

            except Exception as e:
                logger.error(f"retrival_online attempt {attempt + 1} failed: {str(e)}")

            # 如果不是最后一次尝试，等待一段时间再重试
            if attempt < 9:
                await asyncio.sleep(1)

        # 如果retrival_online失败，尝试retrival_online_openalex，最多3次
        if not online_success or len(result_online) < 15:
            logger.info("retrival_online failed after 10 attempts, trying retrival_online_openalex")

            for attempt in range(3):
                try:
                    logger.info(f"Attempting retrival_online_openalex, attempt {attempt + 1}/3")
                    with ThreadPoolExecutor() as executor:
                        future_openalex = loop.run_in_executor(executor, retrival_online_openalex, data)
                        result_online_openalex = await future_openalex

                    # 检查是否成功获取到数据
                    if result_online_openalex and len(result_online_openalex) > 0:
                        logger.info(f"retrival_online_openalex succeeded on attempt {attempt + 1}")
                        break
                    else:
                        logger.warning(f"retrival_online_openalex attempt {attempt + 1} returned empty result")

                except Exception as e:
                    logger.error(f"retrival_online_openalex attempt {attempt + 1} failed: {str(e)}")

                # 如果不是最后一次尝试，等待一段时间再重试
                if attempt < 2:
                    await asyncio.sleep(1)

        logger.info(f"offline retrival num is: {len(result_offline)}")
        logger.info(f"online retrival num is: {len(result_online)}")
        logger.info(f"online openalex retrival num is: {len(result_online_openalex)}")

        # 合并结果，优先级：offline < online < openalex
        merged_result = result_offline.copy()
        merged_result.update(result_online)
        merged_result.update(result_online_openalex)

        return merged_result, {
            "offline": len(result_offline),
            "online": len(result_online),
            "openalex": len(result_online_openalex)
        }

    except Exception as e:
        logger.error(f"Error in academic retrieval: {traceback.format_exc()}")
        return {}, {"offline": 0, "online": 0, "openalex": 0}


async def run_requests_parallel(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据query_domain参数选择检索方法:
    - 'academic': 使用学术检索 (run_academic_retrieval)
    - 'general': 使用通用网络检索 (retrival_general_web)
    - 'mixed' 或其他: 同时使用两种检索方法
    """
    try:
        t0 = time.time()
        query_domain = data.get("query_domain", "academic").lower()

        logger.info(f"Query domain: {query_domain}")

        # 根据query_domain选择检索策略
        if query_domain == "academic":
            # 只使用学术检索
            logger.info("Using academic retrieval only")
            merged_result, sources = await run_academic_retrieval(data)

        elif query_domain == "general":
            # 只使用通用网络检索
            logger.info("Using general web retrieval only")
            general_result = await retrival_general_web(data)
            merged_result = general_result
            sources = {"general_web": len(general_result), "offline": 0, "online": 0, "openalex": 0}

        else:
            # 混合模式：同时使用学术检索和通用网络检索
            logger.info("Using mixed retrieval (academic + general web)")

            # 并行执行学术检索和通用网络检索
            academic_task = run_academic_retrieval(data)
            general_task = retrival_general_web(data)

            academic_result, academic_sources = await academic_task
            general_result = await general_task

            # 合并结果，优先级：学术检索 > 通用网络检索
            merged_result = general_result.copy()  # 先添加通用检索结果
            merged_result.update(academic_result)  # 学术检索结果覆盖相同key的内容

            # 合并源统计
            sources = academic_sources.copy()
            sources["general_web"] = len(general_result)

        logger.info(f"Final retrieval result num: {len(merged_result)}")

        t1 = time.time()
        retrival_response = {
            "query": data["query"],
            "ctxs": list(merged_result.values()),
            "n_docs": len(merged_result),
            "time_cost": {"retrival_time": t1 - t0},
            "message": "search completed",
            "status": 200,
            "retrival_sources": sources,
            "query_domain": query_domain
        }
        return retrival_response

    except Exception as e:
        logger.error(f"Error in run_requests_parallel: {traceback.format_exc()}")
        return {
            "status": 500,
            "message": f"parallel search failed: {str(e)}",
            "query": data.get("query", ""),
            "ctxs": [],
            "n_docs": 0,
            "query_domain": data.get("query_domain", "mixed")
        }

# # 定义 FastAPI 路由
# @app.post("/process")
# async def process_requests(data: Dict[str, Any]):
#     try:
#         # 并行处理两个请求并合并结果
#         result = await run_requests_parallel(data)
#         return JSONResponse(content={"status": "success", "result": result})
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error processing requests: {str(e)}"
#         )

# # 启动服务器
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8192)
