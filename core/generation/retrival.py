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
from generation.api_web import (
    get_doc_info_from_api,
    search_paper_via_query_from_openalex,
)
import requests
from generation.global_config import recall_server_url
import traceback
from generation.generation_instructions import (
    template_extract_keywords_source_aware,
)
from generation.local_request_v2 import get_from_llm
import re

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


async def run_requests_parallel(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Priority is given to calling retrival_online.
    If all 5 attempts fail, then call retrival_online_openalex, with a maximum retry count of 3.
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
                logger.info(f"Attempting retrival_online, attempt {attempt + 1}/5")
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
            if attempt < 4:
                await asyncio.sleep(1)  # 等待1秒再重试

        # 如果retrival_online失败，尝试retrival_online_openalex，最多3次
        if not online_success or len(result_online)<15:
            logger.info("retrival_online failed after 5 attempts, trying retrival_online_openalex")

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
                    await asyncio.sleep(1)  # 等待1秒再重试

        logger.info(f"offline retrival num is: {len(result_offline)}")
        logger.info(f"online retrival num is: {len(result_online)}")
        logger.info(f"online openalex retrival num is: {len(result_online_openalex)}")

        # 合并结果，优先级：offline < online < openalex
        merged_result = result_offline.copy()
        merged_result.update(result_online)
        merged_result.update(result_online_openalex)

        logger.info(f"retrival final result num is: {len(merged_result)}")

        t1 = time.time()
        retrival_response = {
            "query": data["query"],
            "ctxs": list(merged_result.values()),
            "n_docs": len(merged_result),
            "time_cost": {"retrival_time": t1 - t0},
            "message": "search completed",
            "status": 200,
            "retrival_sources": {
                "offline": len(result_offline),
                "online": len(result_online),
                "openalex": len(result_online_openalex)
            }
        }
        return retrival_response

    except Exception as e:
        logger.error(f"Error in run_requests_parallel: {traceback.format_exc()}")
        return {
            "status": 500,
            "message": f"parallel search failed: {str(e)}",
            "query": data.get("query", ""),
            "ctxs": [],
            "n_docs": 0
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
