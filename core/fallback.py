# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================


import json
import requests
from typing import Dict, Any, List, Optional, Tuple, Callable, Union


def create_reflection_fallback(
    params: Dict[str, Any], error_message: str
) -> Dict[str, Any]:
    """Create fallback response for section reflection failures"""
    return {
        "section_content": {
            "parent_section": params.get("parent_section", ""),
            "section_text": params.get("section_text", ""),
            "section_key_point": params.get("section_key_point", ""),
            "section_summary": f"[Summary could not be generated due to reflection error: {error_message}]",
            "section_index": params.get("section_index", 0),
            "search_query": params.get("search_query", ""),
            "reportIndexList": params.get("reportIndexList", []),
            "main_figure_data": params.get("main_figure_data", ""),
            "main_figure_caption": params.get("main_figure_caption", ""),
        },
        "search_queries": [],
        "reflection_result": {
            "meets_requirements": False,
            "feedback": f"Reflection failed with error: {error_message}",
            "improvement_queries": [],
        },
    }


def flow_information_sync(
    task_id: str,
    status: str = "processing",
    base_url: str = "",
    content: str = None,
    is_deal: bool = False,
    flowchart_data: dict = None,
    report_content: str = None,
) -> dict:
    """
    更新任务状态和内容的通用方法

    Args:
        task_id (str): 任务ID
        status (str): 任务状态
        base_url (str): API基础URL
        content (str, optional): 任务内容. Defaults to None.
        is_deal (bool, optional): 是否处理. Defaults to False.
        flowchart_data (dict, optional): 流程图数据. Defaults to None.
        report_content (str, optional): 报告内容. Defaults to None.

    Returns:
        dict: API响应数据

    Raises:
        Exception: 当API请求失败时抛出异常
    """
    if not base_url:
        base_url = "http://test.flagopen.baai.ac.cn"
    # 构建API端点
    url = f"{base_url}/chat/api/scisage/update_task"

    # 构建请求数据
    payload = {"task_id": task_id, "status": status, "is_deal": is_deal}

    # 添加可选参数
    if content:
        payload["content"] = content

    # print(f"flowchart_data: {flowchart_data}")
    if flowchart_data:
        payload["flowchart"] = json.dumps(flowchart_data)

    if report_content:
        payload["report"] = report_content

    # 设置请求头
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    print(f"payload: {payload}")
    try:
        # 发送请求
        response = requests.post(url, json=payload, headers=headers)
        # response.raise_for_status()  # 抛出非200响应的异常
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {str(e)}")
        pass
