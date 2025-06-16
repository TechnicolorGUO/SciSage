# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import jsonlines
import re
import csv
import os
import json
import hashlib
import requests

def load_jsonlines(file):
    try:
        with jsonlines.open(file, 'r') as jsonl_f:
            lst = [obj for obj in jsonl_f]
    except:
        lst = []
        with open(file) as f:
            for line in f:
                lst.append(json.loads(line))
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def extract_titles(text):
    # Regular expression pattern to match the titles
    pattern = r'\[\d+\]\s(.+)'  # Matches [number] followed by any characters
    # Find all matches of the pattern in the reference text
    if "References:" not in text:
        return []
    else:
        reference_text = text.split("References:")[1]
        print(reference_text)
        matches = re.findall(pattern, reference_text)
        return matches

def save_tsv_dict(data, fp, fields):
    # build dir
    dir_path = os.path.dirname(fp)
    os.makedirs(dir_path, exist_ok=True)

    # writing to csv file
    with open(fp, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t',)
        writer.writeheader()
        writer.writerows(data)



def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = "".join(letters)
    return result.lower()



def get_md5(string):
    # 创建 MD5 对象
    md5_hash = hashlib.md5()
    # 将字符串编码为字节（MD5 需要字节输入）
    md5_hash.update(string.encode("utf-8"))
    # 获取 16 进制表示的哈希值
    return md5_hash.hexdigest()



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
    # print(f"payload: {payload}")
    # try:
    #     # 发送请求
    #     response = requests.post(url, json=payload, headers=headers)
    #     # response.raise_for_status()  # 抛出非200响应的异常
    #     return response.json()

    # except requests.exceptions.RequestException as e:
    #     print(f"API请求失败: {str(e)}")
    #     pass
