# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import json
import time
import requests

base_url = "http://localhost:8080"

# 准备请求数据
request_data = {
    "user_name": "researcher",
    "user_query": "The impact of artificial intelligence on modern healthcare",
    "section_writer_model":"Qwen3-8B",
    "outline_max_sections": 4,
    "do_query_understand": True,
    "do_section_reflection": True,
    "do_global_reflection": True,
    "translate_to_chinese": True

}
response = requests.post(f"{base_url}/generate_sync", json=request_data)
if response.status_code == 200:
    result = response.json()
    print("Paper generation completed successfully.")
    if "final_paper_content" in result:
        print(f"Final paper content: {result['final_paper_content'][:200]}...")  # 只显示前200字符
    print(f"Full response: {result}")
else:
    print(f"Failed to generate paper: {response.status_code}")
    print(f"Error: {response.text}")

response = response.json()
print(response.keys())