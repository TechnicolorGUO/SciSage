# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from local_model_langchain import LOCAL_MODELS

# Initialize AI models

llm_map = {
    "qwen-plus-latest": ChatOpenAI(
        model="qwen-plus-latest",
        base_url="YOUR_BASE_URL_HERE",  # 替换为你的base URL
        api_key="YOUR_API_KEY_HERE",    # 替换为你的API key
        temperature=0.7,
    ),
    "gpt-4": ChatOpenAI(
        model="gpt-4",
        base_url="YOUR_BASE_URL_HERE",  # 替换为你的base URL
        api_key="YOUR_API_KEY_HERE",    # 替换为你的API key
        temperature=0.7,
    ),
    "gpt-4o-mini": ChatOpenAI(
        model="gpt-4o-mini",
        base_url="YOUR_BASE_URL_HERE",  # 替换为你的base URL
        api_key="YOUR_API_KEY_HERE",    # 替换为你的API key
        temperature=0.7,
    ),
} | LOCAL_MODELS

print("ALL MODELS ", list(llm_map.keys()))

chat_models = {
    key: value for key, value in llm_map.items() if "r1" not in key and "qwq" not in key
}
print("ALL chat_models", list(chat_models.keys()))


local_chat_models = {
    key: value
    for key, value in LOCAL_MODELS.items()
    if "r1" not in key and "qwq" not in key and "openscholar" not in key
}

print("LOCAL chat_models", list(local_chat_models.keys()))
