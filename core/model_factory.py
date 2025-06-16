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
    "gpt-4": AzureChatOpenAI(
        openai_api_type="azure",
        openai_api_version="your api version",
        azure_deployment="gpt-4",
        azure_endpoint="you endpoint",
        api_key="your-api-key-here",
    ),
    "gpt-4o-mini": AzureChatOpenAI(
        openai_api_type="azure",
        openai_api_version="your api version",
        azure_deployment="gpt-4o-mini",
        azure_endpoint="you endpoint",
        api_key="your-api-key-here",
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
