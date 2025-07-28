import os
import json
import arxiv

GOOGLE_KEY = "xxx"  # find here:

S2_API_KEY = ""  # semantic scholar api

DO_REFERENCE_SEARCH = True
DO_REFERENCE_SEARCH = False

proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}


# bge embedding model
retrival_model_url = "http://0.0.0.0:9655/v1/emb/encoder"

# 召回服务
# recall_server_url = recall_server_url

recall_server_url = "http://0.0.0.0:5008/search"
# 排序模型
rerank_model_url = "http://0.0.0.0:9756/v1/score/rerank"

# 核心chat模型

chat_url = "http://0.0.0.0:9089/v1/chat/completions"  # r1-70b

CHAT_MODEL_NAME = "r1-llama70b"
