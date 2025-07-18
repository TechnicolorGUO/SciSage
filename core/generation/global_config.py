import os
import json
import arxiv

GOOGLE_KEY = "xxx"  # find here: 

S2_API_KEY = ""  # semantic scholar api

DO_REFERENCE_SEARCH = True
DO_REFERENCE_SEARCH = False

proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}

arxiv_client = arxiv.Client(delay_seconds=0.05)


# 向量化模型
retrival_model_url = "http://172.24.212.149:9655/v1/emb/encoder"

# 召回服务
# recall_server_url = recall_server_url

recall_server_url = "http://0.0.0.0:5008/search"
# 排序模型
rerank_model_url = "http://0.0.0.0:9756/v1/score/rerank"

# 核心chat模型
chat_system = "You are a helpful AI assistant for scientific literature review. Please carefully follow user's instruction and help them to understand the most recent papers."

chat_url = "http://0.0.0.0:9089/v1/chat/completions"  # r1-70b

CHAT_MODEL_NAME = "r1-llama70b"
