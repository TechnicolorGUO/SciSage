import os
import json
import arxiv


proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}


# bge embedding model
retrival_model_url = "http://0.0.0.0:9655/v1/emb/encoder"

# 召回服务
recall_server_url = "http://0.0.0.0:5008/search"
# 排序模型
rerank_model_url = "http://0.0.0.0:9756/v1/score/rerank"

# 核心chat模型

# chat_url = f"http://120.92.91.62:9757/v1/chat/completions" ## openscholar
chat_url = "http://0.0.0.0:9089/v1/chat/completions"  # r1-70b

CHAT_MODEL_NAME = "r1-llama70b"
