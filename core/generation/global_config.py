import os
import json
import arxiv

# GOOGLE_KEY = "36ad974e0dc97f3470762db318cd076f119a1474"  # shixf1989@gmail.com
GOOGLE_KEY = "763472cad2f21aeb110715293b0c472efae149ae"  # jinxin
GOOGLE_KEY = "1bcfd523ebf332cb3dcab96ca826cff9019ca4f8"  # 13521092651

S2_API_KEY = ""  # semantic scholar api

DO_REFERENCE_SEARCH = True
DO_REFERENCE_SEARCH = False

proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}

arxiv_client = arxiv.Client(delay_seconds=0.05)


# 向量化模型
retrival_model_url = "http://172.24.212.149:9655/v1/emb/encoder"

# 召回服务
# recall_server_url = recall_server_url

recall_server_url = "http://172.24.214.62:5008/search"
# 排序模型
rerank_model_url = "http://120.92.91.62:9756/v1/score/rerank"

# 核心chat模型
chat_system = "You are a helpful AI assistant for scientific literature review. Please carefully follow user's instruction and help them to understand the most recent papers."

# chat_url = f"http://120.92.91.62:9757/v1/chat/completions" ## openscholar
chat_url = "http://120.92.91.62:9089/v1/chat/completions"  # r1-70b

# CHAT_MODEL_NAME = "r1-llama70b"
CHAT_MODEL_NAME = "openscholar"
