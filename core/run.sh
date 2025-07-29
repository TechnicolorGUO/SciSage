#!/bin/bash
# ---------------------------------------------------------------
# [Author]       : shixiaofeng
# [Descriptions] :

## for scholar search: get from https://serper.dev
export GOOGLE_SERPER_KEY="xxxx"
## for general search: get from https://serpapi.com/dashboard
export SERPAPI_API_KEY="xxx"

## for Local LLM inference url, change to your local LLM server address
export LOCAL_LLM_HOST="http://0.0.0.0"

export PYTHONPATH=$PWD

python3 main_workflow_opt_for_paper.py