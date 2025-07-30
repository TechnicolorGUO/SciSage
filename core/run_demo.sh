#!/bin/bash
# ---------------------------------------------------------------
# [Author]       : shixiaofeng
# [Descriptions] :
## for Local LLM inference url, change to your local LLM server address

export LOCAL_LLM_HOST="http://0.0.0.0"

export PYTHONPATH=$PWD

python3 main_workflow_opt_for_paper.py