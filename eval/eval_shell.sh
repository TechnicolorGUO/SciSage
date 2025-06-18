#!/bin/bash
# Evaluate all JSONL files in a folder to obtain scores.

# Modify this to your Anaconda environment path
export LD_LIBRARY_PATH=${HOME}/anaconda3/envs/YOUR_ENV_NAME/lib/YOUR_PYTHON_VERSION/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Input folder path
json_folder=$1

# Check whether a folder path was provided
if [ -z "$json_folder" ]; then
    echo "Please provide the folder path containing JSON files as a parameter"
    exit 1
fi

# Create a log directory
mkdir -p ./output_mydata/eval/log

# Iterate through all .json and .jsonl files
for json_file in "$json_folder"/*.json "$json_folder"/*.jsonl; do
    if [ -f "$json_file" ]; then
        current_datetime=$(date +%Y%m%d_%H%M%S)
        base_name=$(basename "$json_file")
        name_no_ext="${base_name%.*}"
        saving_path="./output_mydata/eval/${name_no_ext}_eval"
        log_file="./output_mydata/eval/log/${current_datetime}_${name_no_ext}_eval_survey.log"

        echo "Starting evaluation: $json_file"
        echo "Log file: $log_file"

        python ./evaluation/all_eval.py \
            --jsonl_file "$json_file" \
            --saving_path "$saving_path" \
            --eval_model gemini-2.0-flash-thinking-exp-1219 \
            --infer_type OpenAI \
            --method_name LLMxMRv2 \
            2>&1 | tee "$log_file"

        sleep 1
    fi
done
