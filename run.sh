#!/usr/bin/env bash
CUDA_VISIBLE=2
logs_dir=./logs/
log_name=data_center_impt_modified_batched.log
config_file=./config/graphbart_configV3.json
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} \
    nohup python run_summarization.py "${config_file}" > "${logs_dir}${log_name}" 2>&1 &