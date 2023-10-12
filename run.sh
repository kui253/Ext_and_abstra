#!/usr/bin/env bash
CUDA_VISIBLE=2
logs_dir=./logs/
log_name=baseline.log
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} \
    nohup python -u run_summarization.py > "${logs_dir}${log_name}" 2>&1 &