#!/usr/bin/env bash
# first Bert_with_sents_limits_5
# second Bert_with_sents_limits_5_label_unchanged
# third Bert_only_with_sents_limits_5_label_unchaged
CUDA_VISIBLE=2
logs_dir=./logs/
log_name=Bert_only_with_sents_limits_5_label_unchaged.log
config_file=./config/bertExt.json
output_file=../results/bertbase/third
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} \
    nohup python run_bert.py "${config_file}" "${output_file}" > "${logs_dir}${log_name}" 2>&1 &