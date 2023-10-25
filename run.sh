#!/usr/bin/env bash
# first just with_pred_only_train_with_unsorted
# second with concat_bert_preds_train_with_sorted
# thrid with_label_train_with_sorted
# forth with_label_sorted_trainn_with_sorted
CUDA_VISIBLE=3
logs_dir=./logs/
log_name=with_label_sorted_trainn_with_sorted.log
output_file=../results/bartlarge/forth
config_file=./config/graphbart_configV3.json
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} \
    nohup python run_summarization.py "${config_file}" "${output_file}" > "${logs_dir}${log_name}" 2>&1 &