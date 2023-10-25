import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# my own module

from torch.utils import data

import sys
from dataclasses import dataclass, field, asdict
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    BertModel,
    set_seed,
)
from transformers.optimization import AdamW
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from _transformers.seq2seq_trainer import BertTrainer
import torch.nn as nn
# my own module
from utils.args_util import ModelArguments, DataTrainingArguments, check_args
from _transformers.data_collator import MyDataCollatorForSeq2Seq, MyDataCollatorForBert
from _transformers.seq2seq_trainer import Seq2SeqTrainer
from _transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from utils import model_util, data_util, training_util
from utils.CONSTANT import *
import logging
from utils.metrics_util import get_bert_score, get_rouge_score, get_meteor_score
from bertExt.bert import BertExtra

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.7.0.dev0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main_bert():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    config_file = "./config/bertExt.json"

    config_file = sys.argv[1]
    output_file = sys.argv[2]
    model_args, data_args, training_args = parser.parse_json_file(
        config_file
    )  # 这里会重写之前hfArgumentParser中的值
    data_args, model_args, training_args = check_args(
        data_args, model_args, training_args
    )
    # 出现版本问题 最好使用transformers == 4.8.2
    # save config file
    training_args.output_dir = output_file
    output_dir = training_args.output_dir
    if not os.path.isdir(output_dir):
        os.system(f"mkdir {output_dir}")
    os.system(f"cp {config_file} {output_dir}/run_config.json")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    # ## add log to file
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.system(f"cp -r model {training_args.output_dir}")
    os.system(f"cp -r utils {training_args.output_dir}")

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # log args
    args_dict = {
        "data_args": data_args,
        "model_args": model_args,
        "training_args": training_args,
    }
    keys_ls = (
        list(asdict(training_args).keys())
        + list(asdict(model_args).keys())
        + list(asdict(data_args).keys())
    )
    max_width = max([len(arg_name) for arg_name in keys_ls])
    for k, v in args_dict.items():
        logger.info("*" * SCREEN_WIDTH)
        logger.info(k)
        for arg_name, arg_value in asdict(v).items():
            logger.info(f"{arg_name:{max_width}}  {arg_value}")
    # 上面是保存这些模型的参数的值
    # Set seed before initializing model.
    set_seed(training_args.seed)
    model = BertExtra(args=model_args)
    raw_dataset = data_util.get_dataset(data_args, model_args)
    def save_results(path_to,preds,labels):
        import json
        save_dict = []
        for pred,label in zip(preds,labels):
            pred =[int(i) for i in np.nonzero(pred)[0]]
            label = [int(i) for i in np.nonzero(label)[0]]
            temp = {"pred":pred,"label":label}
            save_dict.append(temp)
        # save_dict = [{"pred":list(np.nonzero(pred)[0].astype(int)),"label":list(np.nonzero(label)[0].astype(int))} for pred,label in zip(preds,labels)]
        json.dump(save_dict,open(path_to,'w'))
    def my_compute_metrics(eval_pred,metric_prefix):

        preds, labels, mask = eval_pred
        if metric_prefix == "last_eval":
            save_preds = preds * mask
            save_label = labels * mask
            save_results(os.path.join(training_args.output_dir,"last_eval.json"),save_preds,save_label)
        elif metric_prefix == "predict":
            save_preds = preds * mask
            save_label = labels * mask
            save_results(os.path.join(training_args.output_dir,"predict.json"),save_preds,save_label)
        all_valid = mask.sum()
        acc = ((preds == labels) * mask).sum() / all_valid
        return {"acc": acc}

    trainer = BertTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["validation"],
        data_collator=MyDataCollatorForBert,
        compute_metrics=my_compute_metrics,
    )
    # training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix='last_eval')

        trainer.log_metrics("validation", metrics)
        trainer.save_metrics("validation", metrics)
    if training_args.do_predict:
        test_result = trainer.predict(
            metric_key_prefix='predict',
            test_dataset=raw_dataset["test"],
        )
        metrics = test_result.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    file_ls = os.listdir(training_args.output_dir)
    for file in file_ls:
        if file.startswith("checkpoint"):
            os.system(f"rm -rf {os.path.join(training_args.output_dir,file)}")

    all_results_dir = os.path.join(training_args.output_dir, "all_results.json")
    best_rouge = json.load(open(all_results_dir))["acc"]

    log_dir = "/".join(training_args.output_dir.split("/")[:-1])

    os.system(f"mv {training_args.output_dir} {os.path.join(log_dir,str(best_rouge))}")



if __name__ == "__main__":
    main_bert()
