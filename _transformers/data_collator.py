import random
from re import X
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

InputDataClass = NewType("InputDataClass", Any)


@dataclass
class MyDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    dis_offset: int = 463
    adj_mat_ls: list = None

    def __call__(self, features):
        # features List[dict]
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                feature["labels"] = (
                    feature["labels"] + remainder
                    if padding_side == "right"
                    else remainder + feature["labels"]
                )
        # if self.max_utt_threshold > 0:
        #     for feature in feature:
        #         for n in ['gt_input_ids','gt_attention_mask','']

        if "gt_input_ids" in features[0].keys():
            normal_features_ls = [
                {
                    "attention_mask": x["attention_mask"],
                    "input_ids": x["input_ids"],
                    "labels": x["labels"],
                }
                for x in features
            ]
            normal_padded_featurs = self.tokenizer.pad(
                normal_features_ls,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            padded_features = {
                "gt_input_ids": [],
                "gt_attention_mask": [],
                "adj_mats": [],
            }
            (
                padded_features["num_utt_ls"],
                padded_features["gt_input_ids"],
                padded_features["gt_attention_mask"],
            ) = pad_list_of_tensor(features, self.tokenizer)
            max_num_utt = max([len(feature["gt_input_ids"]) for feature in features])

            adj_mats_ls = self.adj_mat_ls
            if any([m in features[0].keys() for m in adj_mats_ls]):
                padded_features["adj_mats"] = {}
            for adj_type in adj_mats_ls:
                if adj_type in features[0].keys():
                    padded_features["adj_mats"][adj_type] = []

            for feature in features:
                #  sudo_features = []
                #  for idx in range(len(feature['gt_attention_mask'])):
                #      sudo_dict = {'attention_mask':feature['gt_attention_mask'][idx],'input_ids':feature['gt_input_ids'][idx]}
                #      sudo_features.append(sudo_dict)

                #  padded_sudo_features = self.tokenizer.pad(
                #      sudo_features, #List[Dict[str,List[int]]]
                #      padding=self.padding,
                #      max_length=self.max_length,
                #      pad_to_multiple_of=self.pad_to_multiple_of,
                #      return_tensors="pt",
                #  )
                #  # dict[str:tensor(batch_first)]
                #  padded_features['gt_input_ids'].append(padded_sudo_features['input_ids'])
                #  padded_features['gt_attention_mask'].append(padded_sudo_features['attention_mask'])
                for adj_type in adj_mats_ls:
                    if adj_type in feature.keys():
                        mat = feature[adj_type]
                        ori_mat_size = len(mat)
                        mat = torch.tensor(mat)
                        if not ori_mat_size <= max_num_utt:
                            print(adj_type)
                        padded_mat = torch.zeros(
                            (max_num_utt, max_num_utt), dtype=mat.dtype
                        )
                        padded_mat[:ori_mat_size, :ori_mat_size] = mat
                        if adj_type == "distance_adj":
                            padded_mat += self.dis_offset
                    padded_features["adj_mats"][adj_type].append(padded_mat)

            if "adj_mats" in padded_features.keys():
                for k, v in padded_features["adj_mats"].items():
                    padded_features["adj_mats"][k] = torch.stack(v)

            padded_features["input_ids"] = normal_padded_featurs["input_ids"]
            padded_features["attention_mask"] = normal_padded_featurs["attention_mask"]
            padded_features["labels"] = normal_padded_featurs["labels"]
        elif feature.get("sorted_attention_mask", None) is not None:
            normal_features_ls = [
                {
                    "attention_mask": x["attention_mask"],
                    "input_ids": x["input_ids"],
                    "labels": x["labels"],
                }
                for x in features
            ]
            normal_padded_featurs = self.tokenizer.pad(
                normal_features_ls,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            temp_padded_features = [
                {
                    "input_ids": f["sorted_input_ids"],
                    "attention_mask": f["sorted_attention_mask"],
                }
                for f in features
            ]
            temp_padded_features = self.tokenizer.pad(
                temp_padded_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            padded_features = {
                "input_ids": normal_padded_featurs["input_ids"],
                "attention_mask": normal_padded_featurs["attention_mask"],
                "labels": normal_padded_featurs["labels"],
                "sorted_input_ids": temp_padded_features["input_ids"],
                "sorted_attention_mask": temp_padded_features["attention_mask"],
            }
            padded_features = pad_other_features(
                padded_features, self.tokenizer.pad_token_id
            )
        else:
            padded_features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=padded_features["labels"]
            )
            padded_features["decoder_input_ids"] = decoder_input_ids

        return padded_features


def pad_to_same_dim(feature_list, pad_token_id, is_input_ids=False):
    max_len = min(max([len(x) for x in feature_list]), 512)
    if is_input_ids:
        all_attentions = []
    for idx in range(len(feature_list)):
        if len(feature_list[idx]) < max_len:
            feature_list[idx] += [pad_token_id] * (max_len - len(feature_list[idx]))
        else:
            feature_list[idx] = feature_list[idx][:max_len]
        if is_input_ids:
            if len(feature_list[idx]) < max_len:
                attention_mask = [1] * len(feature_list[idx]) + [0] * (
                    max_len - len(feature_list[idx])
                )
            else:
                attention_mask = [1] * max_len
            all_attentions.append(attention_mask)

    if not is_input_ids:
        return torch.tensor(feature_list)
    else:
        return torch.tensor(feature_list), torch.tensor(
            all_attentions, dtype=torch.bool
        )


def MyDataCollatorForBert(features):
    paded_features = {}
    paded_features["input_ids"], paded_features["attention_mask"] = pad_to_same_dim(
        [f["input_ids"] for f in features], 0, True
    )
    paded_features["token_type_ids"] = pad_to_same_dim(
        [f["tokenType_id"] for f in features], 0
    )
    for f in features:
        f["clss_pos"] = [i for i in f["clss_pos"] if i < 512]
        f["labels"] = f["labels"][: len(f["clss_pos"])]
        assert len(f["labels"]) == len(f["clss_pos"]),'wrong input dim'
    paded_features["labels"] = pad_to_same_dim([f["labels"] for f in features], 0)
    paded_features["clss_pos"] = pad_to_same_dim([f["clss_pos"] for f in features], -1)
    clss_mask = torch.ones(paded_features["clss_pos"].shape, dtype=torch.bool)
    clss_mask[paded_features["clss_pos"] == -1] = 0
    paded_features["clss_mask"] = clss_mask
    
    paded_features["original_sent"] = [f["original_sent"] for f in features]
    return paded_features


def pad_other_features(features, pad_token_id):
    res_len = features["input_ids"].shape[1] - features["sorted_input_ids"].shape[1]
    if res_len > 0:
        features["sorted_input_ids"] = torch.cat(
            [
                features["sorted_input_ids"],
                torch.ones(
                    (features["sorted_input_ids"].shape[0], res_len), dtype=torch.long
                )
                * pad_token_id,
            ],
            dim=1,
        )
        features["sorted_attention_mask"] = torch.cat(
            [
                features["sorted_attention_mask"],
                torch.zeros(
                    (features["sorted_attention_mask"].shape[0], res_len),
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
    else:
        res_len = -res_len
        features["input_ids"] = torch.cat(
            [
                features["input_ids"],
                torch.ones((features["input_ids"].shape[0], res_len), dtype=torch.long)
                * pad_token_id,
            ],
            dim=1,
        )
        features["attention_mask"] = torch.cat(
            [
                features["attention_mask"],
                torch.zeros(
                    (features["attention_mask"].shape[0], res_len),
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
    return features


def pad_list_of_tensor(features, tokenizer):
    """
    features: List[Dict[str,List[str]]
    """
    max_seq_len_in_a_batch = 0
    len_ls = []
    for sample in features:
        len_ls.append(len(sample["gt_input_ids"]))
        for utt in sample["gt_input_ids"]:
            l = len(utt)
            if l > max_seq_len_in_a_batch:
                max_seq_len_in_a_batch = l

    for sample in features:
        for k, v in sample.items():
            if k == "gt_input_ids":
                for utt in v:
                    diff = max_seq_len_in_a_batch - len(utt)
                    utt += [tokenizer.pad_token_id] * diff  # utt = utt + is wrong
            elif k == "gt_attention_mask":
                for mask in v:
                    diff = max_seq_len_in_a_batch - len(mask)
                    mask += [0] * diff
    return (
        len_ls,
        torch.cat([torch.tensor(x["gt_input_ids"]) for x in features], dim=0),
        torch.cat([torch.tensor(x["gt_attention_mask"]) for x in features], dim=0),
    )
