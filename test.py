import re
import nltk
from tqdm import tqdm
import torch

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.


def get_rouge_score(hyps, refs):
    import rouge

    evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=100,
        length_limit_type="words",
        apply_avg=False,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )
    py_rouge_scores = evaluator.get_scores(hyps, refs)
    return py_rouge_scores


# def concat_same_speaker_inseq(example):
#     names = example['names']
#     utts = example['unit_utts'].split("#")
#     for i in range(len(names)-1):
#         if names[i] == names[i+1]:
#             utts +=


def get_sum_center_data(path, mode="train", save_path=None, use_unit=False):
    import json

    with open(path, "r") as f:
        data = json.load(f)
    pbar = tqdm(data, desc="get_sum_center_data")
    first = True
    for idx, example in enumerate(pbar):
        if idx == 1000 and first:
            first = False
            temp_path = save_path + "samsum_{}_sum_center{}.json".format(mode, idx)
            json.dump(data, open(temp_path, "w"))
        summary_pieces = nltk.sent_tokenize(example["summary"])
        if use_unit:
            coefed_utts = example["unit_utts"].split("#")
        else:
            coefed_utts = example["pred_utt"].split("#")
        if len(coefed_utts) >= len(summary_pieces):
            divnum = len(coefed_utts) // len(summary_pieces)

        else:
            raise ValueError("summary_pieces > coefed_utts")

        if divnum > 3:
            divnum = 3
        idxs = []
        for idx, piece in enumerate(summary_pieces):
            if idx > len(coefed_utts):
                break
            repeat_piece = [piece] * len(coefed_utts)
            scores = get_rouge_score(repeat_piece, coefed_utts)
            f1 = [scores["rouge-1"][i]["f"][0] for i in range(len(scores["rouge-1"]))]
            tensor_list = torch.tensor(f1)
            result = torch.topk(tensor_list, divnum).indices
            idxs.extend(result.tolist())
        if len(summary_pieces) != 1:
            scores2 = get_rouge_score(
                [example["summary"]] * len(coefed_utts), coefed_utts
            )
            f2 = [scores2["rouge-1"][i]["f"][0] for i in range(len(scores2["rouge-1"]))]
            tensor_list2 = torch.tensor(f2)
            result2 = torch.topk(tensor_list2, divnum).indices
            idxs.extend(result2.tolist())

        example["utts_idx_inorder"] = idxs
    save_path += "samsum_{}_sum_center.json".format(mode)
    json.dump(data, open(save_path, "w"))


if __name__ == "__main__":
    get_sum_center_data(
        "/data1/whd/diaResearch/SDDS/data/samsum/samsum_train2.json",
        "train",
        "/data1/whd/diaResearch/SDDS/data/samsum/",
    )
    get_sum_center_data(
        "/data1/whd/diaResearch/SDDS/data/samsum/samsum_validation2.json",
        "validation",
        "/data1/whd/diaResearch/SDDS/data/samsum/",
    )
    get_sum_center_data(
        "/data1/whd/diaResearch/SDDS/data/samsum/samsum_test2.json",
        "test",
        "/data1/whd/diaResearch/SDDS/data/samsum/",
    )
