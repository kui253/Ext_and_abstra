from openie import StanfordOpenIE
import json
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer


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


# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
def filter_triples(triples: list, names: list, utterances, name_seq, utts_coefed):
    scores = {}
    select_actions = {}
    all_tri_names = [triple["subject"] for triple in triples]
    for name_set in names:
        if scores.get(name_set) is None:
            scores[name_set] = []
        if name_set in all_tri_names:
            idxs_in_utt = [i for i, name in enumerate(name_seq) if name == name_set]
            idxs_in_tri = [
                i for i, name in enumerate(all_tri_names) if name_set == name
            ]
            for name_tri_id in idxs_in_tri:
                select_utt = [utterances[i] for i in idxs_in_utt]
                triple_sent = (
                    triples[name_tri_id]["relation"]
                    + " "
                    + triples[name_tri_id]["object"]
                    + "."
                )
                triple_score = get_rouge_score(
                    [triple_sent] * len(select_utt), select_utt
                )
                f1 = [
                    triple_score["rouge-1"][i]["f"][0]
                    for i in range(len(triple_score["rouge-1"]))
                ]
                utt_idx = np.argmax(np.array(f1))
                max_score = max(f1)
                if max_score < 0.2:
                    continue
                sen_id = idxs_in_utt[utt_idx]

                scores[name_set].append(
                    {
                        "scores": max_score,
                        "sent_id": sen_id,
                        "tri": triples[name_tri_id],
                    }
                )
        else:
            for tri in triples:
                triple_sent = (
                    tri["subject"] + " " + tri["relation"] + " " + tri["object"] + "."
                )
                triple_score = get_rouge_score(
                    [triple_sent] * len(utts_coefed), utts_coefed
                )
                f1 = [
                    triple_score["rouge-1"][i]["f"][0]
                    for i in range(len(triple_score["rouge-1"]))
                ]
                utt_idx = int(np.argmax(np.array(f1)))
                max_score = max(f1)
                scores[name_set].append(
                    {"scores": max_score, "sent_id": utt_idx, "tri": tri}
                )
    for k, v in scores.items():
        if len(v) == 0:
            continue
        elif len(v) == 1:
            select_actions[k] = v
        else:
            select_actions[k] = sorted(v, key=lambda x: x["scores"], reverse=True)[:2]
    return select_actions


def extra_sent(mode, save_path):
    properties = {
        "openie.affinity_probability_cap": 2 / 3,
    }

    with StanfordOpenIE(properties=properties) as client:
        path = (
            "/data1/whd/diaResearch/SDDS/data/samsum/samsum_{}_sum_center.json".format(
                mode
            )
        )
        path2 = "/data1/whd/diaResearch/SDDS/data/samsum/samsum_{}_names.json".format(
            mode
        )
        with open(path, "r") as f:
            data = json.load(f)
        with open(path2, "r") as f:
            names = json.load(f)
        pbar = tqdm(data, desc="get triples")
        for idx, example in enumerate(pbar):
            if idx == 10:
                save_path += "samsum_{}_sum_center2.json".format(mode)
                json.dump(data[:10], open(save_path, "w"))
            triples = []
            name = names[idx]
            utts = example["pred_utt"].split("#")
            for utt in utts:
                triples += client.annotate(utt)
            select_actions = filter_triples(
                triples, name["name"], example["utterances"], example["names"], utts
            )
            example["triples"] = select_actions
    save_path += "samsum_{}_sum_center2.json".format(mode)
    json.dump(data, open(save_path, "w"))


def fix_int64(mode):
    path = "/data1/whd/diaResearch/SDDS/data/samsum/action_extra_ver/samsum_{}_sum_center2.json".format(
        mode
    )
    with open(path, "r") as f:
        data = json.load(f)
    pbar = tqdm(data, desc="fixing")
    all_data = []
    for example in data:
        sample = {}
        sample_sent_ids = []
        action_sent = []
        for name, v in example["triples"].items():
            for tri in v:
                sample_sent_ids.append(tri["sent_id"])
                action_sent.append(
                    " ".join(
                        [
                            tri["tri"]["subject"],
                            tri["tri"]["relation"],
                            tri["tri"]["object"],
                        ]
                    )
                )
        sample["action_sent"] = list(set(action_sent))
        sample["sent_id"] = list(set(sample_sent_ids))
        all_data.append(sample)
    save_path = "/data1/whd/diaResearch/SDDS/data/samsum/action_extra_ver/samsum_{}_sum_center3.json".format(
        mode
    )
    json.dump(all_data, open(save_path, "w"))


def merge_json(path_to, path_from, path_save):
    with open(path_from, "r") as f:
        data_from = json.load(f)
    with open(path_to, "r") as f:
        data_to = json.load(f)
    pbar = tqdm(data_from, desc="merging")
    for idx, example in enumerate(pbar):
        data_to[idx]["action_sent"] = (
            example["action_sent"] if len(example["action_sent"]) > 0 else ["none"]
        )
        data_to[idx]["sent_id"] = (
            example["sent_id"] if len(example["sent_id"]) > 0 else [-1]
        )
    json.dump(data_to, open(path_save, "w"))


def prepare_bertExt(path_from, path_to):
    with open(path_from, "r") as f:
        data_from = json.load(f)
    # with open(path_to, "r") as f:
    #     data_to = json.load(f)
    data_to = []
    pbar = tqdm(data_from, desc="preparing")
    for idx, example in enumerate(pbar):
        sents = example["pred_utt"].split("#")
        bert_tokenizer = BertTokenizer.from_pretrained(
            "/data2/trace/common_param/bert-base-uncased"
        )
        tokenized_word = bert_tokenizer.batch_encode_plus(sents)
        tokenType_id = []
        input_ids = []
        clss_pos = []
        labels = example["utts_idx_inorder_sorted"]
    
        # if len(labels) == len(sents):
        #     if len(filtered_sent) > 0:
        #         labels = filtered_sent
        one_hot_labels = [0] * len(sents)
        for i in labels:
            if i < len(sents):
                one_hot_labels[i] = 1
        for i, tokens in enumerate(tokenized_word["input_ids"]):
            if i == 0:
                clss_pos.append(0)
            else:
                clss_pos.append(clss_pos[-1] + token_len)
            if i % 2 == 0:
                tokenType_id += [0] * len(tokens)
            else:
                tokenType_id += [1] * len(tokens)
            input_ids += tokens
            token_len = len(tokens)
        assert len(sents) == len(clss_pos), "wrong input ids dim"
        data_to.append(
            {
                "tokenType_id": tokenType_id,
                "input_ids": input_ids,
                "clss_pos": clss_pos,
                "labels": one_hot_labels,
                'original_sent':sents
            }
        )
    json.dump(data_to, open(path_to, "w"))


def add_original_sent_to(data_path, data_to):
    with open(data_path, "r") as f:
        data = json.load(f)
    with open(data_to, "r") as f:
        dataset_to = json.load(f)
    pbar = tqdm(data, desc="adding")
    for idx, example in enumerate(pbar):
        sents = example["unit_utts"].split("#")
        dataset_to[idx]["original_sent"] = sents
    json.dump(dataset_to, open(data_to, "w"))

def establish_final_dataset(path_original,Ext_data,path_save,mode):
    with open(path_original,'r') as f:
        data = json.load(f)
    with open(Ext_data,'r') as f:
        Exts = json.load(f)
    pbar = tqdm(data,desc='processing')
    for idx, example in enumerate(pbar):

        example.pop('document_clusters')
        example.pop('clusters')
        example.pop('unit_utts_version2')
        if mode!='train':
            example.pop('utts_idx_inorder')
            example['selected_sents_bert'] = [int(i) for i in Exts[idx]['pred']]
            example['label_sents'] = [int(i) for i in Exts[idx]['label']]
            if len(example['sent_id']) == 1 and example['sent_id'][0] == -1:
                example['concat_action_select'] = [int(i) for i in Exts[idx]['pred']]
            else:
                example['concat_action_select'] = sorted([int(i) for i in set(Exts[idx]['pred'] + example['sent_id'])])
        else:
            example['utts_idx_inorder_sorted'] = [int(i) for i in Exts[idx]['utts_idx_inorder_sorted']]
    json.dump(data,open(path_save,'w'))

def fix_intV(path,mode):
    with open(path,'r') as f:
        data = json.load(f)
    pbar = tqdm(data,desc='processing')
    for idx, example in enumerate(pbar):
        if mode!='train':
           
            example['utts_idx_inorder_sorted'] = [-1]
            example['utts_idx_inorder'] = [-1]

        else:

            example['selected_sents_bert'] = [-1]
            example['label_sents'] = [-1]
            
            example['concat_action_select'] = [-1]
    json.dump(data,open(path,'w'))
if __name__ == "__main__":
    # prepare_bertExt(
    #     "/data1/whd/diaResearch/SDDS/data/samsum/samsum_train_sum_center_sorted.json",
    #     '/data1/whd/diaResearch/SDDS/data/samsum/bert_ext_train.json',
    # )
    prepare_bertExt(
        "/data1/whd/diaResearch/SDDS/data/samsum/samsum_validation_sum_center_sorted.json",
        '/data1/whd/diaResearch/SDDS/data/samsum/bert_ext_validation.json',
    )
    prepare_bertExt(
        "/data1/whd/diaResearch/SDDS/data/samsum/samsum_test_sum_center_sorted.json",
        '/data1/whd/diaResearch/SDDS/data/samsum/bert_ext_test.json',
    )
