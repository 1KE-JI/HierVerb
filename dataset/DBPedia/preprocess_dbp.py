# -*- coding:utf-8 -*-
"""
Python 3.6
author：jike
date：2022年09月29日
"""
import json
import os
import csv
import re
import torch
from tqdm import tqdm
from openprompt.data_utils.utils import InputExample
import random
import json
from openprompt.data_utils.data_sampler import FewShotSampler

base_path = "dataset"

data_path = os.path.join(base_path, "DBPedia")

formatted_path = os.path.join(data_path, "formatted_data")


def text_cleaner(text):

    text = text.replace("\\", "")
    text = text.replace("\'", "")
    text = text.replace(";", "")
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()

    return text.lower()


def get_dbp_dataset(is_cover=False, clean_data=True, choice=-1):
    print("----------------- dbp get_dbp_dataset -----------------")
    all_data_path = os.path.join(formatted_path, f"all_data.pt")
    if os.path.exists(os.path.join(formatted_path, "l1_label.txt")) and os.path.exists(all_data_path) and not is_cover:
        print("loading all_data from", all_data_path)
        all_data = torch.load(all_data_path)

        with open(os.path.join(formatted_path, "l1_label.txt"), 'r') as fp:
            l1 = [i.strip() for i in list(fp)]
        with open(os.path.join(formatted_path, "l2_label.txt"), 'r') as fp:
            l2 = [i.strip() for i in list(fp)]
        with open(os.path.join(formatted_path, "l3_label.txt"), 'r') as fp:
            l3 = [i.strip() for i in list(fp)]

        all_labels = torch.load(os.path.join(formatted_path, "all_labels.pt"))
        return all_data, l1, l2, l3, all_labels

    l1_dict = dict()
    l2_dict = dict()
    l3_dict = dict()

    all_data = []
    with open(os.path.join(data_path, "wiki_data.csv"), 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=",")

        for i, row in tqdm(enumerate(reader)):
            if i == 0:
                continue
            idx, sentence, l1, l2, l3, wiki_name, text_len = row

            all_data.append([sentence, l1, l2, l3])

            l1_dict[l1] = l1_dict.get(l1, 0) + 1
            l2_dict[l2] = l2_dict.get(l2, 0) + 1
            l3_dict[l3] = l3_dict.get(l3, 0) + 1

    if clean_data:
        print("----------------- dbp text_cleaner -----------------")
        all_data = [[text_cleaner(example[0]), example[1], example[2], example[3]] for example in all_data]
        print("----------------- end dbp text_cleaner -----------------")
    l1_labels = list(l1_dict.keys())
    l2_labels = list(l2_dict.keys())
    l3_labels = list(l3_dict.keys())
    all_labels = list(l1_dict.keys()) + list(l2_dict.keys()) + list(l3_dict.keys())
    if not os.path.exists(os.path.join(formatted_path, "l1_label.txt")) or is_cover:
        with open(os.path.join(formatted_path, "l1_label.txt"), 'w') as fp:
            fp.write("\n".join(l1_labels))

        with open(os.path.join(formatted_path, "l2_label.txt"), 'w') as fp:
            fp.write("\n".join(l2_labels))

        with open(os.path.join(formatted_path, "l3_label.txt"), 'w') as fp:
            fp.write("\n".join(l3_labels))

        torch.save(all_labels, os.path.join(formatted_path, "all_labels.pt"))
    if not os.path.exists(all_data_path) or is_cover:
        print("saving all_data to", all_data_path)
        torch.save(all_data, all_data_path)
    print("----------------- end dbp get_dbp_dataset -----------------")
    return all_data, l1_labels, l2_labels, l3_labels, all_labels


def get_mapping(is_cover=False, clean_data=True, choice=-1):
    print("----------------- dbp get_mapping -----------------")
    if not os.path.exists(formatted_path):
        os.mkdir(formatted_path)
    all_data, l1_labels, l2_labels, l3_labels, all_labels = get_dbp_dataset(is_cover, clean_data, choice=choice)

    with open(os.path.join(data_path, "dbp_train_test_split.json")) as fp:
        index = json.load(fp)

    l1_length = len(l1_labels)
    l2_length = len(l2_labels)
    l3_length = len(l3_labels)

    l1_label2id = dict({i: idx for idx, i in enumerate(l1_labels)})
    l2_label2id = dict({i: idx for idx, i in enumerate(l2_labels)})
    l3_label2id = dict({i: idx for idx, i in enumerate(l3_labels)})

    l1_2_l2 = dict()
    l2_2_l3 = dict()

    for idx in range(l1_length):
        l1_2_l2[idx] = set()
    for idx in range(l2_length):
        l2_2_l3[idx] = set()

    for example in all_data:
        sentence, l1, l2, l3 = example
        l1_id = l1_label2id[l1]
        l2_id = l2_label2id[l2]
        l3_id = l3_label2id[l3]

        l1_2_l2[l1_id].add(l2_id)
        l2_2_l3[l2_id].add(l3_id)

    for key in l1_2_l2:
        l1_2_l2[key] = list(l1_2_l2[key])
    for key in l2_2_l3:
        l2_2_l3[key] = list(l2_2_l3[key])

    l2_2_l1 = dict()
    l3_2_l2 = dict()
    for key, value in l1_2_l2.items():
        for v in value:
            l2_2_l1[v] = key

    for key, value in l2_2_l3.items():
        for v in value:
            l3_2_l2[v] = key

    train_index = index['train']
    test_index = index['test']

    all_data = [[i[0], i[1], l2_label2id[i[2]], l3_label2id[i[3]]] for i in all_data]

    train_data = [all_data[i] for i in train_index]
    test_data = [all_data[i] for i in test_index]
    print("----------------- end dbp get_mapping -----------------")
    return train_data, test_data, l1_labels, l2_labels, l3_labels, all_labels, l1_2_l2, l2_2_l1, l2_2_l3, l3_2_l2


def few_shot_sample(data, shot=5, seed=171, choice=-1):

    sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
    examples = []

    for idx, i in enumerate(data):
        examples.append(InputExample(guid=str(idx), text_a=str(idx), label=i[choice]))
    fewshot_examples, dev_examples = sampler(examples, seed=seed)
    index = []
    dev_index = []
    for example in fewshot_examples:
        index.append(int(example.guid))

    for example in dev_examples:
        dev_index.append(int(example.guid))

    return index, dev_index


def sub_dataset(data, shot=5, seed=171, choice=-1):
    print(f"------------ using seed {seed} ------------")
    print(f"------------ loading few-shot for {shot} shot ------------")

    fewshot_path = os.path.join(base_path, "few-shot", "DBPedia")
    if not os.path.exists(fewshot_path):
        os.mkdir(fewshot_path)
    if choice == -1:
        cur_path = os.path.join(fewshot_path, f"train-seed_{seed}-shot_{shot}.json")
        cur_dev_path = os.path.join(fewshot_path, f"dev-seed_{seed}-shot_{shot}.json")
    else:
        cur_path = os.path.join(fewshot_path, f"train-seed_{seed}-shot_{shot}-choice{choice}.json")
        cur_dev_path = os.path.join(fewshot_path, f"dev-seed_{seed}-shot_{shot}-choice{choice}.json")

    if os.path.exists(cur_path):
        index = json.load(open(cur_path, 'r'))
        dev_index = json.load(open(cur_dev_path, 'r'))
        print(f"load index from path {cur_path}")
    else:
        index, dev_index = few_shot_sample(data, shot=shot, seed=seed, choice=choice)
        json.dump(index, open(cur_path, 'w'))
        json.dump(dev_index, open(cur_dev_path, 'w'))
    fewshot_data = [data[i] for i in index]
    dev_data = [data[i] for i in dev_index]
    print(f"------------ length few-shot: {len(fewshot_data)} ------------")
    print(f"------------ length dev: {len(dev_data)} ------------")
    return fewshot_data, dev_data