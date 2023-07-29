# -*- coding:utf-8 -*-
"""
Python 3.6
author：jike
date：2022年08月23日
"""
import os
import datasets
import copy
import torch
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from openprompt.data_utils.utils import InputExample
import random
import json
from openprompt.data_utils.data_sampler import FewShotSampler

base_path = "dataset"
data_path = os.path.join(base_path, "WebOfScience")

low_res_path = os.path.join(data_path, "low-res")
few_shot_path = os.path.join(base_path, "few-shot", "WebOfScience")
if not os.path.exists(few_shot_path):
    os.mkdir(few_shot_path)

label1 = [line.strip() for line in open(os.path.join(data_path, "formatted_data", "label1.txt")).readlines()]


def get_train_dev_test(mode=None):
    if mode is None:
        mode = ['train', 'val', 'test']
    wos_dataset = dict()
    for p in mode:
        wos_dataset[p] = process_data(os.path.join(data_path, f"wos_{p}.json"))
    return wos_dataset


def process_data(path):
    if os.path.exists(path+".pt"):
        return torch.load(path+".pt")
    datas = []
    with open(path, 'r') as fp:
        for line in list(fp):
            line = line.strip()
            data = json.loads(line)
            datas.append([data['doc_token'], label1.index(data['doc_label'][1])])
    torch.save(datas, path+".pt")
    return datas


dataset = get_train_dev_test()
fewshot_dataset = copy.deepcopy(dataset)
low_res_dataset = copy.deepcopy(dataset)


def few_shot_sample(dataset, shot=5, seed=171, type="train"):
    # index = [i for i in range(len(dataset['train']))]
    train = dataset[type]

    sampler = FewShotSampler(num_examples_per_label=shot)
    examples = []

    for idx, label in enumerate(train[1]):
        examples.append(InputExample(guid=str(idx), text_a=str(idx), label=label))
    fewshot_examples = sampler(examples, seed=seed)
    index = []
    for example in fewshot_examples:
        index.append(int(example.guid))

    return index


# 通过seed和shot参数获取数据集，本地已预加载过index
def sub_dataset(shot=5, seed=171, ratio=-1, ratio_flag=0):
    print(f"------------ using seed {seed} ------------")
    if ratio > 0:
        print(f"------------ loading low-res for  {ratio} and {ratio_flag} ------------")
        if os.path.exists(os.path.join(low_res_path, f'low_res-{ratio}-{ratio_flag}.json')):
            index = json.load(open(os.path.join(low_res_path, f'low_res-{ratio}-{ratio_flag}.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(low_res_path, f'low_res-{ratio}-{ratio_flag}.json'), 'w'))
        low_res_dataset['train'] = [dataset['train'][i] for i in index[len(index) // 5:len(index) // 10 * 3]]
        print(f"------------ length low-res: {len(low_res_dataset['train'])} ------------")
        return low_res_dataset
    if shot > 0:
        print(f"------------ loading few-shot for {shot} shot ------------")
        cur_path = os.path.join(few_shot_path, f"seed_{seed}-shot_{shot}.json")
        if os.path.exists(cur_path):
            index = json.load(open(cur_path, 'r'))
            print(f"load index from path {cur_path}")
        else:
            index = few_shot_sample(dataset, shot=shot, seed=seed)
            json.dump(index, open(cur_path, 'w'))
        fewshot_dataset['train'] = [dataset['train'][i] for i in index]
        print(f"------------ length few-shot: {len(fewshot_dataset['train'])} ------------")
        return fewshot_dataset
    print(f"------------ loading whole dataset ------------")
    print(f"------------ length dataset: {len(dataset['train'])} ------------")
    return dataset
