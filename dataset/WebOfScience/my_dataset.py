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


hpt_data_path = 'dataset/WebOfScience'
ours_path = "dataset/WebOfScience/low-res"
fewshot_path = "dataset/WebOfScience/few-shot"
model = "prompt"
# data_path = os.path.join('data', data)
dataset = datasets.load_from_disk(os.path.join(hpt_data_path, model))
dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
fewshot_dataset = copy.deepcopy(dataset)
low_res_dataset = copy.deepcopy(dataset)


def few_shot_sample(dataset, shot=5, seed=171, type="train", choice=1):
    # index = [i for i in range(len(dataset['train']))]
    train = dataset[type]

    sampler = FewShotSampler(num_examples_per_label=shot)
    examples = []

    for idx, label in enumerate(train.data[2]):
        label = label[choice]
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
        if os.path.exists(os.path.join(ours_path, f'low_res-{ratio}-{ratio_flag}.json')):
            index = json.load(open(os.path.join(ours_path, f'low_res-{ratio}-{ratio_flag}.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(ours_path, f'low_res-{ratio}-{ratio_flag}.json'), 'w'))
        low_res_dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])
        print(f"------------ length low-res: {len(low_res_dataset['train'])} ------------")
        return low_res_dataset
    if shot > 0:
        print(f"------------ loading few-shot for {shot} shot ------------")
        cur_path = os.path.join(fewshot_path, f"seed_{seed}-shot_{shot}.json")
        if os.path.exists(cur_path):
            index = json.load(open(cur_path, 'r'))
            print(f"load index from path {cur_path}")
        else:
            index = few_shot_sample(dataset, shot=shot, seed=seed)
            json.dump(index, open(cur_path, 'w'))
        fewshot_dataset['train'] = dataset['train'].select(index)
        print(f"------------ length few-shot: {len(fewshot_dataset['train'])} ------------")
        return fewshot_dataset
    print(f"------------ loading whole dataset ------------")
    print(f"------------ length dataset: {len(dataset['train'])} ------------")
    return dataset