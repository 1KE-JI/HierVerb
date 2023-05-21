# -*- coding:utf-8 -*-
import json
import os
from tqdm import tqdm
import torch

root_path = "dataset/WebOfScience/"


def get_mapping(flag=True):
    # flag 各级标签是否平移至从0开始index， True即平移
    slot2value = torch.load(os.path.join(root_path, 'slot.pt'))
    with open(os.path.join(root_path, "formatted_data", "label0.txt"), encoding='utf-8') as fp:
        label0_list = [line.strip() for line in list(fp)]
    with open(os.path.join(root_path, "formatted_data", "label1.txt"), encoding='utf-8') as fp:
        label1_list = [line.strip() for line in list(fp)]

    label0_label2id = dict({label: idx for idx, label in enumerate(label0_list)})
    label1_label2id = dict({label: idx for idx, label in enumerate(label1_list)})

    label0_to_label1_mapping = dict()
    for k, v in slot2value.items():
        if flag:
            v = [i-7 for i in v]
        label0_to_label1_mapping[k] = list(v)

    label1_to_label0_mapping = {}
    for key, value in label0_to_label1_mapping.items():
        for v in value:
            label1_to_label0_mapping[v] = key
    return label0_list, label1_list, label0_label2id, label1_label2id, label0_to_label1_mapping, label1_to_label0_mapping


def write_formatted_data(type):
    print(f"write_formatted_data: {type}")
    with open(os.path.join(root_path, f"wos_{type}.json"), 'r', encoding='utf-8') as fp:
        lines = [line.strip() for line in list(fp)]
    datas = []
    for line in lines:
        datas.append(json.loads(line))

    source = []

    for data in tqdm(datas):
        sentence = data['doc_token']
        labels = data['doc_label']
        label_0 = labels[0]
        label_1 = labels[1]
        tmp_str = sentence + "\t" + str(label0_label2id[label_0]) + "\t" + str(label1_label2id[label_1])
        source.append(tmp_str)

    with open(os.path.join(root_path, "formatted_data", f"{type}.txt"), 'w', encoding='utf-8') as fp:
        fp.write("\n".join(source))


if __name__ == "__main__":
    with open(os.path.join(root_path, "wos_train.json"), 'r', encoding='utf-8') as fp:
        lines = [line.strip() for line in list(fp)]

    datas = []
    for line in lines:
        datas.append(json.loads(line))
    label0_list, label1_list, label0_label2id, label1_label2id, label0_to_label1_mapping, label1_to_label0_mapping = get_mapping()

    if not os.path.exists(os.path.join(root_path, "formatted_data")):
        os.mkdir(os.path.join(root_path, "formatted_data"))

    with open(os.path.join(root_path, "formatted_data", "label0.txt"), 'w', encoding='utf-8') as fp:
        fp.write("\n".join(label0_list))
    with open(os.path.join(root_path, "formatted_data", "label1.txt"), 'w', encoding='utf-8') as fp:
        fp.write("\n".join(label1_list))

    # write_formatted_data("train")
    # write_formatted_data("val")
    # write_formatted_data("test")