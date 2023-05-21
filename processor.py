# -*- coding:utf-8 -*-

import torch
import os
from openprompt.data_utils.utils import InputExample
from dataset.WebOfScience.trans_format import get_mapping
from tqdm import tqdm

from dataset.WebOfScience.my_dataset import sub_dataset

base_path = "./"


class WOSProcessor:

    def __init__(self, ratio=-1, seed=171, shot=-1, ratio_flag=0):
        super().__init__()
        self.name = 'WebOfScience'
        label0_list, label1_list, label0_label2id, label1_label2id, label0_to_label1_mapping, label1_to_label0_mapping = get_mapping()
        self.labels = label1_list
        self.coarse_labels = label0_list
        self.all_labels = label0_list + label1_list
        self.label_list = [label0_list, label1_list]
        self.label0_to_label1_mapping = label0_to_label1_mapping
        self.label1_to_label0_mapping = label1_to_label0_mapping

        self.data_path = os.path.join(base_path, "dataset", "WebOfScience")
        self.flat_slot2value, self.value2slot, self.depth2label = self.get_tree_info()
        self.hier_mapping = [[label0_to_label1_mapping, label1_to_label0_mapping]]

        self.ratio = ratio
        self.seed = seed
        self.shot = shot
        self.dataset = sub_dataset(self.shot, self.seed, self.ratio, ratio_flag=ratio_flag)
        print("length dataset['train']:", len(self.dataset['train']))

        self.train_data = self.get_dataset("train")

        self.dev_data = self.get_dataset("val")
        self.test_data = self.get_dataset("test")
        self.train_example = self.convert_data_to_examples(self.train_data)
        self.dev_example = self.convert_data_to_examples(self.dev_data)
        self.test_example = self.convert_data_to_examples(self.test_data)

        self.train_inputs = [i[0] for i in self.train_data]
        self.dev_inputs = [i[0] for i in self.dev_data]
        self.test_inputs = [i[0] for i in self.test_data]

        self.size = len(self.train_example) + len(self.test_example)

    def get_tree_info(self):
        flat_slot2value = torch.load(os.path.join(self.data_path, 'slot.pt'))

        value2slot = {}
        num_class = 0
        for s in flat_slot2value:
            for v in flat_slot2value[s]:
                value2slot[v] = s
                if num_class < v:
                    num_class = v
        num_class += 1
        for i in range(num_class):
            if i not in value2slot:
                value2slot[i] = -1

        def get_depth(x):
            depth = 0
            while value2slot[x] != -1:
                depth += 1
                x = value2slot[x]
            return depth

        depth_dict = {i: get_depth(i) for i in range(num_class)}
        max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
        depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
        return flat_slot2value, value2slot, depth2label

    def get_dataset(self, type="train"):
        data = []
        cur_dataset = self.dataset[type]
        length = len(cur_dataset)
        for i in tqdm(range(length)):
            text_a = cur_dataset[i][0]
            label = cur_dataset[i][1]
            data.append([text_a, label])
        return data

    def convert_data_to_examples(self, data):
        examples = []
        for idx, sub_data in enumerate(data):
            examples.append(InputExample(guid=str(idx), text_a=sub_data[0], label=sub_data[1]))

        return examples


PROCESSOR = {
    "wos": WOSProcessor,
    "WebOfScience": WOSProcessor,
}

