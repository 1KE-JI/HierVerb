import torch
import torch.nn as nn

from torch.utils.data.sampler import RandomSampler, SequentialSampler

from torch.utils.data import Dataset
from typing import *
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from collections import defaultdict

from openprompt.data_utils import InputExample, InputFeatures
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from openprompt.utils import signature


from torch.utils.data import (DataLoader, TensorDataset)


class SinglePathPromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """

    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper_class: TokenizerWrapper,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 multi_gpu: bool = False,
                 mode: str = "train",
                 **kwargs,
                 ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
            "max_seq_length": max_seq_length,
            "truncate_method": truncate_method,
            "decoder_max_length": decoder_max_length,
            "predict_eos_token": predict_eos_token,
            "tokenizer": tokenizer,
            **kwargs,
        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}

        self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # processs
        self.wrap()
        self.tokenize()

        print("start convert_features_to_dataset")
        self.tensor_dataset = self.convert_features_to_dataset()
        if multi_gpu:
            self.shuffle = False

        if mode == 'train' and multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self.tensor_dataset)
        else:
            if self.shuffle:
                sampler = RandomSampler(self.tensor_dataset)
            else:
                sampler = None
        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            sampler=sampler,
            collate_fn=SinglePathPromptDataLoader.collate_fct,
            drop_last=drop_last,
            pin_memory=True
        )

    def convert_features_to_dataset(self):

        def convert_tensor_to_numpy(item):
            if isinstance(item, torch.Tensor):
                if item.dim() == 0:
                    pass
                elif item.dim() == 1:
                    item = item.cpu().detach().numpy()
            elif isinstance(item, str):
                item = int(item)
            else:
                pass
            return item

        all_input_ids = torch.tensor([convert_tensor_to_numpy(f['input_ids']) for f in self.tensor_dataset],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([convert_tensor_to_numpy(f['attention_mask']) for f in self.tensor_dataset],
                                      dtype=torch.long)
        all_label_ids = torch.tensor([convert_tensor_to_numpy(f['label']) for f in self.tensor_dataset],
                                     dtype=torch.long)
        all_loss_ids = torch.tensor([convert_tensor_to_numpy(f['loss_ids']) for f in self.tensor_dataset],
                                    dtype=torch.long)
        all_guid_ids = torch.tensor([convert_tensor_to_numpy(f['guid']) for f in self.tensor_dataset], dtype=torch.long)
        tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_loss_ids,
                                       all_guid_ids)
        return tensor_dataset

    @staticmethod
    def collate_fct(batch):
        r'''
                This function is used to collate the input_features.

                Args:
                    batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

                Returns:
                    :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
                '''
        collate_batch = []
        for idx in range(len(batch[0])):
            collate_batch.append(torch.stack([i[idx] for i in batch]))
        return collate_batch
        # return_dict = {}
        # keys = ['input_ids', 'attention_mask', 'label', 'loss_ids', 'guid']
        # for i, key in enumerate(keys):
        #     return_dict[key] = torch.stack([data[i] for data in batch])
        # return InputFeatures(**return_dict)

    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer,
                                                           'wrap_one_example'):  # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset), desc='tokenizing'):
            # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(
                **self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing),
                **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self, ):
        return self.dataloader.__iter__()


class MyPromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """

    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper_class: TokenizerWrapper,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                 ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
            "max_seq_length": max_seq_length,
            "truncate_method": truncate_method,
            "decoder_max_length": decoder_max_length,
            "predict_eos_token": predict_eos_token,
            "tokenizer": tokenizer,
            **kwargs,
        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}

        self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # processs
        self.wrap()
        self.tokenize()

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=MyPromptDataLoader.collate_fct,
            drop_last=drop_last,
        )

    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer,
                                                           'wrap_one_example'):  # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset), desc='tokenizing'):
            # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(
                **self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing),
                **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self, ):
        return self.dataloader.__iter__()

    @staticmethod
    def collate_fct(batch: List):
        r'''
        This function is used to collate the input_features.

        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        '''

        elem = batch[0]
        return_dict = {}
        for key in elem:
            if key == "encoded_tgt_text":
                return_dict[key] = [d[key] for d in batch]
            elif key == "label":
                batch_labels = [d[key] for d in batch]
                return_dict[key] = batch_labels
            else:
                try:
                    return_dict[key] = default_collate([d[key] for d in batch])
                except:
                    print(f"key{key}\n d {[batch[i][key] for i in range(len(batch))]} ")

        return InputFeatures(**return_dict)
