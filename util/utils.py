# -*- coding:utf-8 -*-
import torch
import numpy as np
from transformers import __version__ as transformers_version
import random
from transformers import BertTokenizer

from transformers import BertConfig, BertForMaskedLM
from openprompt.plms.mlm import MLMTokenizerWrapper
import argparse

logger = None


def print_info(info, file=None):
    if logger is not None:
        logger.info(info)
    else:
        print(info, file=file)


def parse_args(model="hierCRF"):
    parser = argparse.ArgumentParser("")

    parser.add_argument("--model", type=str, default=model, choices=['hierVerb', 'hierCRF'])
    parser.add_argument("--model_name_or_path", default='bert-base-uncased')
    parser.add_argument("--result_file", type=str, default="few_shot_train.txt")
    parser.add_argument("--multi_mask", type=int, default=1)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--shuffle", default=0, type=int)

    parser.add_argument("--do_train", default=1, type=int)
    parser.add_argument("--do_dev", default=1, type=bool)
    parser.add_argument("--do_test", default=1, type=bool)

    parser.add_argument("--not_manual", default=False, type=int)
    parser.add_argument("--depth", default=2, type=int)

    parser.add_argument("--device", default=0, type=int)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--dataset", default="wos", type=str)
    parser.add_argument("--eval_mode", default=0, type=int)
    parser.add_argument("--use_hier_mean", default=1, type=int)
    parser.add_argument("--multi_verb", default=1, type=int)

    parser.add_argument("--use_scheduler1", default=1, type=int)
    parser.add_argument("--use_scheduler2", default=1, type=int)

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_seq_lens", default=512, type=int, help="Max sequence length.")
    parser.add_argument("--use_withoutWrappedLM", default=False, type=bool)
    parser.add_argument('--mean_verbalizer', default=True, type=bool)

    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=171)

    parser.add_argument("--freeze_plm", default=0, type=int)

    parser.add_argument("--plm_eval_mode", default=False)
    parser.add_argument("--verbalizer", type=str, default="soft")

    parser.add_argument("--template_id", default=0, type=int)

    parser.add_argument("--multi_label", default=0, type=int)

    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument("--eval_full", default=0, type=int)
    if model == "hierVerb":
        parser.add_argument("--use_new_ct", default=1, type=int)
        parser.add_argument("--contrastive_loss", default=1, type=int)
        parser.add_argument("--contrastive_level", default=1, type=int)
        parser.add_argument("--contrastive_alpha", default=0.99, type=float)
        parser.add_argument("--contrastive_logits", default=1, type=int)
        parser.add_argument("--use_dropout_sim", default=1, type=int)
        parser.add_argument("--imbalanced_weight", default=True, type=bool)
        parser.add_argument("--imbalanced_weight_reverse", default=True, type=bool)
        parser.add_argument("--max_epochs", type=int, default=20)
        parser.add_argument("--constraint_loss", default=1, type=int)
        parser.add_argument("--constraint_alpha", default=-1, type=float)
        parser.add_argument("--cs_mode", default=0, type=int)

        parser.add_argument("--lm_training", default=1, type=int)
        parser.add_argument("--lm_alpha", default=0.999, type=float)

        parser.add_argument("--lr", default=5e-5, type=float)
        parser.add_argument("--lr2", default=1e-4, type=float)

        parser.add_argument("--batch_size", default=5, type=int)
        parser.add_argument("--eval_batch_size", default=20, type=int)

        args = parser.parse_args()
        return args
    elif model == "hierCRF":

        parser.add_argument("--lr", default=5e-5, type=float)
        parser.add_argument("--lr2", default=1e-4, type=float)
        parser.add_argument("--lr3", default=5e-2, type=float)
        parser.add_argument("--max_epochs", type=int, default=50)
        parser.add_argument("--hierCRF_loss", default=1, type=int)

        parser.add_argument("--hierCRF_alpha", default=-1, type=float)
        parser.add_argument("--batch_size", default=10, type=int)
        parser.add_argument("--eval_batch_size", default=20, type=int)

        parser.add_argument("--multi_verb_loss", default=1, type=int)
        parser.add_argument("--multi_verb_loss_alpha", default=-1, type=int)

        parser.add_argument("--lm_training", default=0, type=int)
        parser.add_argument("--lm_alpha", default=0.999, type=float)

        args = parser.parse_args()
        return args
    else:
        raise NotImplementedError


def load_plm_from_config(args, model_path, specials_to_add=None, **kwargs):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    model_config = BertConfig.from_pretrained(model_path)

    model_config.hidden_dropout_prob = args.dropout

    model = BertForMaskedLM.from_pretrained(model_path, config=model_config)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    wrapper = MLMTokenizerWrapper

    return model, tokenizer, model_config, wrapper


def seed_torch(seed=1029):
    print('Set seed to', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _mask_tokens(tokenizer, input_ids):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = input_ids.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
    if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
        ignore_value = -100
    else:
        ignore_value = -1

    labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels