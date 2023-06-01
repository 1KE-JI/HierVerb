'''
Author: jike
Date: 2022-10-08 09:40:03
LastEditTime: 2022-11-21 15:57:29
LastEditors: jike
FilePath: /mnt/jike/paper/nlu/paper/train.py
'''

import datetime
import logging
from tqdm import tqdm
import os
import torch
import argparse
import openprompt
from openprompt.utils.reproduciblity import set_seed
from openprompt.prompts import SoftVerbalizer, ManualTemplate

from models.hierVerb import HierVerbPromptForClassification

from processor import PROCESSOR

from util.utils import load_plm_from_config, print_info
from util.data_loader import SinglePathPromptDataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

use_cuda = True


def main():
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser("")

    parser.add_argument("--model", type=str, default='bert')
    parser.add_argument("--model_name_or_path", default='bert-base-uncased')
    parser.add_argument("--result_file", type=str, default="few_shot_train.txt")

    parser.add_argument("--multi_mask", type=int, default=1)

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--shuffle", default=0, type=int)
    parser.add_argument("--contrastive_logits", default=1, type=int)
    parser.add_argument("--constraint_loss", default=1, type=int)
    parser.add_argument("--cs_mode", default=0, type=int)
    parser.add_argument("--dataset", default="wos", type=str)
    parser.add_argument("--eval_mode", default=0, type=int)
    parser.add_argument("--use_hier_mean", default=1, type=int)
    parser.add_argument("--freeze_plm", default=0, type=int)

    parser.add_argument("--multi_label", default=0, type=int)
    parser.add_argument("--multi_verb", default=1, type=int)

    parser.add_argument("--use_scheduler1", default=1, type=int)
    parser.add_argument("--use_scheduler2", default=1, type=int)

    parser.add_argument("--constraint_alpha", default=-1, type=float)

    parser.add_argument("--imbalanced_weight", default=True, type=bool)
    parser.add_argument("--imbalanced_weight_reverse", default=True, type=bool)

    parser.add_argument("--device", default=-1, type=int)

    parser.add_argument("--lm_training", default=1, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lr2", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_seq_lens", default=512, type=int, help="Max sequence length.")

    parser.add_argument("--use_new_ct", default=1, type=int)
    parser.add_argument("--contrastive_loss", default=1, type=int)
    parser.add_argument("--contrastive_alpha", default=0.99, type=float)

    parser.add_argument("--contrastive_level", default=1, type=int)
    parser.add_argument("--use_dropout_sim", default=1, type=int)
    parser.add_argument("--batch_size", default=5, type=int)

    parser.add_argument("--use_withoutWrappedLM", default=False, type=bool)
    parser.add_argument('--mean_verbalizer', default=True, type=bool)
    parser.add_argument("--lm_alpha", default=0.999, type=float)

    parser.add_argument("--shot", type=int, default=1)

    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument("--plm_eval_mode", default=False)
    parser.add_argument("--verbalizer", type=str, default="soft")

    parser.add_argument("--template_id", default=0, type=int)

    parser.add_argument("--not_manual", default=False, type=int)
    parser.add_argument("--depth", default=2, type=int)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)

    parser.add_argument("--early_stop", default=10, type=int)

    parser.add_argument("--eval_full", default=0, type=int)

    args = parser.parse_args()
    if args.device != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
        device = torch.device("cuda:0")
        use_cuda = True
    else:
        use_cuda = False
        device = torch.device("cpu")

    if args.contrastive_loss == 0:
        args.contrastive_logits = 0
        args.use_dropout_sim = 0

    if args.shuffle == 1:
        args.shuffle = True
    else:
        args.shuffle = False
    print_info(args)
    processor = PROCESSOR[args.dataset](shot=args.shot, seed=args.seed)

    train_data = processor.train_example
    dev_data = processor.dev_example
    test_data = processor.test_example
    train_data = [[i.text_a, i.label] for i in train_data]
    dev_data = [[i.text_a, i.label] for i in dev_data]
    test_data = [[i.text_a, i.label] for i in test_data]
    hier_mapping = processor.hier_mapping
    args.depth = len(hier_mapping) + 1

    print_info("final train_data length is: {}".format(len(train_data)))
    print_info("final dev_data length is: {}".format(len(dev_data)))
    print_info("final test_data length is: {}".format(len(test_data)))

    args.template_id = 0

    set_seed(args.seed)

    plm, tokenizer, model_config, WrapperClass = load_plm_from_config(args, args.model_name_or_path)
    # dataset
    dataset = {}
    dataset['train'] = processor.train_example
    dataset['dev'] = processor.dev_example
    dataset['test'] = processor.test_example

    max_seq_l = args.max_seq_lens
    batch_s = args.batch_size

    if args.multi_mask:
        template_file = f"{args.dataset}_mask_template.txt"
    else:
        template_file = "manual_template.txt"
    template_path = "template"
    text_mask = []
    for i in range(args.depth):
        text_mask.append(f'{i + 1} level: {{"mask"}}')
    text = f'It was {" ".join(text_mask)}. {{"placeholder": "text_a"}}'
    if not os.path.exists(template_path):
        os.mkdir(template_path)
    if not os.path.exists("ckpts"):
        os.mkdir("ckpts")
    template_path = os.path.join(template_path, template_file)
    if not os.path.exists(template_path):
        with open(template_path, 'w', encoding='utf-8') as fp:
            fp.write(text)
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(template_path, choice=args.template_id)

    print_info("train_size: {}".format(len(dataset['train'])))

    ## Loading dataset
    train_dataloader = SinglePathPromptDataLoader(dataset=dataset['train'], template=mytemplate, tokenizer=tokenizer,
                                                  tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                                  decoder_max_length=3,
                                                  batch_size=batch_s, shuffle=args.shuffle, teacher_forcing=False,
                                                  predict_eos_token=False, truncate_method="tail",
                                                  num_works=2,
                                                  multi_gpu=(args.device == -2), )
    if args.dataset == "wos":
        full_name = "WebOfScience"
    elif args.dataset == "dbp":
        full_name = "DBPedia"
    else:
        raise NotImplementedError

    test_path = os.path.join(f"dataset", full_name, f"test_dataloader-multi_mask.pt")
    dev_path = os.path.join("dataset", full_name, f"dev_dataloader-multi_mask.pt")
    eval_batch_s = 20
    if args.dataset != "dbp" and os.path.exists(dev_path):
        validation_dataloader = torch.load(dev_path)
    else:
        validation_dataloader = SinglePathPromptDataLoader(dataset=dataset["dev"], template=mytemplate,
                                                           tokenizer=tokenizer,
                                                           tokenizer_wrapper_class=WrapperClass,
                                                           max_seq_length=max_seq_l,
                                                           decoder_max_length=3,
                                                           batch_size=eval_batch_s, shuffle=False,
                                                           teacher_forcing=False,
                                                           predict_eos_token=False,
                                                           truncate_method="tail",
                                                           multi_gpu=False,
                                                           )
        if args.dataset != "dbp":
            torch.save(validation_dataloader, dev_path)
    if not os.path.exists(test_path):
        test_dataloader = SinglePathPromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                                     tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                                     decoder_max_length=3,
                                                     batch_size=eval_batch_s, shuffle=False, teacher_forcing=False,
                                                     predict_eos_token=False,
                                                     truncate_method="tail",
                                                     multi_gpu=False,
                                                     mode='test',
                                                     )
        torch.save(test_dataloader, test_path)
    else:
        test_dataloader = torch.load(test_path)

    ## build verbalizer and model
    verbalizer_list = []
    label_list = processor.label_list

    for i in range(args.depth):
        if "0.1.2" in openprompt.__path__[0]:
            verbalizer_list.append(SoftVerbalizer(tokenizer, model=plm, classes=label_list[i]))
        else:
            # verbalizer_list.append(SoftVerbalizer(tokenizer, plm=plm, classes=label_list[i]))
            verbalizer_list.append(SoftVerbalizer(tokenizer, model=plm, classes=label_list[i]))

    print_info("loading prompt model")
    prompt_model = HierVerbPromptForClassification(plm=plm, template=mytemplate, verbalizer_list=verbalizer_list,
                                              freeze_plm=args.freeze_plm, args=args, processor=processor,
                                              plm_eval_mode=args.plm_eval_mode, use_cuda=use_cuda)

    if use_cuda:
        prompt_model = prompt_model.cuda()

    ## Prepare training parameters
    # it's always good practice to set no decay to biase and LayerNorm parameters
    no_decay = ['bias', 'LayerNorm.weight']

    named_parameters = prompt_model.plm.named_parameters()

    optimizer_grouped_parameters1 = [

        {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters
    # use a learning rate of 1e−4 to fasten the convergence of its hierarchical label words’ embeddings of verbalizer0
    verbalizer = prompt_model.verbalizer
    optimizer_grouped_parameters2 = [
        {'params': verbalizer.group_parameters_1, "lr": args.lr},
        {'params': verbalizer.group_parameters_2, "lr": args.lr2},
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.lr)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs

    warmup_steps = 0
    scheduler1 = None
    scheduler2 = None
    if args.use_scheduler1:
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer1,
            num_warmup_steps=warmup_steps, num_training_steps=tot_step)
    if args.use_scheduler2:
        scheduler2 = get_linear_schedule_with_warmup(
            optimizer2,
            num_warmup_steps=warmup_steps, num_training_steps=tot_step)

    contrastive_alpha = args.contrastive_alpha
    best_score_macro = 0
    best_score_micro = 0
    best_score_macro_epoch = -1
    best_score_micro_epoch = -1
    early_stop_count = 0

    if not args.imbalanced_weight:
        args.imbalanced_weight_reverse = False
    this_run_unicode = f"{args.dataset}-seed{args.seed}-shot{args.shot}-lr{args.lr}-lr2{args.lr2}-batch_size{args.batch_size}-multi_mask{args.multi_mask}-use_new_ct{args.use_new_ct}-cs_mode{args.cs_mode}-ctl{args.contrastive_logits}" \
                       f"-contrastive_alpha{contrastive_alpha}-shuffle{args.shuffle}-constraint_loss{args.constraint_loss}-multi_verb{args.multi_verb}" \
                       f"-contrastive_level{args.contrastive_level}--use_dropout_sim{args.use_dropout_sim}-length{len(dataset['train'])}"
    print_info("saved_path: {}".format(this_run_unicode))

    if args.eval_full:
        best_record = dict()
        keys = ['p_micro_f1', 'p_macro_f1', 'c_micro_f1', 'c_macro_f1', 'P_acc']
        for key in keys:
            best_record[key] = 0

    ## start training
    for epoch in range(args.max_epochs):
        print_info("------------ epoch {} ------------".format(epoch + 1))
        if early_stop_count >= args.early_stop:
            print_info("Early stop!")
            break

        print_info(
            f"cur lr\tscheduler1: {scheduler1.get_lr() if scheduler1 is not None else args.lr}\tscheduler2: {scheduler2.get_lr() if scheduler2 is not None else 1e-4}")

        loss_detailed = [0, 0, 0, 0]
        prompt_model.train()
        idx = 0

        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            batch = {"input_ids": batch[0], "attention_mask": batch[1],
                     "label": batch[2], "loss_ids": batch[3]}

            logits, loss, cur_loss_detailed = prompt_model(batch)
            loss_detailed = [loss_detailed[idx] + value for idx, value in enumerate(cur_loss_detailed)]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), args.max_grad_norm)

            optimizer1.step()
            optimizer2.step()

            if scheduler1 is not None:
                scheduler1.step()
            if scheduler2 is not None:
                scheduler2.step()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            idx = idx + 1
            # torch.cuda.empty_cache()
        print_info("multi-verb loss, lm loss, constraint loss, contrastive loss are: ")
        print_info(loss_detailed)

        scores = prompt_model.evaluate(validation_dataloader, processor, desc="Valid",
                                       mode=args.eval_mode)
        early_stop_count += 1
        if args.eval_full:
            score_str = ""
            for key in keys:
                score_str += f'{key} {scores[key]}\n'
            print_info(score_str)
            for k in best_record:
                if scores[k] > best_record[k]:
                    best_record[k] = scores[k]
                    torch.save(prompt_model.state_dict(), f"ckpts/{this_run_unicode}-{k}.ckpt")
                    early_stop_count = 0

        else:
            macro_f1 = scores['macro_f1']
            micro_f1 = scores['micro_f1']
            print_info('macro {} micro {}'.format(macro_f1, micro_f1))
            if macro_f1 > best_score_macro:
                best_score_macro = macro_f1
                torch.save(prompt_model.state_dict(), f"ckpts/{this_run_unicode}-macro.ckpt")
                # save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
                early_stop_count = 0
                best_score_macro_epoch = epoch

            if micro_f1 > best_score_micro:
                best_score_micro = micro_f1
                torch.save(prompt_model.state_dict(), f"ckpts/{this_run_unicode}-micro.ckpt")
                # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
                early_stop_count = 0
                best_score_micro_epoch = epoch

    ## evaluate
    if args.eval_full:
        best_keys = ['P_acc']
        for k in best_keys:
            prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}-{k}.ckpt"))

            scores = prompt_model.evaluate(test_dataloader, processor, desc="test", mode=args.eval_mode,
                                           args=args)
            tmp_str = ''
            tmp_str += f"finally best_{k} "
            for i in keys:
                tmp_str += f"{i}: {scores[i]}\t"
            print_info(tmp_str)

    else:
        # for best macro
        prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}-macro.ckpt"))

        if use_cuda:
            prompt_model = prompt_model.cuda()

        scores = prompt_model.evaluate(test_dataloader, processor, desc="test", mode=args.eval_mode)
        macro_f1_1 = scores['macro_f1']
        micro_f1_1 = scores['micro_f1']
        acc_1 = scores['acc']
        print_info('finally best macro {} {} micro {} acc {}'.format(best_score_macro_epoch, macro_f1_1, micro_f1_1, acc_1))

        # for best micro
        prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}-micro.ckpt"))

        scores = prompt_model.evaluate(test_dataloader, processor, desc="test", mode=args.eval_mode)
        macro_f1_2 = scores['macro_f1']
        micro_f1_2 = scores['micro_f1']
        acc_2 = scores['acc']
        print_info('finally best micro {} {} micro {} acc {}'.format(best_score_micro_epoch, macro_f1_2, micro_f1_2, acc_2))

    ## print and record parameter details
    content_write = "=" * 20 + "\n"
    content_write += f"start_time {start_time}" + "\n"
    content_write += f"end_time {datetime.datetime.now()}\t"
    for hyperparam, value in args.__dict__.items():
        content_write += f"{hyperparam} {value}\t"
    content_write += "\n"
    if args.eval_full:
        cur_keys = ['P_acc']
        for key in cur_keys:
            content_write += f"best_{key} "
            for i in keys:
                content_write += f"{i}: {best_record[i]}\t"
            content_write += f"\n"
    else:
        content_write += f"best_macro macro_f1: {macro_f1_1}\t"
        content_write += f"micro_f1: {micro_f1_1}\t"
        content_write += f"acc: {acc_1}\t\n"

        content_write += f"best_micro macro_f1: {macro_f1_2}\t"
        content_write += f"micro_f1: {micro_f1_2}\t"
        content_write += f"acc: {acc_2}\t"
    content_write += "\n\n"

    print_info(content_write)
    if not os.path.exists("result"):
        os.mkdir("result")
    with open(os.path.join("result", args.result_file), "a") as fout:
        fout.write(content_write)


if __name__ == "__main__":
    main()
