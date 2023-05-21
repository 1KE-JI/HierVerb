import openprompt
from openprompt import PromptForClassification
from openprompt.prompt_base import Template, Verbalizer
import torch
from typing import List
from transformers.utils.dummy_pt_objects import PreTrainedModel
from tqdm import tqdm
from transformers import BertTokenizer
from util.utils import _mask_tokens
from util.eval import compute_score, compute_based_on_path
from models.loss import constraint_multi_depth_loss_func, flat_contrastive_loss_func



class HierVerbPromptForClassification(PromptForClassification):
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 verbalizer_list: List[Verbalizer],
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 verbalizer_mode=False,
                 args=None,
                 processor=None,
                 logger=None,
                 use_cuda=True
                 ):
        super().__init__(plm=plm, template=template, verbalizer=verbalizer_list[0], freeze_plm=freeze_plm,
                         plm_eval_mode=plm_eval_mode)
        self.verbalizer_list = verbalizer_list
        self.verbLength = len(self.verbalizer_list)
        self.verbalizer_mode = verbalizer_mode

        for idx, verbalizer in enumerate(self.verbalizer_list):
            self.__setattr__(f"verbalizer{idx}", verbalizer)
        self.args = args
        self.processor = processor
        self.use_cuda = use_cuda
        self.logger = logger
        if self.args.mean_verbalizer:
            self.init_embeddings()
        self.flag_constraint_loss = False
        self.flag_contrastive_loss = False
        self.flag_contrastive_logits = False

    def forward(self, batch) -> torch.Tensor:
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
        """

        # debug
        loss = 0
        loss_details = [0, 0, 0, 0]
        lm_loss = None
        constraint_loss = None
        contrastive_loss = None
        args = self.args
        if args.use_dropout_sim and self.training:
            if not self.flag_contrastive_logits:
                print("using contrastive_logits")
                self.flag_contrastive_logits = True
            contrastive_batch = dict()
            for k, v in batch.items():
                tmp = []
                for i in v:
                    tmp.append(i)
                    tmp.append(i)
                contrastive_batch[k] = torch.stack(tmp) if isinstance(tmp[0], torch.Tensor) else tmp
                contrastive_batch[k] = contrastive_batch[k].to("cuda:0")
            batch = contrastive_batch

        outputs = self.prompt_model(batch)
        outputs = self.verbalizer_list[0].gather_outputs(outputs)
        # outputs = self.verbalizer1.gather_outputs(outputs)

        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)
        logits = []
        for idx in range(self.verbLength):
            label_words_logtis = self.__getattr__(f"verbalizer{idx}").process_outputs(outputs_at_mask[:, idx, :],
                                                                                      batch=batch)
            logits.append(label_words_logtis)

        if self.training:

            labels = batch['label']

            hier_labels = []
            hier_labels.insert(0, labels)
            for idx in range(args.depth - 2, -1, -1):
                cur_depth_labels = torch.zeros_like(labels)
                for i in range(len(labels)):
                    # cur_depth_labels[i] = label1_to_label0_mapping[labels[i].tolist()]
                    cur_depth_labels[i] = self.processor.hier_mapping[idx][1][hier_labels[0][i].tolist()]
                hier_labels.insert(0, cur_depth_labels)

            ## MLM loss
            if args.lm_training:

                input_ids = batch['input_ids']
                input_ids, labels = _mask_tokens(self.tokenizer, input_ids.cpu())

                lm_inputs = {"input_ids": input_ids, "attention_mask": batch['attention_mask'], "labels": labels}

                for k, v in lm_inputs.items():
                    if v is not None:
                        lm_inputs[k] = v.to(self.device)
                lm_loss = self.plm(**lm_inputs)[0]

            if args.multi_label:
                loss_func = torch.nn.BCEWithLogitsLoss()
            else:
                loss_func = torch.nn.CrossEntropyLoss()

            for idx, cur_depth_label in enumerate(hier_labels):
                cur_depth_logits = logits[idx]
                if args.multi_label:
                    cur_multi_label = torch.zeros_like(cur_depth_logits)

                    for i in range(cur_multi_label.shape[0]):
                        cur_multi_label[i][cur_depth_label[i]] = 1
                    cur_depth_label = cur_multi_label
                loss += loss_func(cur_depth_logits, cur_depth_label)

            loss_details[0] += loss.item()  # 层级二loss
            ## hierarchical constraint chain
            if args.constraint_loss:
                if not self.flag_constraint_loss:
                    print(f"using constraint loss with cs_mode {args.cs_mode} eval_mode {args.eval_mode}")
                    self.flag_constraint_loss = True
                constraint_loss = constraint_multi_depth_loss_func(logits, loss_func, hier_labels, self.processor, args,
                                                                   use_cuda=self.use_cuda, mode=args.cs_mode)
            ## flat contrastive loss
            if args.contrastive_loss:
                if not self.flag_contrastive_loss:
                    print(f"using flat contrastive loss with alpha {args.contrastive_alpha}")
                    if args.use_dropout_sim:
                        print("using use_dropout_sim")
                    self.flag_contrastive_loss = True
                contrastive_loss = flat_contrastive_loss_func(hier_labels, self.processor,
                                                                            outputs_at_mask,
                                                                            imbalanced_weight=args.imbalanced_weight,
                                                                            contrastive_level=args.contrastive_level,
                                                                            imbalanced_weight_reverse=args.imbalanced_weight_reverse,
                                                                            depth=args.depth,
                                                                            use_cuda=self.use_cuda)

            if lm_loss is not None:
                if args.lm_alpha != -1:
                    loss = loss * args.lm_alpha + (1 - args.lm_alpha) * lm_loss
                else:
                    loss += lm_loss
                loss_details[1] += lm_loss.item()

            if constraint_loss is not None:
                if args.constraint_alpha != -1:
                    loss = loss * args.constraint_alpha + (1 - args.constraint_alpha) * constraint_loss
                else:
                    loss += constraint_loss
                loss_details[2] += constraint_loss.item()

            if contrastive_loss is not None:
                if args.contrastive_alpha != -1:
                    # loss = loss * contrastive_alpha + (1 - contrastive_alpha) * contrastive_loss
                    loss += (1 - args.contrastive_alpha) * contrastive_loss
                else:
                    loss += contrastive_loss
                loss_details[3] += contrastive_loss.item()

            return logits, loss, loss_details
        else:
            return logits

    def init_embeddings(self):
        self.print_info("using label emb for soft verbalizer")

        label_emb_list = []
        for idx in range(self.args.depth):

            label_dict = self.processor.label_list[idx]
            label_dict = dict({idx: v for idx, v in enumerate(label_dict)})
            label_dict = {i: self.tokenizer.encode(v) for i, v in label_dict.items()}
            label_emb = []
            input_embeds = self.plm.get_input_embeddings()

            for i in range(len(label_dict)):
                label_emb.append(
                    input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
            label_emb = torch.stack(label_emb)
            label_emb_list.append(label_emb)
        if self.args.use_hier_mean:
            for depth_idx in range(self.args.depth - 2, -1, -1):
                cur_label_emb = label_emb_list[depth_idx]
                cur_depth_length = len(self.processor.label_list[depth_idx])
                for i in range(cur_depth_length):
                    cur_label_emb[i] = cur_label_emb[i] + label_emb_list[depth_idx + 1][
                                                          self.processor.hier_mapping[depth_idx][0][i], :].mean(dim=0)
                label_emb_list[depth_idx] = cur_label_emb

        for idx in range(self.args.depth):
            label_emb = label_emb_list[idx]
            self.print_info(f"depth {idx}: {label_emb.shape}")
            if "0.1.2" in openprompt.__path__[0]:
                self.__getattr__(f"verbalizer{idx}").head_last_layer.weight.data = label_emb
                self.__getattr__(f"verbalizer{idx}").head_last_layer.weight.data.requires_grad = True
            else:
                getattr(self.__getattr__(f"verbalizer{idx}").head.predictions,
                        'decoder').weight.data = label_emb
                getattr(self.__getattr__(f"verbalizer{idx}").head.predictions,
                        'decoder').weight.data.requires_grad = True

    def evaluate(self, dataloader, processor, desc="Valid", mode=0, device="cuda:0", args=None):
        self.eval()
        pred = []
        truth = []
        pbar = tqdm(dataloader, desc=desc)
        hier_mapping = processor.hier_mapping
        depth = len(hier_mapping) + 1
        all_length = len(processor.all_labels)
        for step, batch in enumerate(pbar):
            if hasattr(batch, 'cuda'):
                batch = batch.cuda()
            else:
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                batch = {"input_ids": batch[0], "attention_mask": batch[1],
                         "label": batch[2], "loss_ids": batch[3]}
            logits = self(batch)
            leaf_labels = batch['label']

            hier_labels = []
            hier_labels.insert(0, leaf_labels)
            for idx in range(depth - 2, -1, -1):
                cur_depth_labels = torch.zeros_like(leaf_labels)
                for i in range(len(leaf_labels)):
                    cur_depth_labels[i] = hier_mapping[idx][1][hier_labels[0][i].tolist()]
                hier_labels.insert(0, cur_depth_labels)

            if isinstance(logits, list):
                leaf_logits = logits[-1]
            elif isinstance(logits, torch.Tensor):
                leaf_logits = logits[:, -1, :]
            leaf_logits = torch.softmax(leaf_logits, dim=-1)
            batch_preds = []
            batch_golds = []

            leaf_preds = torch.argmax(leaf_logits, dim=-1).cpu().tolist()
            leaf_labels = leaf_labels.cpu().tolist()

            batch_preds.insert(0, leaf_preds)
            batch_golds.insert(0, leaf_labels)

            batch_s = leaf_logits.shape[0]
            flat_slot2value = processor.flat_slot2value
            hier_logits = []
            hier_logits.insert(0, leaf_logits)

            for depth_idx in range(depth - 2, -1, -1):
                ori_logits = torch.softmax(logits[depth_idx], dim=-1)

                if ori_logits.shape[-1] != all_length:
                    cur_logits = torch.zeros(batch_s, len(processor.label_list[depth_idx]))
                    for i in range(cur_logits.shape[-1]):
                        cur_logits[:, i] = torch.mean(hier_logits[0][:, list(hier_mapping[depth_idx][0][i])], dim=-1)
                else:
                    cur_logits = torch.zeros(batch_s, all_length)
                    cd_labels = processor.depth2label[depth_idx]
                    for i in range(all_length):
                        if i in cd_labels:
                            cur_logits[:, i] = torch.sum(hier_logits[0][:, list(flat_slot2value[i])], dim=-1)

                cur_logits = cur_logits.to(device)

                # for i in range(cur_label_size):
                #     cur_logits[:, i] = torch.sum(hier_logits[0][:, cur_mapping[0][i]], dim=-1)
                if mode == 0:
                    softmax_label_logits = ori_logits
                elif mode == 1:
                    softmax_label_logits = torch.softmax(cur_logits, dim=-1)
                elif mode == 2:
                    softmax_label_logits = torch.softmax(cur_logits, dim=-1) + ori_logits
                    softmax_label_logits = torch.softmax(softmax_label_logits, dim=-1)

                cur_preds = torch.argmax(softmax_label_logits, dim=-1).cpu().tolist()
                cur_golds = hier_labels[depth_idx].cpu().tolist()

                hier_logits.insert(0, softmax_label_logits)
                batch_preds.insert(0, cur_preds)
                batch_golds.insert(0, cur_golds)
            batch_preds = torch.tensor(batch_preds).transpose(1, 0).cpu().tolist()
            batch_golds = torch.tensor(batch_golds).transpose(1, 0).cpu().tolist()

            for i in range(batch_s):
                sub_preds = []
                sub_golds = []
                prev_label_size = 0
                for depth_idx in range(depth):

                    if depth_idx == 0:
                        sub_preds.append(batch_preds[i][depth_idx])
                        sub_golds.append(batch_golds[i][depth_idx])

                        continue
                    prev_mapping = hier_mapping[depth_idx - 1]
                    prev_label_size = len(prev_mapping[0]) + prev_label_size
                    if leaf_logits.shape[-1] == all_length:
                        sub_preds.append(batch_preds[i][depth_idx])
                    else:
                        sub_preds.append(batch_preds[i][depth_idx] + prev_label_size)
                    sub_golds.append(batch_golds[i][depth_idx] + prev_label_size)
                pred.append(sub_preds)
                truth.append(sub_golds)

        label_dict = dict({idx: label for idx, label in enumerate(processor.all_labels)})
        if args is None:
            scores = compute_score(pred, truth, label_dict)
        else:
            scores = compute_based_on_path(pred, truth, label_dict, processor, args)
        return scores

    def print_info(self, info):
        if self.logger is not None:
            self.logger.info(info)
        else:
            print(info)

    def state_dict(self, *args, **kwargs):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
        _state_dict['template'] = self.template.state_dict(*args, **kwargs)
        for idx in range(self.verbLength):
            _state_dict[f'verbalizer{idx}'] = self.__getattr__(f"verbalizer{idx}").state_dict(*args, **kwargs)
        return _state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
        self.template.load_state_dict(state_dict['template'], *args, **kwargs)

        for idx in range(self.verbLength):
            self.__getattr__(f"verbalizer{idx}").load_state_dict(state_dict[f'verbalizer{idx}'], *args, **kwargs)