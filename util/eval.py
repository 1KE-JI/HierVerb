from tqdm import tqdm
import torch
from tqdm import tqdm



def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate_for_soft_verb(prompt_model, dataloader, processor, desc="Valid", mode=0):
    prompt_model.eval()
    pred = []
    truth = []
    pbar = tqdm(dataloader, desc=desc)
    hier_mapping = processor.hier_mapping
    depth = len(hier_mapping) + 1

    batch_s = 5
    for step, inputs in enumerate(pbar):
        inputs = inputs.cuda()
        logits = prompt_model(inputs)
        logits = torch.softmax(logits, dim=-1)
        cur_preds = torch.argmax(logits, dim=-1).cpu().tolist()

        leaf_labels = inputs['label']
        hier_labels = []
        hier_labels.insert(0, leaf_labels)
        for idx in range(depth - 2, -1, -1):
            cur_depth_labels = torch.zeros_like(leaf_labels)
            for i in range(len(leaf_labels)):
                # cur_depth_labels[i] = label1_to_label0_mapping[labels[i].tolist()]
                cur_depth_labels[i] = hier_mapping[idx][1][hier_labels[0][i].tolist()]
            hier_labels.insert(0, cur_depth_labels)

        batch_golds = []
        for i in range(hier_labels[0].shape[0]):
            batch_golds.append([hier_labels[0][i].tolist(), (hier_labels[1][i] + 7).tolist()])
        for i in range(batch_s):
            pred.append(cur_preds[i])
            truth.append(batch_golds[i])

    label_dict = dict({idx: label for idx, label in enumerate(processor.all_labels)})

    scores = compute_score(pred, truth, label_dict)
    return scores


def evaluate_multi_base(prompt_model, dataloader, label0_to_label1_mapping, label1_to_label0_mapping, desc="Valid"):
    prompt_model.eval()
    pred = []
    truth = []
    pbar = tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        inputs = inputs.cuda()
        label0_logits, label1_logits = prompt_model(inputs)
        label1_gold = inputs['label']

        label0_labels = torch.zeros_like(label1_gold)
        for i in range(len(label1_gold)):
            label0_labels[i] = label1_to_label0_mapping[label1_gold[i].tolist()]

        softmax_label1_logits = torch.softmax(label1_logits, dim=-1)
        label1_pred = torch.argmax(softmax_label1_logits, dim=-1).cpu().tolist()
        label1_gold = label1_gold.cpu().tolist()

        softmax_label0_logits = torch.softmax(label0_logits, dim=-1)

        label0_pred = torch.argmax(softmax_label0_logits, dim=-1).cpu().tolist()
        label0_gold = label0_labels.cpu().tolist()

        assert len(label0_pred) == len(label0_gold)
        assert len(label1_pred) == len(label1_gold)
        assert len(label0_pred) == len(label1_pred)
        for i in range(len(label0_pred)):
            pred.append([label0_pred[i], label1_pred[i] + 7])
            truth.append([label0_gold[i], label1_gold[i] + 7])

    with open("dataset/WebOfScience/formatted_data/label0.txt", 'r') as fp:
        labels = [i.strip().lower() for i in list(fp)]
    with open("dataset/WebOfScience/formatted_data/label1.txt", 'r') as fp:
        labels.extend([i.strip().lower() for i in list(fp)])
    label_dict = dict({idx: label for idx, label in enumerate(labels)})
    scores = compute_score(pred, truth, label_dict)
    return scores


def compute_score(epoch_predicts, epoch_labels, id2label):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[int]], predicted, label id
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    # label2id = vocab.v2i['label']
    # id2label = vocab.i2v['label']
    # epoch_gold_label = list()
    # # get id label name of ground truth
    # for sample_labels in epoch_labels:
    #     sample_gold = []
    #     for label in sample_labels:
    #         assert label in id2label.keys(), print(label)
    #         sample_gold.append(id2label[label])
    #     epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    ## acc
    acc_right = 0
    acc_total = 0
    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        # count for the gold and right items
        for gold in sample_gold:
            acc_total += 1
            for label in sample_predict_id_list:
                if gold == label:
                    acc_right += 1
    acc = acc_right / acc_total
    # initialize confusion matrix

    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        # np_sample_predict = np.array(sample_predict, dtype=np.float32)
        # sample_predict_descent_idx = np.argsort(-np_sample_predict)
        # sample_predict_id_list = []

        # for j in range(top_k):
        #     if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
        #         sample_predict_id_list.append(sample_predict_descent_idx[j])

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) \
        if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'acc': acc,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}


# 1.    epoch_predicts中每一个sample是一个logits列表，长度为num_labels，该函数通过0.5阈值获得模型预测的值列表，
# 2.    然后分别统计各个label对应的gold、right、predict
def evaluate_multi_path(prompt_model, dataloader, processor, desc="Valid", mode=0, threshold=0.5, args=None):
    prompt_model.eval()
    pred = []
    truth = []
    pbar = tqdm(dataloader, desc=desc)
    depth = len(processor.hier_mapping) + 1

    label_length = [len(i) for i in processor.label_list]
    for step, inputs in enumerate(pbar):

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda:0")
        logits = prompt_model(inputs)
        flat_labels = inputs['label']

        for i in range(len(flat_labels)):
            preds = []
            for depth_idx in range(depth):
                cur_logits = torch.sigmoid(logits[depth_idx][i]).tolist()
                for idx, value in enumerate(cur_logits):
                    if value > threshold:
                        preds.append(idx + sum(label_length[:depth_idx]))
            pred.append(preds)
            truth.append(flat_labels[i])

    label_dict = dict({idx: label for idx, label in enumerate(processor.all_labels)})
    scores = dict()
    if args is not None:
        scores = compute_based_on_path(pred, truth, label_dict, processor, args)
    scores_ori = compute_score(pred, truth, label_dict)
    scores['macro_f1'] = scores_ori['macro_f1']
    scores['micro_f1'] = scores_ori['micro_f1']
    return scores





'''
以两层为案例,
核心讨论问题为：
1.是以样本所有路径为基本判断单元，还是以单路径为基本判断单元；
2.针对这些预测出的组不成路径的游离的原子label，去进行什么策略。
一、
策略：如果采用基于路径的方法，即预测对哪些路径，这些对应的路径confusion matrix + 1
可能存在争论：
1.如果严格来看，只有一个样本的所有路径都预测对，才认为这个样本计算正确
2.如果模型趋向于将其中一层完全对应正确，其他的层次的预测总是倾向于保持和已预测的label不属于一个label
（这样可以规避我们的算法计算，让我们无法统计到本该去进行错误统计的完整路径去增加predicted_confusion_matrix,进而无法增加分母去惩罚precison，也会间接导致recall偏高）

gold：path A and B
gold_count_list[A,B] += 1

case 1：
完美预测成功
pred: A(a, a4), B(b,b2) 

right_count_list[A,B] += 1
predicted_count_list[A,B] += 1


case 2：
部分预测成功,存在有其他预测错误的原子label之间也能组合成路径
pred: A(a,a1), B(b,b3), C(c,c2), d

right_count_list[A,B] += 1
predicted_count_list[A,B,C] += 1


case 3:
部分预测成功，其他预测错误的原子label之间组不成任何完整路径
pred: A(a,a1), c, d, e, f2,f3,f5,f6, q2,q4

right_count_list[A] += 1
predicted_count_list[A] += 1


case 4:
全部预测成功，但存在其他额外预测错误的原子label之间组不成任何完整路径
pred: A, B, c1, d2 

思路:
1.根据gold_path统计出其对应的原子label总数 total_atom_label
2.根据pred_path统计出其实际预测的原子label总数，pred_atom_label, 得到a=pred_atom_label/total_atom_label
3.将所有label的confusion matrix合并在一起，得到right_total, pred_total, total,去计算Micro-F1

二、
策略：如果采用严格意义基于路径的方法，即基于样本所有路径的预测正确与否，这时只有当所有label都预测正确，才将这些对应的路径 confusion matrix + 1
好处：
1.因为以样本的gold路径全集为基本判断单元，所以可以连带规避掉"预测错的组不成路径的游离原子label采取什么策略"这个问题

坏处：
1.自相矛盾，针对pathA而言，如果pathA预测对了，但是该样本其他path出错了，对于pathA来说理应进行 right和predicted都+1的

结论：直接进行acc计算

case 1：
完美预测成功
pred: A(a, a4), B(b,b2) 

right_count_list[A,B] += 1
predicted_count_list[A,B] += 1


case 2：
部分预测成功,存在有其他预测错误的原子label之间也能组合成路径
pred: A(a,a1), B(b,b3), C(c,c2), d

right_count_list[A,B] += 1
predicted_count_list[A,B,C] += 1


case 3:
部分预测成功，其他预测错误的原子label之间组不成任何完整路径
pred: A(a,a1), c, d, e, f2,f3,f5,f6, q2,q4

right_count_list[A] += 1
predicted_count_list[A] += 1


case 4:
全部预测成功，但存在其他额外预测错误的原子label之间组不成任何完整路径
pred: A, B, c1, d2 

思路:
用基于单样本所有路径单元去做acc计算
完全正确即acc的分子right+1，最后right/count_samples
'''
import os

def _get_mapping(data):
    slot2value = torch.load(os.path.join('dataset', data, 'slot.pt'))
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
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

    return slot2value, value2slot, depth2label


def get_path_set(processor):
    data = processor.name
    if data == 'wos':
        data = 'WebOfScience'
    elif data == 'dbp':
        data = 'DBPedia'

    slot2value, value2slot, depth2label = _get_mapping(data)

    all_labels = list(range(len(processor.all_labels)))
    leaf_labels = list(set(all_labels).difference(set(list(slot2value.keys()))))

    path_set = []

    for leaf_label in leaf_labels:
        cur = set()
        cur.add(leaf_label)
        parent = value2slot.get(leaf_label, -1)
        while parent != -1:
            cur.add(parent)
            parent = value2slot.get(parent, -1)
        path_set.append([cur, leaf_label])

    return path_set, value2slot


def compute_based_on_path(epoch_predicts, epoch_labels, id2label, processor, args):
    '''
    compute micro-F1 and macro-F1 based on the path unit
    :param epoch_predicts:
    :param epoch_labels:
    :param id2label:
    :return:
    '''
    path_set, value2slot = get_path_set(processor)

    epoch_gold = epoch_labels

    acc_right = 0
    acc_total = len(epoch_labels)

    predict_not_valid_atom_count = 0
    gold_atom_count = 0

    id2path = dict({i: path for i, path in enumerate(path_set)})
    # Ours Eval

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        # count for the gold and right items
        if len(set(sample_predict_id_list).intersection(set(sample_gold))) == len(sample_predict_id_list) \
                and len(sample_predict_id_list) == len(sample_gold):
            acc_right += 1
        predict_not_valid_atom_count += len(sample_predict_id_list)
        gold_atom_count += len(sample_gold)
    # evaluate acc based on the sample
    acc = acc_right / acc_total

    ## initialize confusion matrix
    # P-matrix
    right_count_list = [0 for _ in range(len(id2path))]
    gold_count_list = [0 for _ in range(len(id2path))]
    predicted_count_list = [0 for _ in range(len(id2path))]
    # C-matrix
    c_right_count_list = [0 for _ in range(len(id2label))]
    c_gold_count_list = [0 for _ in range(len(id2label))]
    c_predicted_count_list = [0 for _ in range(len(id2label))]

    wrong_atom_count = 0

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        predict_path_idxs = []
        gold_path_idxs = []
        right_idxs = []
        # compute for our P-matrix
        for idx, path in enumerate(path_set):
            # count for predict confusion matrix
            path = path[0]
            if len(path.intersection(set(sample_predict_id_list))) == len(path):
                predicted_count_list[idx] += 1
                predict_path_idxs.append(idx)

            if len(path.intersection(set(sample_gold))) == len(path):
                gold_count_list[idx] += 1
                gold_path_idxs.append(idx)
        for right_idx in set(gold_path_idxs).intersection(predict_path_idxs):
            right_count_list[right_idx] += 1
            right_idxs.append(right_idx)
        valid_count = 0
        for idx in predict_path_idxs:
            valid_count += len(path_set[idx][0])
        wrong_atom_count += len(sample_predict_id_list) - valid_count
        # compute for alibaba C-matrix
        # count for the gold and right items
    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        gold_idxs = []
        right_idxs = []

        for gold in sample_gold:
            for label in sample_predict_id_list:
                if gold == label:
                    right_idxs.append(gold)
        for right_idx in right_idxs:
            flag = True
            parent = value2slot.get(right_idx, -1)

            while parent != -1:
                if parent not in right_idxs:
                    flag = False
                    break
                parent = value2slot.get(parent, -1)
            if flag:
                c_right_count_list[right_idx] += 1

        for gold in sample_gold:
            c_gold_count_list[gold] += 1
        # count for the predicted items
        for label in sample_predict_id_list:
            c_predicted_count_list[label] += 1

    ## P-matrix
    p_precision_dict = dict()
    p_recall_dict = dict()
    p_fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, path in id2path.items():
        leaf_label = path[1]
        label = str(leaf_label) + '_' + str(i)
        p_precision_dict[label], p_recall_dict[label], p_fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                                   predicted_count_list[
                                                                                                       i],
                                                                                                   gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # PMacro-F1
    p_precision_macro = sum([v for _, v in p_precision_dict.items()]) / len(list(p_precision_dict.keys()))
    p_recall_macro = sum([v for _, v in p_recall_dict.items()]) / len(list(p_precision_dict.keys()))
    p_ori_macro_f1 = sum([v for _, v in p_fscore_dict.items()]) / len(list(p_fscore_dict.keys()))

    # PMicro-F1
    p_precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    p_recall_micro = float(right_total) / gold_total if gold_total > 0 else 0.0
    p_ori_micro_f1 = 2 * p_precision_micro * p_recall_micro / (p_precision_micro + p_recall_micro) \
        if (p_precision_micro + p_recall_micro) > 0 else 0.0
    x = wrong_atom_count / gold_atom_count
    a = 1 - 2 * (1 / (1 + torch.e ** (-x)) - 0.5)
    p_macro_f1 = a * p_ori_macro_f1
    p_micro_f1 = a * p_ori_micro_f1
    ## C-matrix
    c_precision_dict = dict()
    c_recall_dict = dict()
    c_fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, path in id2label.items():
        leaf_label = path[1]
        label = leaf_label + '_' + str(i)
        c_precision_dict[label], c_recall_dict[label], c_fscore_dict[label] = _precision_recall_f1(
            c_right_count_list[i],
            c_predicted_count_list[i],
            c_gold_count_list[i]
        )
        right_total += c_right_count_list[i]
        gold_total += c_gold_count_list[i]
        predict_total += c_predicted_count_list[i]
    # CMacro-F1
    c_precision_macro = sum([v for _, v in c_precision_dict.items()]) / len(list(c_precision_dict.keys()))
    c_recall_macro = sum([v for _, v in c_recall_dict.items()]) / len(list(c_precision_dict.keys()))
    c_macro_f1 = sum([v for _, v in c_fscore_dict.items()]) / len(list(c_fscore_dict.keys()))
    # CMicro-F1
    c_precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    c_recall_micro = float(right_total) / gold_total
    c_micro_f1 = 2 * c_precision_micro * c_recall_micro / (c_precision_micro + c_recall_micro) \
        if (c_precision_micro + c_recall_micro) > 0 else 0.0
    result = {'p_precision': p_precision_micro,
              'p_recall': p_recall_micro,
              'p_micro_f1': p_micro_f1,
              'p_macro_f1': p_macro_f1,
              'p_ori_micro_f1': p_ori_micro_f1,
              'p_ori_macro_f1': p_ori_macro_f1,
              'c_precision': c_precision_micro,
              'c_recall': c_recall_micro,
              'c_micro_f1': c_micro_f1,
              'c_macro_f1': c_macro_f1,
              'P_acc': acc,
              'full': [p_precision_dict, p_recall_dict, p_fscore_dict, right_count_list, predicted_count_list,
                       gold_count_list,
                       c_precision_dict, c_recall_dict, c_fscore_dict, c_right_count_list, c_predicted_count_list,
                       c_gold_count_list,
                       ]}
    name = 'full'
    if not os.path.exists(name):
        os.mkdir(name)
    target_path = os.path.join(name,
                               f'data{args.dataset}-seed{args.seed}-shot{args.shot}-cs{args.constraint_loss}-ct{args.contrastive_loss}-ctl{args.contrastive_level}')
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    torch.save(epoch_predicts, os.path.join(target_path, "epoch_predicts.pt"))
    torch.save(epoch_labels, os.path.join(target_path, "epoch_labels.pt"))
    torch.save(result, os.path.join(target_path, "full_result.pt"))
    return result