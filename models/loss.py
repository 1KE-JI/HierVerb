import torch
import torch.nn.functional as F

flag_imbalanced_contrastive_loss = False
flag_imbalanced_weight_reverse = False
flag_print_loss_weight = False


def sim(x, y):
    norm_x = F.normalize(x, dim=-1)
    norm_y = F.normalize(y, dim=-1)
    return torch.matmul(norm_x, norm_y.transpose(1, 0))


# flat contrastive_loss
def flat_contrastive_loss_func(hier_labels, processor, output_at_mask, imbalanced_weight=False, depth=2,
                                             contrastive_level=1,
                                             imbalanced_weight_reverse=True, use_cuda=True):
    global flag_imbalanced_contrastive_loss, flag_imbalanced_weight_reverse, flag_print_loss_weight
    ## output_at_mask = [batch_size, multi_mask, 768]
    if use_cuda:
        output_at_mask = output_at_mask.cuda()
    cur_batch_size = output_at_mask.shape[0]
    assert cur_batch_size == len(hier_labels[0])

    loss_ins = 0
    # 不同层的权重，默认同等地位
    if not imbalanced_weight:
        loss_weight = [1 for i in range(depth)]
    else:
        if not flag_imbalanced_contrastive_loss:
            print(f"using imbalanced contrastive loss with contrastive_level:{contrastive_level}")
            flag_imbalanced_contrastive_loss = True
        loss_weight = [1 / 2 ** (i * contrastive_level) for i in range(depth)]

    if imbalanced_weight_reverse:
        if not flag_imbalanced_weight_reverse:
            print("imbalanced weight reversed ")
            flag_imbalanced_weight_reverse = True

        loss_weight.reverse()
    if not flag_print_loss_weight:
        print("loss_weight:", loss_weight)
        flag_print_loss_weight = True

    for mask_idx in range(depth):
        # shape: [batch_size, 768]
        cur_output_at_mask = output_at_mask[:, mask_idx, :]
        sim_score = sim(cur_output_at_mask, cur_output_at_mask)
        sim_score = torch.exp(sim_score)
        cur_loss_weight = loss_weight[depth - 1 - mask_idx:]

        for cur_depth in range(mask_idx):

            cur_loss_ins = 0

            cur_hier_matrix = torch.zeros(cur_batch_size, cur_batch_size)

            cur_hier_labels = hier_labels[cur_depth]

            for i in range(len(cur_hier_labels)):
                tmp = cur_hier_labels[i]
                for j in range(len(cur_hier_labels)):
                    if isinstance(tmp, list):
                        flag = False
                        if len(set(tmp).intersection(set(cur_hier_labels[j]))) != 0:
                            flag = True
                        if flag:
                            cur_hier_matrix[i][j] = 1
                        else:
                            cur_hier_matrix[i][j] = 0
                    else:
                        if cur_hier_labels[j] == tmp:
                            cur_hier_matrix[i][j] = 1
                        else:
                            cur_hier_matrix[i][j] = 0

            for i in range(len(cur_hier_matrix)):
                y_true = cur_hier_matrix[i]
                # 如果当前instance A 在当前层级与其他所有instance的label_matrix矩阵值全为0，
                # 那么认为此instance A的gold label路径没有延伸至当前depth，故不在当前depth计算对比学习损失
                if len(y_true[y_true != 0]) == 0:
                    continue
                # sim.shape = [batch_size, batch_size]
                cur_sim_score = sim_score[i]
                # sim_score = sim_score - torch.eye(cur_batch_size).cuda() * 1e12
                pos_sim = cur_sim_score[y_true != 0].sum()
                neg_sim = cur_sim_score[y_true == 0].sum()
                cur_loss_ins += - torch.log(pos_sim / (pos_sim + neg_sim))

            loss_ins += cur_loss_ins * cur_loss_weight[cur_depth]

    loss_ins = loss_ins / (cur_batch_size ** 2)

    return loss_ins


def constraint_multi_depth_loss_func(logits, loss_func, hier_labels, processor, args, use_cuda=True, mode=0):
    if isinstance(logits, list):
        leaf_logits = logits[-1]
    elif isinstance(logits, torch.Tensor):
        leaf_logits = logits[:, -1, :]
    contrastive_level = 0
    hier_mapping = processor.hier_mapping
    flat_slot2value = processor.flat_slot2value
    # batch_size * label_size(134)
    depth = len(hier_mapping) + 1

    loss_weight = [1 / 2 ** (i * contrastive_level) for i in range(depth - 1)]

    leaf_logits = torch.softmax(leaf_logits, dim=-1)
    hier_logits = []
    hier_logits.insert(0, leaf_logits)

    batch_s = leaf_logits.shape[0]
    constraint_loss = 0

    all_length = len(processor.all_labels)
    for depth_idx in range(depth - 2, -1, -1):
        if isinstance(logits, list):
            ori_logits = logits[depth_idx]
        elif isinstance(logits, torch.Tensor):
            ori_logits = logits[:, depth_idx, :]
        ## True
        if args.multi_verb:
            cur_logits = torch.zeros(batch_s, len(processor.label_list[depth_idx]))

            for i in range(cur_logits.shape[-1]):
                # sum
                cur_logits[:, i] = torch.sum(hier_logits[0][:, list(hier_mapping[depth_idx][0][i])], dim=-1)
                # mean
                # cur_logits[:, i] = torch.mean(hier_logits[0][:, list(hier_mapping[depth_idx][0][i])], dim=-1)
        else:
            cur_logits = torch.zeros(batch_s, all_length)
            cd_labels = processor.depth2label[depth_idx]
            for i in range(all_length):
                if i in cd_labels:
                    cur_logits[:, i] = torch.sum(hier_logits[0][:, list(flat_slot2value[i])], dim=-1)
            # ver.weight.shape  [7+ 64 + 140, 768]
        cur_labels = hier_labels[depth_idx]

        if use_cuda:
            cur_logits = cur_logits.cuda()
            cur_labels = cur_labels.cuda()
        # default mode = 0
        if mode:
            cur_logits = cur_logits + ori_logits

        if args.multi_label:
            cur_multi_label = torch.zeros_like(cur_logits)
            for i in range(cur_multi_label.shape[0]):
                cur_multi_label[i][cur_labels[i]] = 1
            cur_labels = cur_multi_label

            # cur_logits = torch.softmax(cur_logits, dim=-1)
        hier_logits.insert(0, cur_logits)
        cur_constraint_loss = loss_func(cur_logits, cur_labels)
        constraint_loss += cur_constraint_loss * loss_weight[depth_idx]
    return constraint_loss


def multi_path_constraint_multi_depth_loss_func(logits, loss_func, hier_labels, processor, args, use_cuda=True, mode=0):
    contrastive_level = 0
    hier_mapping = processor.hier_mapping

    depth = len(hier_mapping) + 1

    loss_weight = [1 / 2 ** (i * contrastive_level) for i in range(depth - 1)]

    batch_s = logits[0].shape[0]
    constraint_loss = 0

    for depth_idx in range(depth - 2, -1, -1):
        if isinstance(logits, list):
            pre_logits = logits[depth_idx+1]
            ori_logits = logits[depth_idx]
        elif isinstance(logits, torch.Tensor):
            pre_logits = logits[:, depth_idx+1, :]
            ori_logits = logits[:, depth_idx, :]
        else:
            print(type(logits))
            raise TypeError
        cur_logits = torch.zeros(batch_s, len(processor.label_list[depth_idx]))

        for i in range(cur_logits.shape[-1]):
            ## sum
            # cur_logits[:, i] = torch.sum(hier_logits[0][:, list(hier_mapping[depth_idx][0][i])], dim=-1)
            ## mean
            if len(hier_mapping[depth_idx][0][i]) != 0:
                # ori_logits[:, i] = torch.mean(pre_logits[:, list(hier_mapping[depth_idx][0][i])], dim=-1)
                # ori_logits[:, i] = torch.mean(ori_logits[:, i] + torch.mean(pre_logits[:, list(hier_mapping[depth_idx][0][i])], dim=-1), dim=-1)
                ori_logits[:, i] = ori_logits[:, i] * 0.99 + torch.mean(pre_logits[:, list(hier_mapping[depth_idx][0][i])], dim=-1) * 0.01

        cur_labels = hier_labels[depth_idx]

        if use_cuda:
            ori_logits = ori_logits.cuda()

        cur_multi_label = torch.zeros_like(cur_logits).to("cuda:0")
        for i in range(cur_multi_label.shape[0]):
            for j in cur_labels[i]:
                cur_multi_label[i][j] = 1
        cur_labels = cur_multi_label

        cur_constraint_loss = loss_func(ori_logits, cur_labels)
        constraint_loss += cur_constraint_loss * loss_weight[depth_idx]
    return constraint_loss