import copy
import math
import torch.nn as nn
import torch.optim

from src.masked_cross_entropy import masked_cross_entropy
from src.model_utils import copy_list
from src.model_utils import get_all_number_encoder_outputs
from src.model_utils import generate_tree_input

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score           = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack      = copy_list(node_stack)
        self.left_childs     = copy_list(left_childs)
        self.out             = copy.deepcopy(out)


# SubTree embedding
class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_tree(input_batch,       input_length,      target_batch,       target_length,
               nums_stack_batch,  num_size_batch,    generate_nums,
               encoder,           predict,           generate,           merge,
               encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
               output_lang,       num_pos,           english=False):
    # input_batch:  source sentence
    # input_batch     = [472, 38, 1290, 1031, 618, 1326, 2, 10, 411, 2159, 42, 32, 64, 47, 1, 44, 10, 1244, 2159, 42,
    #                    32, 64, 47, 1, 44, 10, 1060, 411, 2159, 173, 1244, 2159, 1233, 576, 17, 67, 660, 10, 1111,
    #                    1237, 540, 2, 22, 472, 38, 1259, 618, 99, 758, 2159, 17, 2574, 20, 36]
    # input_length    =  54
    # target_batch: target sentence
    # target_batch    =  [2, 3, 5, 5, 3, 2, 5, 7, 2, 5, 8, 0, 0, 0, 0, 0, 0]
    #                 =  ['/', '+', '1', '1', '+', '/', '1', 'N0', '/', '1', 'N1']
    #                 =  (1+1)/((1/N0)+(1/N1))
    #                 =  (1+1)/(1/3+1/6)
    # target_length   =  11

    # num_stack_batch =  []
    # num_size_batch  =  2
    # generate_nums   =  [5, 6]   = constant quantity vocab pos

    # output_lang     =  output vocab util
    # num_pos         =  [14, 23] = numeric  quantity token pos

    # sequence mask for attention
    # pad token填充为1，非pad token填充为0
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)
    # seq_mask: [batch_size, seq_len]

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    # num_mask: [batch_size, num_size + constant_size]

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor( input_batch).transpose(0, 1)
    target    = torch.LongTensor(target_batch).transpose(0, 1)
    # input_var: [seq_len, batch_size]
    # target:    [tgt_len, batch_size]

    # padding_hidden: [1, batch_size]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size     = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var      = input_var.cuda()
        seq_mask       = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask       = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    # input_var:    [seq_len, batch_size] Tensor
    # input_length: [batch_size] list
    encoder_outputs, problem_output = encoder(input_seqs=input_var,
                                              input_lengths=input_length,
                                              hidden=None)
    # encoder_outputs: [seq_len, batch_size, hidden_size]
    # problem_output:  [         batch_size, hidden_size]

    # Prepare input and output variables

    # TreeNode.embedding: ROOT GOAL VECTOR
    node_stacks = [[TreeNode(embedding=_, left_flag=False)] for _ in problem_output.split(1, dim=0)]
    # len(node_stacks): batch_size
    # node[0][0].left_flag = False

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size     = max(copy_num_len)  # num_size

    # pad token hidden_size填充为0
    # 从原始文本中指定索引的位置取出 number embedding matrix
    # NUMBER EMBEDDING MATRIX: e(y|P) = M_{num}
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs=encoder_outputs,
                                                              num_pos=num_pos,
                                                              batch_size=batch_size,
                                                              num_size=num_size,
                                                              hidden_size=encoder.hidden_size)
    # all_nums_encoder_outputs: [batch_size, num_size, hidden_size]

    num_start = output_lang.num_start
    # num_start: 5

    # embedding_stacks: token_embedding
    embeddings_stacks = [[]   for _ in range(batch_size)]
    left_childs       = [None for _ in range(batch_size)]  # debug: 如何理解这里的left_childs
    for t in range(max_target_length):
        # node_stacks(len):         [batch_size]
        # left_childs(len):         [batch_size]
        # encoder_outputs:          [seq_len,  batch_size, hidden_size]
        # all_nums_encoder_outputs: [batch_size, num_size, hidden_size]
        # padding_hidden:           [1, hidden_size]
        # seq_mask:                 [batch_size, seq_len]
        # num_mask:                 [batch_size, num_size + constant_size]
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks=node_stacks,
            left_childs=left_childs,
            encoder_outputs=encoder_outputs,
            num_pades=all_nums_encoder_outputs,
            padding_hidden=padding_hidden,
            seq_mask=seq_mask,
            mask_nums=num_mask)
        # num_score:              [batch_size, num_size + constant_size]
        # op / op_score:          [batch_size, operator_size]

        # GOAL VECTOR q
        # current_embeddings:     [batch_size, 1, hidden_size]

        # CONTEXT VECTOR c
        # current_context:        [batch_size, 1, hidden_size]

        # CURRENT NUMBER EMBEDDING MATRIX M_{num}
        # current_num_embeddings: [batch_size, num_size + constant_size, hidden_size]

        # all_leafs.append(p_leaf)
        # outputs: target分类器分数, y^
        outputs = torch.cat((op, num_score), dim=1)
        # outputs: [batch_size, operator_size + num_size + constant_size]

        all_node_outputs.append(outputs)  # prediction
        target_t, generate_input = generate_tree_input(target=target[t].tolist(),
                                                       decoder_output=outputs,
                                                       nums_stack_batch=nums_stack_batch,
                                                       num_start=num_start,  # num_start: 5
                                                       unk=unk)              # unk: 5
        # target_t:       target token index = The token with the highest probability
        # generate_input: target token index = The token with the highest probability
        # target_t:        [batch_size]
        # generate_input:  [batch_size]
        target[t] = target_t  # ground truth
        if USE_CUDA:
            generate_input = generate_input.cuda()

        # current_embeddings: [batch_size, 1, hidden_size] = goal vector q
        # generate_input:     [batch_size]                 = token embedding e(y|p)
        # current_context:    [batch_size, 1, hidden_size] = context vector c
        left_child, right_child, node_label = generate(node_embedding=current_embeddings,
                                                       node_label=generate_input,
                                                       current_context=current_context)
        # left_child:  h_l    = [batch_size, hidden_size]
        # right_child: h_r    = [batch_size, hidden_size]
        # node_label:  e(y|P) = [batch_size, embedding_size]

        left_childs = []  # debug: left_childs如何执行
        for idx, l, r, node_stack, i, o in zip(range(batch_size),
                                               left_child.split(1),
                                               right_child.split(1),
                                               node_stacks,
                                               target[t].tolist(),
                                               embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            # 生成的token为运算符
            if i < num_start:
                # 生成新的右孩子节点, r.embedding: [1, hidden_size]
                node_stack.append(TreeNode(embedding=r, left_flag=False))

                # 生成新的左孩子节点, l.embedding: [1, hidden_size]
                node_stack.append(TreeNode(embedding=l, left_flag=True))

                # 更新非叶子节点的Sub Tree embedding = 当前节点的token embedding e(y^|P)
                o.append(TreeEmbedding(embedding=node_label[idx].unsqueeze(0), terminal=False))  # terminal=False: 非叶子节点
                # sub_tree embedding = node_label: [1, embedding_size]

            # 生成的token为运算数
            else:
                # update sub tree embedding = t
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)

                # current_num: [1, hidden_size]
                while len(o) > 0 and o[-1].terminal:  # 此时为右孩子节点
                    sub_stree = o.pop()  # 左孩子节点
                    op        = o.pop()  # 父节点(操作符)

                    # 更新叶子节点的Tree embedding
                    # 如果此时为右孩子节点，则通过左孩子节点和右孩子节点的subtree embedding来更新根节点的subtree embedding

                    # op.embedding:          [1, embedding_size] = parent node token embedding = e(y^|P)
                    # sub_stree.embedding:   [1,    hidden_size] = left_sub_tree_embedding     = t_l
                    # current_num.embedding: [1,    hidden_size] = right_sub_tree_embedding    = t_r
                    current_num = merge(node_embedding=op.embedding,
                                        sub_tree_1=sub_stree.embedding,
                                        sub_tree_2=current_num)

                o.append(TreeEmbedding(embedding=current_num, terminal=True))  # terminal=True: 叶子节点
                # sub_tree embedding = current_num: [1, hidden_size]

            # left_childs记录所有的sub-tree embedding，并且在生成新节点时更新
            if len(o) > 0 and o[-1].terminal:  # 最后一个节点为叶子节点(操作数)
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    # all_node_outputs: [batch_size, tgt_len, num_size + constant_size + operator_size]

    # target: [tgt_len, batch_size]
    target = target.transpose(0, 1).contiguous()
    # target: [batch_size, tgt_len]

    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()
    # all_node_outputs: [batch_size, tgt_len, num_size + constant_size + operator_size]
    # target:           [batch_size, tgt_len]

    loss = masked_cross_entropy(logits=all_node_outputs,
                                target=target,
                                length=target_length)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()


def evaluate_tree(input_batch, input_length, generate_nums,
                  encoder,     predict,      generate,      merge,
                  output_lang, num_pos,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var      = input_var.cuda()
        seq_mask       = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask       = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_seqs=input_var,
                                              input_lengths=[input_length],
                                              hidden=None)

    # Prepare input and output variables
    node_stacks = [[TreeNode(embedding=_, left_flag=False)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    # get num feature matrix M_{num}
    # pad token hidden_size填充为0
    # NUMBER EMBEDDING MATRIX: e(y|P) = M_{num}
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs=encoder_outputs,
                                                              num_pos=[num_pos],
                                                              batch_size=batch_size,
                                                              num_size=num_size,
                                                              hidden_size=encoder.hidden_size)
    # all_nums_encoder_outputs: [batch_size, num_size, hidden_size]

    num_start = output_lang.num_start  # num_start: 5
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs     = [None for _ in range(batch_size)]

    beams = [TreeBeam(score=0.0,
                      node_stack=node_stacks,
                      embedding_stack=embeddings_stacks,
                      left_childs=left_childs,
                      out=[])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue

            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs
            # node_stacks:              [batch_size]
            # left_childs:              [batch_size]
            # encoder_outputs:          [seq_len, batch_size, hidden_size]
            # all_nums_encoder_outputs: [batch_size, num_size, hidden_size]
            # padding_hidden:           [1, hidden_size]
            # seq_mask:                 [batch_size, seq_len]
            # num_mask:                 [batch_size, num_size + constant_size]
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks=b.node_stack,
                left_childs=left_childs,
                encoder_outputs=encoder_outputs,
                num_pades=all_nums_encoder_outputs,
                padding_hidden=padding_hidden,
                seq_mask=seq_mask,
                mask_nums=num_mask)
            # num_score:              [batch_size, num_size + constant_size]
            # op / op_score:          [batch_size, operator_size]

            # GOAL VECTOR q
            # current_embeddings:     [batch_size, 1, hidden_size]

            # CONTEXT VECTOR c
            # current_context:        [batch_size, 1, hidden_size]

            # CURRENT NUMBER EMBEDDING MATRIX M_{num}
            # current_num_embeddings: [batch_size, num_size + constant_size, hidden_size]

            out_score = torch.cat((op, num_score), dim=1)
            out_score = nn.functional.log_softmax(out_score, dim=1)
            # out_score: [batch_size, num_size + constant_size + operator_size]

            topv, topi = out_score.topk(beam_size)
            # topv, topi: values, indexes

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack  = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                # 预测的token为运算符
                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()

                    # current_embeddings: goal vector q
                    # generate_input:     token embedding e(y|p)
                    # current_context:    context vector c
                    left_child, right_child, node_label = generate(node_embedding=current_embeddings,
                                                                   node_label=generate_input,
                                                                   current_context=current_context)
                    # left_child:  h_l    = [batch_size,    hidden_size]
                    # right_child: h_r    = [batch_size,    hidden_size]
                    # node_label:  e(y|P) = [batch_size, embedding_size]

                    # 生成新的右孩子节点
                    current_node_stack[0].append(TreeNode(right_child, left_flag=False))

                    # 生成新的左孩子节点
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    # 更新非叶子节点的Tree embedding
                    # node_label: [1, embedding_size]
                    current_embeddings_stacks[0].append(
                        TreeEmbedding(embedding=node_label[0].unsqueeze(0), terminal=False))  # terminal=False: 非叶子节点

                # 预测的token为运算数
                else:
                    # 更新叶子节点的Tree embedding = subTree embedding = e(y|P)
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                    # current_num: [1, hidden_size]

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op        = current_embeddings_stacks[0].pop()
                        # 更新叶子节点的Tree embedding
                        # op.embedding:       [1, embedding_size]
                        # sub_tree.embedding: [1,    hidden_size]
                        # current_num:        [1,    hidden_size]

                        # 如果此时为右孩子节点，则通过左孩子节点和右孩子节点的subtree embedding来更新根节点的subtree embedding
                        current_num = merge(node_embedding=op.embedding,
                                            sub_tree_1=sub_stree.embedding,
                                            sub_tree_2=current_num)

                    # current_num: [1, hidden_size]
                    current_embeddings_stacks[0].append(
                        TreeEmbedding(embedding=current_num, terminal=True))  # terminal=True: 叶子节点

                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)

                current_beams.append(TreeBeam(score=b.score+float(tv),
                                              node_stack=current_node_stack,
                                              embedding_stack=current_embeddings_stacks,
                                              left_childs=current_left_childs,
                                              out=current_out))

        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
