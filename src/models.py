import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


# 预测数字的概率
class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H

        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn  = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        # hidden: context vector q
        # encoder_outputs: final hidden state
        # hidden:           [      1, batch_size, hidden_size]
        # encoder_outputs:  [seq_len, batch_size, hidden_size]
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        # hidden: [seq_len, batch_size, hidden_size]
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        # energy_in: [seq_len,  batch_size, 2*hidden_size]
        # energy_in: [seq_len * batch_size, 2*hidden_size]
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        # score_feature: [seq_len * batch_size, 2*hidden_size]

        attn_energies = self.score(score_feature)  # (S x B) x 1
        # attn_energies: [seq_len * batch_size, 1]

        attn_energies = attn_energies.squeeze(1)
        # attn_energies: [seq_len * batch_size]

        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        # attn_energies: [batch_size, seq_len]

        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S
        # attn_energies: [batch_size, seq_len]

        # attn_energies: [batch_size, 1, seq_len]
        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_layers       = n_layers
        self.dropout        = dropout

        self.embedding = nn.Embedding(num_embeddings=input_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers,
                               dropout=dropout,
                               bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # embedded: [seq_len, batch_size, embedding_size]
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)

        # pade_outputs: [seq_len, batch_size, n_direction * hidden_size]
        # pade_hidden:  [n_layer * n_direction, batch_size, hidden_size]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        # problem_output: [batch_size, hidden_size]
        # pade_output:    [seq_len, batch_size, hidden_size]

        # final hidden state
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs   = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H

        return pade_outputs, problem_output


# Top-down Goal Decomposition
class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size  # 512
        self.input_size  = input_size   # 2 (constant size)
        self.op_nums     = op_nums      # 5 (operator nums)

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l  = nn.Linear(hidden_size,     hidden_size)
        self.concat_r  = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size,     hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn  = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self,
                node_stacks,
                left_childs,
                encoder_outputs,
                num_pades,
                padding_hidden,
                seq_mask,
                mask_nums):

        current_embeddings = []
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:  # left sub-tree embedding is None
                c = self.dropout(c)                   # context vector c
                g = torch.tanh(self.concat_l(c))      # goal vector g        = Q_{le}
                t = torch.sigmoid(self.concat_lg(c))  # token embedding t    = g_l
                current_node_temp.append(g * t)       # left sub-goal vector = q_l
            else:
                ld = self.dropout(l)                                      # sub-tree left tree emb = ld
                c = self.dropout(c)                                       # context vector c
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))      # goal vector g         = Q_{re}
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))  # token embedding t     = g_{r}
                current_node_temp.append(g * t)                           # right sub-goal vector = q_{r}

        current_node = torch.stack(current_node_temp)
        # current node: goal vector q
        # current_node: [batch_size, 1, hidden_size]

        current_embeddings = self.dropout(current_node)
        # current_embeddings: context vector c
        # current_embeddings: [batch_size, 1, hidden_size]

        # current_embeddings: goal vector q
        # encoder_outputs:    final hidden state h_{sp}
        # current_embeddings: [      1, batch_size, hidden_size]
        # encoder_outputs:    [seq_len, batch_size, hidden_size]
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        # current_attn:  a_{s}
        # current_attn: [batch_size, 1, seq_len]

        # CONTEXT VECTOR c: summarizes relevant information of the problem at hand
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        # current_context: context vector c
        # current_context: [batch_size,       1, hidden_size]

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N
        # embedding_weight: [batch_size, num_size, hidden_size]

        leaf_input = torch.cat((current_node, current_context), 2)
        # leaf_input: [batch_size, 1, 2*hidden_size]
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)
        # leaf_input: [batch_size, 2*hidden_size]

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        # leaf_input: [1, batch_size, 2*hidden_size]
        # embedding_weight_: [batch_size, num_size+constant_size, hidden_size]
        # mask_nums:         [batch_size, num_size+constant_size]
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        # num_score: [batch_size, num_size]

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)
        # op: [batch_size, op_num]

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight
        # current_node:      goal vector q
        # current_context:   context vector c
        # embedding_weight:  current num embedding matrix


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # op_nums: 操作符数量
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        # node_label:  [batch_size]
        node_label_ = self.embeddings(node_label)
        # node_label_: [batch_size, embedding_size]
        node_label  = self.em_dropout(node_label_)
        # node_label:  [batch_size, embedding_size]

        node_embedding  = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding  = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        # left sub-goal generation
        # node_embedding:  q
        # current_context: c
        # node_label:      e(y|P)
        l_child   = torch.tanh(   self.generate_l( torch.cat((node_embedding, current_context, node_label), 1)))  # C_l
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))  # o_l
        l_child = l_child * l_child_g  # h_l
        # l_child:   C_l
        # l_child_g: o_l
        # l_child:   h_l

        # right sub-goal generation
        # node_embedding:  goal vector q
        # current_context: context vector c
        # node_label: e(y|P)
        r_child   = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))     # C_r
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1))) # o_r
        r_child = r_child * r_child_g  # h_r
        # r_child:   C_r
        # r_child_g: o_r
        # r_child:   h_r

        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge   = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        # node_embedding: e(y|P)
        # sub_tree_1: t_{l}
        # sub_tree_2: t_{r}
        # node_embedding: [1, embedding_size]
        # sub_tree_1:     [1, hidden_size]
        # sub_tree_2:     [1, hidden_size]
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree   = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))       # C_t
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))  # g_t
        sub_tree = sub_tree * sub_tree_g  # t_comb
        return sub_tree
