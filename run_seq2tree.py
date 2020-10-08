# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)
# 文本中出现的constant quantity(常数)
# generate_nums: ['1', '3.14']

temp_pairs = []
# p[0] = input_seq
# p[1] = out_seq
# p[2] = equation(prefix)
# p[3] = nums
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

# 测试集，分为五折
fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end   = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []
# 5折数据集
for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained=pairs_trained,
                                                                    pairs_tested=pairs_tested,
                                                                    trim_min_count=5,
                                                                    generate_nums=generate_nums,
                                                                    copy_nums=copy_nums,
                                                                    tree=True)
    # output_lang.index2word: ['*', '-', '/', '+', '^',
    #                          '1', '3.14',
    #                          'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7',
    #                          'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14',
    #                          'UNK']
    # output_lang.word2index: {'*': 0, '-': 1, '/': 2, '+': 3, '^': 4,
    #                          '1': 5, '3.14': 6,
    #                          'N0': 7, 'N1': 8, 'N2': 9, 'N3': 10, 'N4': 11, 'N5': 12, 'N6': 13, 'N7': 14,
    #                          'N8': 15, 'N9': 16, 'N10': 17, 'N11': 18, 'N12': 19, 'N13': 20, 'N14': 21,
    #                          'UNK': 22}

    # Initialize models
    encoder  = EncoderSeq(  input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)
    predict  = Prediction(  hidden_size=hidden_size,       op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size,       op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), embedding_size=embedding_size)
    merge    = Merge(       hidden_size=hidden_size,       embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder_optimizer  = torch.optim.Adam(encoder.parameters(),  lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer  = torch.optim.Adam(predict.parameters(),  lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer    = torch.optim.Adam(merge.parameters(),    lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler  = torch.optim.lr_scheduler.StepLR(encoder_optimizer,  step_size=20, gamma=0.5)
    predict_scheduler  = torch.optim.lr_scheduler.StepLR(predict_optimizer,  step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler    = torch.optim.lr_scheduler.StepLR(merge_optimizer,    step_size=20, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    # generate_num_ids: [5, 6]
    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    for epoch in range(n_epochs):
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, \
        nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        start = time.time()
        for idx in range(len(input_lengths)):
            loss = train_tree(
                input_batch=input_batches[idx],
                input_length=input_lengths[idx],
                target_batch=output_batches[idx],
                target_length=output_lengths[idx],
                nums_stack_batch=num_stack_batches[idx],
                num_size_batch=num_size_batches[idx],
                generate_nums=generate_num_ids,
                encoder=encoder,
                predict=predict,
                generate=generate,
                merge=merge,
                encoder_optimizer=encoder_optimizer,
                predict_optimizer=predict_optimizer,
                generate_optimizer=generate_optimizer,
                merge_optimizer=merge_optimizer,
                output_lang=output_lang,
                num_pos=num_pos_batches[idx])
            loss_total += loss

        print("loss:", loss_total / len(input_lengths))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        if epoch % 10 == 0 or epoch > n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            for test_batch in test_pairs:
                test_res = evaluate_tree(input_batch=test_batch[0],
                                         input_length=test_batch[1],
                                         generate_nums=generate_num_ids,
                                         encoder=encoder,
                                         predict=predict,
                                         generate=generate,
                                         merge=merge,
                                         output_lang=output_lang,
                                         num_pos=test_batch[5],
                                         beam_size=beam_size)

                val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res=test_res,
                                                                  test_tar=test_batch[2],
                                                                  output_lang=output_lang,
                                                                  num_list=test_batch[4],
                                                                  num_stack=test_batch[6])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            torch.save(encoder.state_dict(),  "models/encoder")
            torch.save(predict.state_dict(),  "models/predict")
            torch.save(generate.state_dict(), "models/generate")
            torch.save(merge.state_dict(),    "models/merge")
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))
