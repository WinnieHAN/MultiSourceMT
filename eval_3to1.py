from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os, math, importlib, itertools, nltk
from nltk.translate.bleu_score import SmoothingFunction

# reload is a buildin in python2. use importlib.reload in python 3

# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')   # Try setting the system default encoding as utf-8 at the start of the script, so that all strings are encoded using that. Or there will be UnicodeDecodeError: 'ascii' codec can't decode byte...

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
from seq2seq.seq2seq import Seq2seq_Model, get_bleu, get_correct
import pickle, random

import spacy, time
import torch
from torchtext import data, datasets
from multitranslation import MultiSourceTranslationDataset
import pickle
# import fairseq_cli.train

uid = uuid.uuid4().hex[:6]


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--cuda', action='store_true', help='using GPU')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--objective', choices=['cross_entropy', 'crf'], default='cross_entropy',
                             help='objective function of training procedure.')
    args_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--p_rnn', nargs=2, type=float, default=0.1, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0.,
                             help='The rate to replace a singleton word with UNK')
    #args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    # args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--model_path', help='path for saving model file.', default='models/temp')
    args_parser.add_argument('--model_name', help='name for saving model file.', default='generator')

    args_parser.add_argument('--seq2seq_save_path', default='checkpoint3to1/model', type=str,
                             help='seq2seq_save_path')
    args_parser.add_argument('--seq2seq_load_path', default='checkpoint5/model', type=str,
                             help='seq2seq_load_path')
    # args_parser.add_argument('--rl_finetune_seq2seq_save_path', default='models/rl_finetune/seq2seq_save_model',
    #                          type=str, help='rl_finetune_seq2seq_save_path')
    # args_parser.add_argument('--rl_finetune_network_save_path', default='models/rl_finetune/network_save_model',
    #                          type=str, help='rl_finetune_network_save_path')
    # args_parser.add_argument('--rl_finetune_seq2seq_load_path', default='models/rl_finetune/seq2seq_save_model',
    #                          type=str, help='rl_finetune_seq2seq_load_path')
    # args_parser.add_argument('--rl_finetune_network_load_path', default='models/rl_finetune/network_save_model',
    #                          type=str, help='rl_finetune_network_load_path')

    args_parser.add_argument('--direct_eval', action='store_true', help='direct eval without generation process')
    args = args_parser.parse_args()

    spacy_en = spacy.load('en_core_web_sm')  # python -m spacy download en

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device('cuda') #torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cpu' if not torch.cuda.is_available() else 'cuda:0'

    def tokenizer_en(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    en_field = data.Field(sequential=True, tokenize=tokenizer_en, lower=False, include_lengths=True, batch_first=True, eos_token='<eos>')  #use_vocab=False fix_length=10
    print('begin loading training data-----')
    seq2seq_train_data = MultiSourceTranslationDataset(
        path='gec_data/sample', exts=('.out1', '.out2', '.trg'),  # st19-train-20.tok

        fields=(en_field, en_field, en_field))
    print('begin loading validation data-----')
    # print('time: ', time.asctime( time.localtime(time.time()) ))
    seq2seq_dev_data = MultiSourceTranslationDataset(
        path='gec_data/wilocABCN-test.tok', exts=('.out1', '.out2', '.out3'), #wilocABCN-test.tok
        fields=(en_field, en_field, en_field))
    print('end loading data-----')

    # en_train_data = datasets.TranslationDataset(path='wmt14_3/sample', exts=('.en', '.en'), fields=(en_field, en_field))
    # print('end en data-----')
    # print('time: ', time.asctime( time.localtime(time.time()) ))
    # de_train_data = datasets.TranslationDataset(path='wmt14_3/sample', exts=('.de', '.de'), fields=(de_field, de_field))
    # fr_train_data = datasets.TranslationDataset(path='wmt14_3/sample', exts=('.fr', '.fr'), fields=(fr_field, fr_field))
    # en_field.build_vocab(en_train_data, max_size=80000)  # ,vectors="glove.6B.100d"
    # de_field.build_vocab(de_train_data, max_size=80000)  # ,vectors="glove.6B.100d"
    # fr_field.build_vocab(fr_train_data, max_size=80000)  # ,vectors="glove.6B.100d"
    # vocab_thread = 20000+2
    # with open(str(vocab_thread)+'_vocab_en.pickle', 'rb') as f:
    #     en_field.vocab = pickle.load(f)
    # with open(str(vocab_thread)+'_vocab_de.pickle', 'rb') as f:
    #     de_field.vocab = pickle.load(f)
    # with open(str(vocab_thread)+'_vocab_fr.pickle', 'rb') as f:
    #     fr_field.vocab = pickle.load(f)
    # with open('vocab_en.pickle', 'rb') as f:
    #     en_field.vocab = pickle.load(f)
    # print('end build vocab-----')
    vocab_thread = 80000+2
    with open(str(vocab_thread)+'_vocab_en_2single.pickle', 'rb') as f:
        en_field.vocab = pickle.load(f)
    print('en_field.vocab: ', len(en_field.vocab.stoi))
    # print('time: ', time.asctime( time.localtime(time.time()) ))
    # trg_field.build_vocab(seq2seq_train_data, max_size=80000)
    # mt_dev shares the fields, so it shares their vocab objects

    train_iter = data.BucketIterator(
        dataset=seq2seq_train_data, batch_size=16,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device, shuffle=True)  # Note that if you are runing on CPU, you must set device to be -1, otherwise you can leave it to 0 for GPU.
    dev_iter = data.BucketIterator(
        dataset=seq2seq_dev_data, batch_size=16,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device, shuffle=False)

    num_words_en = len(en_field.vocab.stoi)
    # Pretrain seq2seq model using denoising autoencoder. model name: seq2seq model
    
    EPOCHS = 100  # 150
    DECAY = 0.97
    # TODO: #len(en_field.vocab.stoi)  # ?? word_embedd ??
    word_dim = 300  # ??
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size1=len(en_field.vocab.stoi), vocab_size2=len(en_field.vocab.stoi), vocab_size3=len(en_field.vocab.stoi), word_embedd=None, device=device, share_emb=True).to(device)  # TODO: random init vocab
    # seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=0.0003)
    seq2seq.load_state_dict(torch.load('checkpoints5/seq2seq_save_model_batch_1500001.pt'))
    #seq2seq.load_state_dict(torch.load(args.seq2seq_load_path +'_batch_'+ str(1125001) + '.pt'))  # TODO: 10.7
    # torch.save(seq2seq.state_dict(), args.seq2seq_save_path +'_batch_'+ str(ii) + '.pt') checkpoint5/seq2seq_save_model_batch_1500001.pt
    seq2seq.to(device)

    def count_parameters(model: torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(seq2seq):,} trainable parameters')
    PAD_IDX = en_field.vocab.stoi['<pad>']
    EOS_IDX = en_field.vocab.stoi['<eos>']
    # criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if True:  # i%1 == 0:
        wf = open('test.predict.out', 'wb')
        seq2seq.eval()
        bleu_ep = 0
        acc_numerator_ep = 0
        acc_denominator_ep = 0
        testi = 0
        for _, batch in enumerate(dev_iter):  # for _ in range(1, num_batches + 1):  word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_dev, batch_size, unk_replace=unk_replace)  # word:(32,50)  char:(32,50,35)
            src1, lengths_src1 = batch.src1  # word:(32,50)  150,64
            src2, lengths_src2 = batch.src2  # word:(32,50)  150,64
            trg, lengths_trg = batch.trg
            sel, _ = seq2seq(src1.long().to(device), src2.long().to(device), LEN=5+max(src1.size()[1], src2.size()[1]))  # TODO:
            sel = sel.detach().cpu().numpy()
            dec_out = trg.cpu().numpy()

            bleus = []


            for j in range(sel.shape[0]):
                bleu = get_bleu(sel[j], dec_out[j], EOS_IDX)  # sel
                bleus.append(bleu)
                numerator, denominator = get_correct(sel[j], dec_out[j], EOS_IDX)
                sel_idxs = sel[j]
                sel_words = idx_to_words(sel_idxs, EOS_IDX, PAD_IDX, en_field.vocab.itos)
                line = ' '.join(sel_words) + '\n'
                wf.write(line.encode('utf-8'))
                acc_numerator_ep += numerator
                acc_denominator_ep += denominator  # .detach().cpu().numpy() TODO: 10.8
            bleu_bh = np.average(bleus)
            bleu_ep += bleu_bh
            testi += 1
        bleu_ep /= testi  # num_batches
        print('testi: ', testi)
        print('Valid bleu: %.4f%%' % (bleu_ep * 100))
        # print(acc_denominator_ep)
        if acc_denominator_ep > 0:
            print('Valid acc: %.4f%%' % ((acc_numerator_ep * 1.0 / acc_denominator_ep) * 100))
        wf.close()

def idx_to_words(out, EOS_IDX, PAD_IDX, itos):
    out = out.tolist()

    stop_token = PAD_IDX
    if stop_token in out:
        out = out[:out.index(stop_token)]
    else:
        out = out

    stop_token = EOS_IDX
    if stop_token in out:
        cnd = out[:out.index(stop_token)]
    else:
        cnd = out
    cnd = [itos[ii] for ii in cnd]
    return cnd

if __name__ == '__main__':
    main()