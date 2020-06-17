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
from seq2seq.single_seq2seq import Single_Seq2seq_Model
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
    # args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    # args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--model_path', help='path for saving model file.', default='models/temp')
    args_parser.add_argument('--model_name', help='name for saving model file.', default='generator')

    args_parser.add_argument('--seq2seq_save_path', default='checkpoints/seq2seq_save_model', type=str,
                             help='seq2seq_save_path')
    args_parser.add_argument('--seq2seq_load_path', default='checkpoints/seq2seq_save_model', type=str,
                             help='seq2seq_load_path')

    args_parser.add_argument('--direct_eval', action='store_true', help='direct eval without generation process')
    args_parser.add_argument('--single_seq2seq', action='store_true', help='1to1 or 2to1')
    args = args_parser.parse_args()

    spacy_en = spacy.load('en_core_web_sm')  # python -m spacy download en
    spacy_de = spacy.load('de_core_news_sm')  # python -m spacy download en
    spacy_fr = spacy.load('fr_core_news_sm')  # python -m spacy download en

    SEED = random.randint(1,100000)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device(
        'cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cpu' if not torch.cuda.is_available() else 'cuda:0'

    def tokenizer_en(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenizer_de(text):  # create a tokenizer function
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenizer_fr(text):  # create a tokenizer function
        return [tok.text for tok in spacy_fr.tokenizer(text)]

    en_field = data.Field(sequential=True, tokenize=tokenizer_en, lower=True, include_lengths=True,
                          batch_first=True)  # use_vocab=False fix_length=10
    de_field = data.Field(sequential=True, tokenize=tokenizer_de, lower=True, include_lengths=True,
                          batch_first=True)  # use_vocab=False
    fr_field = data.Field(sequential=True, tokenize=tokenizer_fr, lower=True, include_lengths=True,
                          batch_first=True)  # use_vocab=False
    print('begin loading training data-----')
    print('time: ', time.asctime(time.localtime(time.time())))
    seq2seq_train_data = MultiSourceTranslationDataset(
        path='wmt14_3/train', exts=('.de', '.fr', '.en'),
        fields=(de_field, fr_field, en_field))
    print('begin loading validation data-----')
    print('time: ', time.asctime(time.localtime(time.time())))
    seq2seq_dev_data = MultiSourceTranslationDataset(
        path='wmt14_3/valid', exts=('.de', '.fr', '.en'),
        fields=(de_field, fr_field, en_field))
    print('end loading data-----')
    print('time: ', time.asctime(time.localtime(time.time())))

    # vocab_thread = 20000 + 2
    # with open(str(vocab_thread) + '_vocab_en.pickle', 'rb') as f:
    #     en_field.vocab = pickle.load(f)
    # with open(str(vocab_thread) + '_vocab_de.pickle', 'rb') as f:
    #     de_field.vocab = pickle.load(f)
    # with open(str(vocab_thread) + '_vocab_fr.pickle', 'rb') as f:
    #     fr_field.vocab = pickle.load(f)
    # print('end build vocab-----')
    # print('time: ', time.asctime(time.localtime(time.time())))

    en_train_data = datasets.TranslationDataset(path='wmt14_3/train', exts=('.en', '.en'), fields=(en_field, en_field))
    print('end en data-----')
    print('time: ', time.asctime( time.localtime(time.time()) ))
    de_train_data = datasets.TranslationDataset(path='wmt14_3/train', exts=('.de', '.de'), fields=(de_field, de_field))
    fr_train_data = datasets.TranslationDataset(path='wmt14_3/train', exts=('.fr', '.fr'), fields=(fr_field, fr_field))
    en_field.build_vocab(en_train_data, max_size=80000)  # ,vectors="glove.6B.100d"
    de_field.build_vocab(de_train_data, max_size=80000)  # ,vectors="glove.6B.100d"
    fr_field.build_vocab(fr_train_data, max_size=80000)  # ,vectors="glove.6B.100d"

    train_iter = data.BucketIterator(
        dataset=seq2seq_train_data, batch_size=16,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device,
        shuffle=True)  # Note that if you are runing on CPU, you must set device to be -1, otherwise you can leave it to 0 for GPU.
    dev_iter = data.BucketIterator(
        dataset=seq2seq_dev_data, batch_size=16,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device, shuffle=False)

    num_words_en = len(en_field.vocab.stoi)
    # Pretrain seq2seq model using denoising autoencoder. model name: seq2seq model

    EPOCHS = 150  # 150
    DECAY = 0.97
    # TODO: #len(en_field.vocab.stoi)  # ?? word_embedd ??
    word_dim = 10#300  # ??
    if args.single_seq2seq:
        seq2seq = Single_Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size1=len(de_field.vocab.stoi), vocab_size2=len(fr_field.vocab.stoi), vocab_size3=len(en_field.vocab.stoi), word_embedd=None, device=device).to(device)  # TODO: random init vocab
    else:
        seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size1=len(de_field.vocab.stoi), vocab_size2=len(fr_field.vocab.stoi), vocab_size3=len(en_field.vocab.stoi), word_embedd=None, device=device).to(device)  # TODO: random init vocab

    # seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=0.0003)

    # seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(2) + '.pt'))  # TODO: 10.7
    seq2seq.to(device)

    def count_parameters(model: torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(seq2seq):,} trainable parameters')
    PAD_IDX = en_field.vocab.stoi['<pad>']
    # criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    for i in range(EPOCHS):
        ls_seq2seq_ep = 0
        seq2seq.train()
        # seq2seq.emb.weight.requires_grad = False
        print('----------' + str(i) + ' iter----------')
        for _, batch in enumerate(train_iter):
            src1, lengths_src1 = batch.src1  # word:(32,50)  150,64
            src2, lengths_src2 = batch.src2  # word:(32,50)  150,64
            trg, lengths_trg = batch.trg

            # max_len1 = src1.size()[1]  # batch_first
            # masks1 = torch.arange(max_len1).expand(len(lengths_src1), max_len1) < lengths_src1.unsqueeze(1)
            # masks1 = masks1.long()
            # max_len2 = src2.size()[1]  # batch_first
            # masks2 = torch.arange(max_len2).expand(len(lengths_src2), max_len2) < lengths_src2.unsqueeze(1)
            # masks2 = masks2.long()
            dec_out = trg
            start_list = torch.ones((trg.shape[0], 1)).long().to(device)
            dec_inp = torch.cat((start_list, trg[:, 0:-1]), dim=1)  # maybe wrong
            # train_seq2seq
            if args.single_seq2seq:
                out = seq2seq(src1.long().to(device), is_tr=True, dec_inp=dec_inp.long().to(device))
            else:
                out = seq2seq(src1.long().to(device), src2.long().to(device), is_tr=True, dec_inp=dec_inp.long().to(device))

            out = out.view((out.shape[0] * out.shape[1], out.shape[2]))
            dec_out = dec_out.view((dec_out.shape[0] * dec_out.shape[1],))

            # max_len_trg = trg.size()[1]  # batch_first
            # masks_trg = torch.arange(max_len_trg).expand(len(lengths_trg), max_len_trg) < lengths_trg.unsqueeze(1)
            # masks_trg = masks_trg.float().to(device)
            # wgt = masks_trg.view(-1)
            # wgt = seq2seq.add_stop_token(masks, lengths_src)  # TODO
            # wgt = wgt.view((wgt.shape[0] * wgt.shape[1],)).float().to(device)
            # wgt = masks.view(-1)

            ls_seq2seq_bh = loss_seq2seq(out, dec_out.long().to(device))  # 9600, 8133
            # ls_seq2seq_bh = (ls_seq2seq_bh * wgt).sum() / wgt.sum()  # TODO
            ls_seq2seq_bh = ls_seq2seq_bh.sum() / ls_seq2seq_bh.numel()

            optim_seq2seq.zero_grad()
            ls_seq2seq_bh.backward()
            optim_seq2seq.step()

            ls_seq2seq_bh = ls_seq2seq_bh.cpu().detach().numpy()
            ls_seq2seq_ep += ls_seq2seq_bh
        print('ls_seq2seq_ep: ', ls_seq2seq_ep)
        for pg in optim_seq2seq.param_groups:
            pg['lr'] *= DECAY

        # test th bleu of seq2seq
        if i > 40:
            print('ss')
        if i > 0:  # i%1 == 0:
            seq2seq.eval()
            bleu_ep = 0
            acc_numerator_ep = 0
            acc_denominator_ep = 0
            testi = 0
            for _, batch in enumerate(
                    train_iter):  # for _ in range(1, num_batches + 1):  word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_dev, batch_size, unk_replace=unk_replace)  # word:(32,50)  char:(32,50,35)
                src1, lengths_src1 = batch.src1  # word:(32,50)  150,64
                src2, lengths_src2 = batch.src2  # word:(32,50)  150,64
                trg, lengths_trg = batch.trg
                if args.single_seq2seq:
                    sel, _ = seq2seq(src1.long().to(device),
                                     LEN=src1.size()[1]+5)  # TODO:
                else:
                    sel, _ = seq2seq(src1.long().to(device), src2.long().to(device),
                                     LEN=max(src1.size()[1], src2.size()[1]))  # TODO:
                sel = sel.detach().cpu().numpy()
                dec_out = trg.cpu().numpy()

                bleus = []

                for j in range(sel.shape[0]):
                    bleu = get_bleu(sel[j], dec_out[j], PAD_IDX)  # sel
                    bleus.append(bleu)
                    numerator, denominator = get_correct(sel[j], dec_out[j], PAD_IDX)
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
        # for debug TODO:
        if i % 5 == 0:
            torch.save(seq2seq.state_dict(), args.seq2seq_save_path + str(i) + '.pt')


if __name__ == '__main__':
    main()