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

    args_parser.add_argument('--seq2seq_save_path', default='checkpoints/seq2seq_save_model', type=str,
                             help='seq2seq_save_path')
    args_parser.add_argument('--seq2seq_load_path', default='checkpoints/seq2seq_save_model', type=str,
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
    spacy_de = spacy.load('de_core_news_sm')  # python -m spacy download en
    spacy_fr = spacy.load('fr_core_news_sm')  # python -m spacy download en

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device('cuda:3') #torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cpu' if not torch.cuda.is_available() else 'cuda:0'

    def tokenizer_en(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]
    def tokenizer_de(text):  # create a tokenizer function
        return [tok.text for tok in spacy_de.tokenizer(text)]
    def tokenizer_fr(text):  # create a tokenizer function
        return [tok.text for tok in spacy_fr.tokenizer(text)]

    en_field = data.Field(sequential=True, tokenize=tokenizer_en, lower=True, fix_length=150, include_lengths=True, batch_first=True)  #use_vocab=False
    de_field = data.Field(sequential=True, tokenize=tokenizer_de, lower=True, fix_length=150, include_lengths=True, batch_first=True)  #use_vocab=False
    fr_field = data.Field(sequential=True, tokenize=tokenizer_fr, lower=True, fix_length=150, include_lengths=True, batch_first=True)  #use_vocab=False
    print('begin loading training data-----')
    print('time: ', time.asctime( time.localtime(time.time()) ))
    seq2seq_train_data = MultiSourceTranslationDataset(
        path='wmt14_3/sample', exts=('.de', '.fr', '.en'),

        fields=(de_field, fr_field, en_field))
    print('begin loading validation data-----')
    print('time: ', time.asctime( time.localtime(time.time()) ))
    seq2seq_dev_data = MultiSourceTranslationDataset(
        path='wmt14_3/sample', exts=('.de', '.fr', '.en'),
        fields=(de_field, fr_field, en_field))
    print('end loading data-----')
    print('time: ', time.asctime( time.localtime(time.time()) ))

    de_train_data = datasets.TranslationDataset(path='wmt14_3/train', exts=('.de', '.de'), fields=(de_field, de_field))
    print('end de data add-----')
    print('time: ', time.asctime( time.localtime(time.time()) ))
    de_field.build_vocab(de_train_data, max_size=80000)  # ,vectors="glove.6B.100d"
    with open('vocab_de.pickle', 'wb') as f:
        pickle.dump(de_field.vocab, f)
    print('end de vocab save-----')
    print('time: ', time.asctime( time.localtime(time.time()) ))
    with open('vocab_de.pickle', 'rb') as f:
        de_field.vocab= pickle.load(f)
    print('end de vocab load-----')
    print('time: ', time.asctime( time.localtime(time.time()) ))

if __name__ == '__main__':
    main()