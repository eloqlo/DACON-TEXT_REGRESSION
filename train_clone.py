import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TextclassificationDataset

import model.Constants as Constants
from model.Models import BERT_like
# from model.Optim import ScheduledOptim

# not yet
def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

# label smoothing ?
# Tensor.scatter() ?
def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return 

# ?
def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src

# ?
def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    """Epoch operation in training phase"""
    
    model.train()
    total_loss, n_word_total, n_word_correct = 0,0,0
    
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininteraval=2, desc=desc, leave=False):
        
        # prepare data
        
        # forward
        
        # backward and update parameters
        
        # note keeping
        pass
    pass

def eval_epoch(model, validation_data, device, opt):
    """Epoch operation in evaluation phase"""
    pass

def train(model, training_data, validation_data, optimizer, device, opt):
    """Start Training"""
    pass

def main():
    
    # some parser things
    
    # some settings here
    
    # some data laodings
    
    # some model generation
    
    # some optimizer generation
    
    pass

def prepare_dataloaders_from_bpe_files(opt, device):
    pass

def prepare_dataloaders(opt, device):
    pass


if __name__ = '__main__':
    main()