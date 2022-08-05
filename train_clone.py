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

# training_data 누가 들어오길래, .src, .trg 등을 할 수 있는거야?
# 어떻게 처리되길래, batch 나 epoch 언급없이 학습이 되는거지? 한 epoch에 대한건가?
def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    """Epoch operation in training phase"""
    
    model.train()
    total_loss, n_word_total, n_word_correct = 0,0,0
    
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininteraval=2, desc=desc, leave=False):
        
        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))
        
        # forward
        optimizer.zero_grad()
        pred= model(src_seq, trg_seq)
        
        # backward and update parameters
        loss, n_correct, n_word = cal_performance( pred, gold, opt.trg_pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()
        
        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    
def eval_epoch(model, validation_data, device, opt):
    """Epoch operation in evaluation phase"""
    pass


# opt 는 어디서 오는거지?
# training_data 어떤 객체지?
def train(model, training_data, validation_data, optimizer, device, opt):
    """Start Training"""
    
    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))
        
    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')
    
    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file
    ))
    
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')
    
    # format 문법 보기
    def print_performances(header, ppl, accu, start_time, lr):  # 'Training',  train_ppl, train_accu, start, lr
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, elapse: {elapse:3.3f} min'
             .format(header=f"({header})", 
                            ppl=ppl,accu=100*accu, 
                            elapse=(time.time()-start_time)/60, 
                            lr=lr))
    
    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        
        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        
        
        pass
    
    
    

def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''
    
    # some parser things
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-data_pkl', default=None)      # all-in-1 data pickle or bpe field
    
    parser.add_argument('-train_path', default=None)
    parser.add_argument('-val_path', default=None)
    
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    
    pass
    
    # some settings here
    
    # some data laodings
    
    # some model generation
    
    # some optimizer generation
    
    pass

def prepare_dataloaders_from_bpe_files(opt, device):
    pass

def prepare_dataloaders(opt, device):
    pass


if __name__ == '__main__':
    main()