import argparse
import math
import time
from zlib import DEF_MEM_LEVEL
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

# ==================================================================#
# ! all of these are deprecated libraries. Think I should try with more general DataLoader 
from torchtext.legacy.data import Dataset, BucketIterator, Field    # * 작성자가 이전버전의 torchtext 사용중이라서 legacy 카테고리의 것들 사용해야한다.
from torchtext.legacy.datasets import TranslationDataset
# ==================================================================#

import model.Constants as Constants
from model.Models import BERT_like, Transformer
from model.Optim import ScheduledOptim
# from model.Optim import ScheduledOptim

# metric?
# not yet, label smoothing?
def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

# loss?
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
    for batch in tqdm(training_data, mininteraval=2, desc=desc, leave=False):   # ? dataloader을 iteration 시키면 그 나오는 값들 형식이 궁금하다. 어떻게 이용하라고 줄까?
        
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


# TODO
def eval_epoch(model, validation_data, device, opt):
    """Epoch operation in evaluation phase"""
    pass


# opt 는 어디서 오는거지?
# training_data 어떤 객체지?
def train(model, training_data, validation_data, optimizer, device, opt):
    """Start Training"""
    # ? opt 는 뭐지
    
    
    # Tensorboard
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))
    
    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')
    
    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file
    ))
    
    # write column names
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
    
    valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        
        # TRAIN
        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))  # ? 저 ppl은 뭐지?
        # Current lr
        lr = optimizer._optimizer.param_groups[0]['lr'] # ? optimizer 내부에 _optimizer/param_groups[0] 에는 뭐가있는거야?
        print_performances('Training', train_ppl, train_accu, start, lr)
        
        # VALIDATION
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)      # TODO
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, start, lr)
        
        valid_losses += [valid_loss]    # * 제일 valid loss 작은 모델 선별위한 비교 set
        
        checkpoint = {'eopch': epoch_i, 'settings': opt, 'model': model.state_dict()}   # ? model.state_dict() 는 무슨 결과를 주지?
        
        # save_mode 옵션따라 모델 저장 처리.
        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)  # ? 동작
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses): # TODO <= 가 아니라 == 도 되겠다.
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('   - [Info] The checkpoint file has been updated to lower valid_loss model !')
        
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:  # * 'a' : 존재하는 파일에 추가하는 모드
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss, ppl=train_ppl, accu=100*train_accu
            ))            
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss, ppl=valid_ppl, accu=100*valid_accu
            ))            
        
        # TODO 텐서보드 API 익히기.
        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val':valid_accu*100}, epoch_i)
            tb_writer.add_scalars('learning_rate', lr, epoch_i)
    
    

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
    
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps',type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)
    
    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true') # * True 를 저장한다
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')    # * choices 컨테이너에 포함되는지 type 변환이 수행한 후 검사
    
    parser.add_argument('-cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    
    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False # ?
        np.random.seed(opt.seed)
        random.seed(opt.seed)
    
    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise
    
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    
    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough. \n'\
            '(sz_b, warmup) = (2048,4000) is the official setting.\n'\
            'Using smaller batch with out longer wramup may cause '\
            'the warmup stage ends with only little data trained.'
            )
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    
    #=============== LOADING DATASETS ================#
    
    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise
    
    print(opt)
        

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx = opt.src_pad_idx,
        trg_pad_idx = opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj
    ).to(device)
    
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09,),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    
    train(transformer, training_data, validation_data, optimizer, device, opt)
    
    
# TODO bpe 복습
def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)
    
    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN    # * vars(object) -> __dict__, 멤버들 사전형으로 반환 : 여기선 src, trg 길이를 의미.
    
    train = TranslationDataset(
        fields=fields,
        path=opt.train_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length
    )
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length
    )
    
    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]    # vocab에서 pad_word 의 index 말하는거.
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)
    
    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator
        

def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))    # ? vocab 데이터가 어떻게 구성되어있는거지?
    
    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]
    
    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)
    
    #============ PREPARING MODEL =============#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'
    
    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}
    
    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)
    
    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    
    return train_iterator, val_iterator

if __name__ == '__main__':
    main()