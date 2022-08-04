# https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze    Tensor.unsqueeze(dim) = np.newaxis
# https://powerofsummary.tistory.com/158    register_buffer 이란?
# https://dololak.tistory.com/84    buffer 란? : 임시 저장공간 like 동영상 버퍼링, dynamic programming

import torch
import torch.nn as nn
import numpy as np
from model.Layers import EncoderLayer, DecoderLayer


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    sz_b, len_s = seq.size()
    subsequent_mask = (1-torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1
    )).bool()
    return subsequent_mask


class PostiionalEncoding(nn.Module):
    
    def __init__(self, d_hid, n_position=200):
        super(PostiionalEncoding, self).__init__()
        
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        # register_buffer: optimizer가 업데이트하지 않음. 하지만 GPU연산 가능한 layer.
        # 네트워크를 end2end 로 학습하려는데, 중간에 업데이트하지 않는 레이어 넣고싶을 때 사용하면 된다.
        # positional encoding 은 업데이트되는 값이 아니므로 이를 사용함.
        """ https://teamdable.github.io/techblog/PyTorch-Module
        
        >>> torch.nn.Module.register_buffer('running_mean', torch.zeros(num_features))  # example
        
        parameter가 말 그대로 buffer을 수행하기 위한 목적으로 활용한다.
        buffer도 state_dict에 저장되지만, backprop을 진행하지 않고 최적화에 사용되지 않는다는 의미이다.
        단순한 buffer로써의 역할을 맡는 본 모듈이다.
        """

    # return sinusoid FloatTensor table (1 S d)
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """sinusoid position encoding table"""
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        ## in original paper code,
        # angles = 1 / tf.pow(10000, (2* (i//2)) / tf.cast(d_model, tf.float32) )   # 임베딩 길이만큼 유니크한 값을 만든다.
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])   # sin, cos 에 넣을 서로다른 radian 값 만들기 완료.
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        # -1 ~ 1 사이의 비규칙적 분포값들로 mapping 된 sinusoid_table !
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)   # batch용 새로운 차원만들기?
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()   # clone() detach() ?