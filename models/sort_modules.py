#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from .transformer_modules import TransformerEncoderLayer
from .transformer_modules import TransformerFeedForward
from .transformer_modules import MultiHeadAttention
import math

RESCALE_COEF = 1 / math.sqrt(2)

def residual_connect(x, y, rescale=False):
    out = x + y
    if rescale:
        out *= RESCALE_COEF
    return out


class TransformerEncoder(nn.Module):
    """
    Self-attention -> FF -> layer norm
    """

    def __init__(self, sr_embed_layer, v_embed_layer, size, n_layers, ff_size=None, n_att_head=8, dropout_ratio=0.1, skip_connect=False, pos_enc=False, add_fc=False):
        super(TransformerEncoder, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.sr_embed_layer = sr_embed_layer
        self.v_embed_layer = v_embed_layer
        self.layer_norm = nn.LayerNorm(size)
        self.encoder_layers = nn.ModuleList()
        self.skip_connect = skip_connect
        self._rescale = 1. / math.sqrt(2)
        self.add_fc = add_fc
        self.pos_enc = pos_enc
        if self.add_fc:
            self.fc_feat = nn.Linear(512, 512)
        for _ in range(n_layers):
            layer = TransformerEncoderLayer(size, ff_size, n_att_head=n_att_head,
                                            dropout_ratio=dropout_ratio)
            self.encoder_layers.append(layer)

    def forward(self, det_seqs_v, det_seqs_sr, mask=None):
        if self.sr_embed_layer is not None and self.v_embed_layer is not None:
            x = self.v_embed_layer(det_seqs_v) + self.sr_embed_layer(det_seqs_sr, positional_encoding=self.pos_enc)
            if self.add_fc:
                x = self.fc_feat(x)
        first_x = x
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, mask)
            if self.skip_connect:
                x = self._rescale * (first_x + x)
        x = self.layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1, relative_pos=False):
        super(TransformerDecoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.cross_attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)
        self.layer_norm3 = nn.LayerNorm(size)

    def forward(self, x, x_mask, y, y_mask, mask_queries=None):
        # mask_queries: (b_s, seq_len, 1), x: (b_s, seq_len, hidden_size)
        # Attention layer
        h1 = self.layer_norm1(x)
        h1, _ = self.attention(h1, h1, h1, mask=x_mask)
        h1 = self.dropout(h1)
        h1 = residual_connect(h1, x)
        # h1 = h1 * mask_queries
        # Cross-attention
        h2 = self.layer_norm2(h1)
        h2, _ = self.attention(h2, y, y, mask=y_mask)
        h2 = self.dropout(h2)
        h2 = residual_connect(h2, h1)
        # h2 = h2 * mask_queries
        # Feed-forward layer
        h3 = self.layer_norm3(h2)
        h3 = self.ff_layer(h3)
        h3 = self.dropout(h3)
        h3 = residual_connect(h3, h2)
        # h3 = h3 * mask_queries
        return h3


class TransformerDecoder(nn.Module):
    """
    Self-attention -> cross-attenion -> FF -> layer norm
    """

    def __init__(self, embed_layer, size, n_layers, ff_size=None, n_att_head=8, dropout_ratio=0.1, skip_connect=False):
        super(TransformerDecoder, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self._skip = skip_connect
        self._reslace = 1. / math.sqrt(2)
        self.embed_layer = embed_layer
        self.layer_norm = nn.LayerNorm(size)
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerDecoderLayer(size, ff_size, n_att_head=n_att_head,
                                            dropout_ratio=dropout_ratio)
            self.encoder_layers.append(layer)

    def forward(self, x, x_mask, y, y_mask):
        # x: seq, y: img_feats
        batch_size, seq_len = x.shape
        first_x = x
        length_mask = (x == 0).unsqueeze(1).float() # b, s --> b, 1, s
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        self_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).unsqueeze(0) # s, s --> 1, s, s
        self_mask = (self_mask + length_mask).unsqueeze(1) # b, s, s --> b, 1, s, s
        self_mask = (self_mask == 0).to(x.device)  # (b_s, 1, seq_len, seq_len)
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, self_mask, y, y_mask)
            if self._skip:
                x = self._reslace * (first_x + x)
        x = self.layer_norm(x)
        return x

