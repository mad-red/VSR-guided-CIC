import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

RESCALE_COEF = 1 / math.sqrt(2)

def residual_connect(x, y, rescale=False):
    out = x + y
    if rescale:
        out *= RESCALE_COEF
    return out


class KeyValAttention(nn.Module):
    
    def __init__(self, scaling=False, dropout_ratio=0):
        """Initialize a key-value attention class.
        Args:
            scaling - Whether normalize the attention weights by sqrt(size)
            dropout_ratio - The probability of dropout on the logits
        """
        super(KeyValAttention, self).__init__()
        self._scaling = scaling
        self._dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else None

    def forward_2d(self, query, keys, values, mask=None, additional_logits=None):
        """Compute attention for 2-dimensional queries (batch x hidden).
        """
        context_vector, weights = self.forward_3d(query.unsqueeze(-2), keys, values, mask, additional_logits)
        return context_vector.squeeze(-2), weights.squeeze(-2)
    
    def forward_3d(self, query, keys, values, mask=None, additional_logits=None):
        """Compute attention for 3-dimensional input (batch x step x hidden).
        """
        logits = torch.matmul(query, keys.transpose(-2, -1))
        if additional_logits is not None:
            logits += additional_logits
        if self._scaling:
            logits /= math.sqrt(query.shape[-1])
        if mask is not None:
            if mask.dim() < logits.dim():
                mask = mask.unsqueeze(-2)
            logits = logits.masked_fill(mask == 0, -1e3)
        if self._dropout is not None:
            weights = self._dropout(F.softmax(logits, dim=-1))
        else:
            weights = F.softmax(logits, dim=-1)
        context_vector = torch.matmul(weights, values)
        return context_vector, weights
    
    def forward(self, query, keys, values, mask=None, additional_logits=None):
        """Compute the context vector with key value attention.
        
        Returns:
            context vector and attention weights.
        """
        if query.dim() == keys.dim() - 1:
            return self.forward_2d(query, keys, values, mask, additional_logits=additional_logits)
        else:
            return self.forward_3d(query, keys, values, mask, additional_logits=additional_logits)


class MultiHeadAttention(nn.Module):
    """The implementation of multi-head attention.
    
    Following the original description in the transformer paper.
    """

    _RELATIVE_POS_CLIP = 2
    
    def __init__(self, out_size, num_head=8, hidden_size=None, additive=False, dropout_ratio=0, relative_pos=False):
        super(MultiHeadAttention, self).__init__()
        if hidden_size is None:
            hidden_size = out_size
        self._num_head = num_head
        self._hidden_size = hidden_size
        self._out_size = out_size
        self._additive = additive
        if relative_pos:
            self.relative_posmatrix = nn.Embedding(self._RELATIVE_POS_CLIP * 2 + 1, hidden_size)
        else:
            self.relative_posmatrix = None
        self._attention = KeyValAttention(scaling=True, dropout_ratio=dropout_ratio, )
        if additive:
            # Taken from RNMT+ paper
            raise NotImplementedError
        else:
            self.linear_Q = nn.Linear(out_size, hidden_size)
            self.linear_K = nn.Linear(out_size, hidden_size)
            self.linear_V = nn.Linear(out_size, hidden_size)
        self.linear_O = nn.Linear(hidden_size, out_size)
    
    def forward_2d(self, query, keys, values, mask=None):
        """Compute attention for 2-dimensional queries (batch x hidden).
        """
        query = query.unsqueeze(1)  # (B, 1, H)
        context_vectors, weights = self.forward_3d(query, keys, values, mask=mask)
        context_vectors = context_vectors.squeeze(1)
        weights = weights.squeeze(1)
        return context_vectors, weights
    
    def forward_3d(self, query, keys, values, mask=None):
        """Compute attention for 3-dimensional input (batch x step x hidden).
        """
        B = query.shape[0]
        head_dim = self._hidden_size // self._num_head
        transformed_query = self.linear_Q(query)
        if self.relative_posmatrix is not None:
            T2 = query.shape[1]
            T1 = keys.shape[1]
            pos = torch.arange(T1).repeat(T2, 1)
            relpos = pos - torch.arange(T2)[:, None]
            relpos = torch.clamp(relpos, -self._RELATIVE_POS_CLIP, self._RELATIVE_POS_CLIP)
            relpos += self._RELATIVE_POS_CLIP
            if torch.cuda.is_available():
                relpos = relpos.cuda()
            relpos_embed = self.relative_posmatrix(relpos)
            relpos_logits = (transformed_query.unsqueeze(-2) * relpos_embed.unsqueeze(0)).sum(-1)
            relpos_logits = relpos_logits.unsqueeze(1)
        else:
            relpos_logits = None
        query = transformed_query.view(B, -1, self._num_head, head_dim).transpose(1, 2)  # (B, 4, T2, H)
        keys = self.linear_K(keys).view(keys.shape[0], -1, self._num_head, head_dim).transpose(1, 2)
        values = self.linear_V(values).view(values.shape[0], -1, self._num_head, head_dim).transpose(1, 2)
        if mask is not None and mask.dim() < keys.dim():
            mask = mask.unsqueeze(1)
        context_vectors, weights = self._attention(query, keys, values, mask=mask, additional_logits=relpos_logits)  # (B, 4, T2, H)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(B, -1, self._num_head * head_dim)  # (B, T2, H)
        context_vectors = self.linear_O(context_vectors)
        return context_vectors, weights
    
    def forward(self, query, keys, values, mask=None):
        """Compute the context vector with key value attention.
        
        Returns:
            context vector and attention weights.
        """
        if query.dim() == 2:
            return self.forward_2d(query, keys, values, mask)
        elif query.dim() == 3:
            return self.forward_3d(query, keys, values, mask)
        else:
            raise NotImplementedError


class LabelSmoothingKLDivLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        self.padding_idx = ignore_index
        super(LabelSmoothingKLDivLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target, mask=None):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # import pdb;pdb.set_trace()
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        if mask is None:
            model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0) # target: b * s --> b * s, 1
        else:
            model_prob.masked_fill_((mask == 0).unsqueeze(1), 0) # target: b * s --> b * s, 1
        return F.kl_div(output, model_prob, reduction="sum")


class TransformerEmbedding(nn.Embedding):
    """
    Rescale the embeddings.
    TODO: share the weight with pre-softmax linear transformation
    """
    
    def __init__(self, num_embeddings, embedding_dim, dropout_ratio=0.1):
        super(TransformerEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.pos_layer = PositionalEmbedding(embedding_dim)
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, x, start=None, positional_encoding=False, mask_pos=None):
        """
        Compute the embeddings with positional encoderi
        Args:
            x - input sequence ~ (batch, len)
            start - the begining position (option)
            positional_encoding - whether using positional encoding
        """
        embed = super(TransformerEmbedding, self).forward(x)
        embed = embed * math.sqrt(self.embedding_dim)
        if positional_encoding:
            if embed.dim() == 2:
                # Collapse one dimension of positional embedding
                pos_embed = self.pos_layer(embed.unsqueeze(1), start=start)
                pos_embed = pos_embed.squeeze(1)
            else:
                pos_embed = self.pos_layer(embed, start=start)
            if mask_pos is None:
                embed += pos_embed
            else:
                embed += pos_embed * mask_pos
        return self.dropout(embed)


class AddPositionalEmbedding(nn.Module):
    """
    Rescale the embeddings.
    TODO: share the weight with pre-softmax linear transformation
    """
    
    def __init__(self, hidden_size, dropout_ratio=0.1):
        super(AddPositionalEmbedding, self).__init__()
        self.pos_layer = PositionalEmbedding(hidden_size)
        # self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, embed, start=None, positional_encoding=True):
        """
        Compute the embeddings with positional encoderi
        Args:
            x - input sequence ~ (batch, len)
            start - the begining position (option)
            positional_encoding - whether using positional encoding
        """
        if positional_encoding:
            if embed.dim() == 2:
                # Collapse one dimension of positional embedding
                pos_embed = self.pos_layer(embed.unsqueeze(1), start=start)
                pos_embed = pos_embed.squeeze(1)
            else:
                pos_embed = self.pos_layer(embed, start=start)
            embed += pos_embed
        return embed


class TemporalMasking(nn.Module):
    """
    Produce (1, size, size) mask for masking out previous positions.
    """
    
    def __init__(self, max_len=1000):
        super(TemporalMasking, self).__init__()
        shape = (1, max_len, max_len)
        subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
        mask = (torch.from_numpy(subsequent_mask) == 0).float()
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        """Compute the temporal mask on given embeddings
        
        Args:
            x - embedding ~ (batch, len, size)
        """
        if type(x) == int:
            seq_len = x
        else:
            seq_len = x.shape[-2]
        return self.mask[:, :seq_len, :seq_len]
        

class PositionalEmbedding(nn.Module):
    """
    This function is stealed from The Annotated Transformer (same as openNMT implementation).
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax
    """
    
    def __init__(self, size, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, size, 2).float() *
                             -(math.log(10000.0) / size)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x, start=None):
        """
        Return 3d tensor with shape (1, len, size).
        """
        if start is None:
            start = 0
        if type(x) == int:
            length = x
        else:
            length = x.shape[1]
        return Variable(self.pe[:, start:start + length], requires_grad=False)


class TransformerFeedForward(nn.Module):
    """The common feed-forward layer."""
    
    def __init__(self, size, hidden_size, dropout_ratio=0.1, activation="relu"):
        super(TransformerFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.dropout = nn.Dropout(dropout_ratio)
        if activation == "relu":
            self._activate = F.relu
        elif activation == "gelu":
            self._activate = gelu
        else:
            raise NotImplementedError
        
    def forward(self, x):
        return self.w_2(self.dropout(self._activate(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1, relative_pos=False):
        super(TransformerEncoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)

    def forward(self, x, src_mask=None):
        # Attention layer
        y1 = self.layer_norm1(x)
        y1, _ = self.attention(y1, y1, y1, mask=src_mask)
        y1 = self.dropout(y1)
        y1 = residual_connect(y1, x)
        # Feed-forward layer
        y2 = self.layer_norm2(y1)
        y2 = self.ff_layer(y2)
        y2 = self.dropout(y2)
        y2 = residual_connect(y2, y1)
        return y2


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
    
    def forward(self, encoder_states, decoder_states, src_mask=None, tgt_mask=None, last_only=False):
        """
        Args:
            last_only - only compute the states for the last position
        """
        # Self-attention layer
        y1 = self.layer_norm1(decoder_states)
        if last_only:
            y1, _ = self.attention(y1[:, -1].unsqueeze(1), y1, y1, mask=tgt_mask)
            y1 = self.dropout(y1)
            y1 = residual_connect(y1, decoder_states[:, -1].unsqueeze(1))
        else:
            y1, _ = self.attention(y1, y1, y1, mask=tgt_mask)
            y1 = self.dropout(y1)
            y1 = residual_connect(y1, decoder_states)
        # Cross-attention layer
        y2 = self.layer_norm2(y1)
        y2, _ = self.attention(y2, encoder_states, encoder_states, mask=src_mask)
        y2 = self.dropout(y2)
        y2 = residual_connect(y2, y1)
        # Feed-forward layer
        y3 = self.layer_norm3(y2)
        y3 = self.ff_layer(y3)
        y3 = self.dropout(y3)
        y3 = residual_connect(y3, y2)
        return y3
