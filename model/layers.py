import torch.nn as nn

from .sub_layers import MultiHeadAttention
from .sub_layers import PointWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout_ratio):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src, src_mask):
        _src, _ = self.multi_head_attention(src, src, src, src_mask)
        src = self.layer_norm(src + self.dropout(_src))

        _src = self.point_wise_feed_forward(src)
        src = self.layer_norm(src + self.dropout(_src))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout_ratio):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, n_head)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, src_output, trg_mask, src_mask):

        _trg, _ = self.multi_head_attention(trg, trg, trg, trg_mask)
        trg = self.attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.multi_head_attention(
            trg, src_output, src_output, src_mask
        )
        trg = self.attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.point_wise_feed_forward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention
