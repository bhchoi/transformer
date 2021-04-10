import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.scale = torch.sqrt(torch.Tensor([self.d_model]))

    def forward(self, Q, K, mask=False):
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2))
        # scores = torch.matmul(Q, K.transpose(2, 3))

        scores = scores / torch.sqrt(self.scale)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(scores, dim=-1)
        return attention
        # return torch.matmul(attention, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)

        self.fc_output = nn.Linear(d_model, d_model)

        self.attn = Attention(d_model)

    def forward(self, query, key, value, mask=None):
        # x: (len, batch_size, d_model)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # n_head로 분리하기.
        # len, batch_size, d_model / n_head, n_head
        # seq_len = query.shape[1]
        batch_size = query.shape[0]
        head_dim = self.d_model // self.n_head

        # Q: (batch_size, len, d_model)
        # Q = Q.view(seq_len, batch_size, head_dim, self.n_head).permute(1, 3, 0, 2)
        # K = K.view(seq_len, batch_size, head_dim, self.n_head).permute(1, 3, 0, 2)
        # V = V.view(seq_len, batch_size, head_dim, self.n_head).permute(1, 3, 0, 2)

        Q = Q.view(batch_size, -1, self.n_head, head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, head_dim).permute(0, 2, 1, 3)

        # Q: (batch_size, head, len, dim)

        attention = self.attn(Q, K, mask)

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.d_model)

        return self.fc_output(x), attention


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, ff_dim):
        super(PointWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)