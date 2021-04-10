import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable


from model.layers import EncoderLayer
from model.layers import DecoderLayer


class Transformer(pl.LightningModule):
    def __init__(self, config, train_iter, val_iter, test_iter):
        super(Transformer, self).__init__()
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.pad_token_id = config.pad_token_id

        self.src_embedding = Embedding(config.input_dim, config.d_model)
        self.trg_embedding = Embedding(config.output_dim, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.max_len, config.d_model, config.dropout_ratio
        )
        self.encoder = Encoder(
            config.d_model,
            config.n_head,
            config.ff_dim,
            config.dropout_ratio,
            config.n_layers,
        )
        self.decoder = Decoder(
            config.d_model,
            config.n_head,
            config.ff_dim,
            config.dropout_ratio,
            config.n_layers,
        )

        self.fc_out = nn.Linear(config.d_model, config.output_dim)

    def forward(self, src, trg):
        src = src.permute(1, 0)
        src_mask = self.get_src_mask(src)
        src_embedded = self.src_embedding(src)
        src_embedded = self.positional_encoding(src_embedded)
        src_output = self.encoder(src_embedded, src_mask)

        trg = trg.permute(1, 0)
        trg_mask = self.get_trg_mask(trg)
        trg_embedded = self.trg_embedding(trg)
        trg_embedded = self.positional_encoding(trg_embedded)
        x, attention = self.decoder(trg_embedded, src_output, trg_mask, src_mask)
        x = self.fc_out(x)

        return x, attention

    def training_step(self, batch, batch_nb):
        output, attention = self(batch.src, batch.trg[:-1, :])

        output_dim = output.shape[2]
        output = output.view(-1, output_dim)
        trg = batch.trg[1:, :].permute(1, 0).contiguous().view(-1)
        loss = torch.nn.functional.cross_entropy(output, trg)

        return loss

    def validation_step(self, batch, batch_nb):
        output, attention = self(batch.src, batch.trg[:-1, :])

        output_dim = output.shape[2]
        output = output.view(-1, output_dim)
        trg = batch.trg[1:, :].permute(1, 0).contiguous().view(-1)
        val_loss = torch.nn.functional.cross_entropy(output, trg)

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        total_val_loss = torch.stack([output["val_loss"] for output in outputs]).mean()
        self.log("val_loss", total_val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    def get_src_mask(self, src):
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def get_trg_mask(self, trg):
        def subsequent_mask(size):
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
            return torch.from_numpy(subsequent_mask) == 0

        trg_mask = (trg != self.pad_token_id).unsqueeze(1).unsqueeze(2)

        trg_mask = trg_mask & Variable(
            subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
        )
        return trg_mask


class Embedding(nn.Module):
    def __init__(self, input_size, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_size, d_model)

    def forward(self, x):
        embedded = self.embedding(x) / math.sqrt(self.d_model)
        return embedded


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout_ratio):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout_ratio)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout_ratio, n_layers):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.encoder_layer = EncoderLayer(d_model, n_head, ff_dim, dropout_ratio)

    def forward(self, src, src_mask):

        for _ in range(self.n_layers):
            src = self.encoder_layer(src, src_mask)

        return src


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout_ratio, n_layers):
        super(Decoder, self).__init__()

        self.n_layers = n_layers
        self.decoder_layer = DecoderLayer(d_model, n_head, ff_dim, dropout_ratio)

    def forward(self, trg, src_output, trg_mask, src_mask):

        for _ in range(self.n_layers):
            trg, attention = self.decoder_layer(trg, src_output, trg_mask, src_mask)

        return trg, attention