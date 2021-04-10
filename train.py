from pytorch_lightning import callbacks
import spacy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from omegaconf import OmegaConf

from net import Transformer


def tokenize_de():
    spacy_de = spacy.load("de_core_news_sm")

    def tokenize(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    return tokenize


def tokenize_en():
    spacy_en = spacy.load("en_core_web_sm")

    def tokenize(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    return tokenize


def main(config):

    SRC = Field(
        tokenize=tokenize_de(), init_token="<sos>", eos_token="<eos>", lower=True
    )

    TRG = Field(
        tokenize=tokenize_en(), init_token="<sos>", eos_token="<eos>", lower=True
    )

    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(SRC, TRG)
    )

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=config.batch_size
    )

    config.pad_token_id = SRC.vocab["<pad>"]
    config.input_dim = len(SRC.vocab)
    config.output_dim = len(TRG.vocab)

    net = Transformer(config, train_iter, valid_iter, test_iter)

    logger = TensorBoardLogger("logs")
    early_stopping_callback = EarlyStopping("val_loss")
    checkpoint_callback = ModelCheckpoint("checkpoint", "{epoch}-{val_loss:.4f}")

    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )

    trainer.fit(net)


if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    main(config)