import math

import torch
import torch.nn as nn
from transformers import BartModel, BertModel

from torchseq.models.pooling import MultiHeadedPooling
from torchseq.models.positional_embeddings import PositionalEncoding
from torchseq.models.multihead_output import MultiHeadOutput
from torchseq.utils.tokenizer import Tokenizer
from torchseq.models.vq_vae import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerMultiHead

from torchseq.models.encoder import SequenceEncoder
from torchseq.models.decoder import SequenceDecoder


class TransformerParaphraseModel(nn.Module):
    def __init__(self, config, src_field="s1"):
        super().__init__()
        self.config = config

        self.src_field = src_field

        self.seq_encoder = SequenceEncoder(config)
        self.seq_decoder = SequenceDecoder(config, embeddings=self.seq_encoder.embeddings)

    def forward(self, batch, output, memory=None, tgt_field=None):
        if memory is None:
            memory = dict()

        # First pass? Construct the encoding
        if "encoding" not in memory:
            encoding_pooled, memory = self.seq_encoder(batch[self.src_field], batch[self.src_field + "_len"], memory)
            memory["encoding"] = encoding_pooled

        # Build some masks
        logits, memory = self.seq_decoder(output, memory)

        return logits, memory
