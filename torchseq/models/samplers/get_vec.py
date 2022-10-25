import torch
import torch.nn as nn

from torchseq.utils.tokenizer import Tokenizer, FAIRSEQ_LANGUAGE_CODES


class GetVecSampler(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(GetVecSampler, self).__init__()
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

    def forward(self, model, batch, tgt_field):
        curr_batch_size = batch[[k for k in batch.keys() if k[-5:] != "_text"][0]].size()[0]

        max_output_len = batch[tgt_field].size()[1]

        BART_HACK = self.config.eval.data.get("prepend_eos", False)
        MBART_HACK = self.config.eval.data.get("prepend_langcode", False)

        # Create vector of SOS + placeholder for first prediction
        output = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.bos_id).to(self.device)
        logits = (
            torch.FloatTensor(curr_batch_size, 1, self.config.prepro.get_first(["output_vocab_size", "vocab_size"]))
            .fill_(float("-inf"))
            .to(self.device)
        )
        logits[:, :, self.tokenizer.bos_id] = float("inf")

        output_done = torch.BoolTensor(curr_batch_size).fill_(False).to(self.device)
        padding = torch.LongTensor(curr_batch_size).fill_(self.tokenizer.pad_id).to(self.device)

        if BART_HACK:
            dummy_token = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.eos_id).to(self.device)
            output = torch.cat([dummy_token, output], dim=1)

        if MBART_HACK:
            lang_token = batch["tgt_lang"].unsqueeze(-1)
            eos_token = torch.LongTensor(curr_batch_size, 1).fill_(self.tokenizer.eos_id).to(self.device)
            output = torch.cat([eos_token, lang_token], dim=-1)

        seq_ix = 0
        memory = {}


        encoding_pooled = model.forward_encoded_vector(batch, output, memory)

        return encoding_pooled