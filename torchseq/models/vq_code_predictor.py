import torch
import torch.nn as nn
import copy

from torchseq.utils.functions import onehot

import logging


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads):
        super(MLPClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim * num_heads)

        self.drop1 = torch.nn.Dropout(p=0.2)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.num_heads = num_heads
        self.output_dim = output_dim

    def forward(self, x, seq=None):
        outputs = self.drop1(torch.nn.functional.relu(self.linear(x)))
        outputs = self.drop2(torch.nn.functional.relu(self.linear2(outputs)))
        outputs = self.linear3(outputs)
        return outputs.reshape(-1, self.num_heads, self.output_dim)


class LstmClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, seq_dim=None):
        super(LstmClassifier, self).__init__()
        self.lstm_in = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTMCell(
            hidden_dim,
            hidden_dim,
        )
        self.lstm_out = torch.nn.Linear(hidden_dim, output_dim)

        if seq_dim is not None and seq_dim != hidden_dim:
            self.seq_proj = torch.nn.Linear(seq_dim, hidden_dim, bias=False)
        else:
            self.seq_proj = None

        self.drop1 = torch.nn.Dropout(p=0.2)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.num_heads = num_heads
        self.output_dim = output_dim

    def forward(self, x, seq=None):
        outputs = self.drop1(torch.nn.functional.relu(self.lstm_in(x)))
        rnn_out = []
        hx, cx = outputs, torch.zeros_like(outputs)
        if self.seq_proj is not None and seq is not None:
            seq = self.seq_proj(seq)
        for hix in range(self.num_heads if seq is None else seq.shape[1]):
            hx, cx = self.rnn(seq[:, hix, :] if seq is not None else hx, (hx, cx))
            rnn_out.append(hx)
        outputs = torch.stack(rnn_out, dim=1)
        outputs = self.lstm_out(outputs)
        return outputs


class RecurrentMlpClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, seq_dim=None):
        super(RecurrentMlpClassifier, self).__init__()
        recur_in_dim = input_dim + (seq_dim if seq_dim is not None else hidden_dim)
        # dims = [input_dim] + [recur_in_dim] * (num_heads - 1)

        self.linear = torch.nn.ModuleList([torch.nn.Linear(recur_in_dim, hidden_dim) for i in range(num_heads)])
        self.linear2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(num_heads)])
        self.linear3 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, output_dim) for i in range(num_heads)])

        self.drop1 = torch.nn.ModuleList([torch.nn.Dropout(p=0.2) for i in range(num_heads)])
        self.drop2 = torch.nn.ModuleList([torch.nn.Dropout(p=0.2) for i in range(num_heads)])
        self.num_heads = num_heads
        self.output_dim = output_dim

    def forward(self, x, seq=None):
        all_outputs = []
        for hix in range(self.num_heads if seq is None else seq.shape[1]):
            full_input = torch.cat([x, seq[:, hix, :]], dim=-1)
            outputs = self.drop1[hix](torch.nn.functional.relu(self.linear[hix](full_input)))
            outputs = self.drop2[hix](torch.nn.functional.relu(self.linear2[hix](outputs)))
            outputs = self.linear3[hix](outputs)
            all_outputs.append(outputs)
        all_outputs = torch.stack(all_outputs, dim=1)
        return all_outputs


class VQCodePredictor(torch.nn.Module):
    def __init__(self, config, transitions=None, embeddings=None):
        super(VQCodePredictor, self).__init__()

        self.logger = logging.getLogger("VQCodePredictor")

        if config.get("use_lstm", False):
            self.classifier = LstmClassifier(
                config.input_dim,
                config.output_dim,
                config.hidden_dim,
                config.num_heads,
                seq_dim=config.get("lstm_seq_dim", None),
            )
        elif config.get("use_recurrent_mlp", False):
            self.classifier = RecurrentMlpClassifier(
                config.input_dim,
                config.output_dim,
                config.hidden_dim,
                config.num_heads,
                seq_dim=config.get("lstm_seq_dim", None),
            )
        else:
            self.classifier = MLPClassifier(config.input_dim, config.output_dim, config.hidden_dim, config.num_heads)

        self.transitions = transitions

        self.embeddings = embeddings

        self.config = config

        # self.criterion = torch.nn.CrossEntropyLoss().cuda() # computes softmax and then the cross entropy

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config.lr)

    def infer(self, encoding, batch={}, top_k=1, outputs_to_block=None):
        # TODO: Batchify this...
        self.classifier.eval()

        seq_dim = (
            self.config.hidden_dim
            if self.config.get("lstm_seq_dim", None) is None
            else self.config.get("lstm_seq_dim", None)
        )
        seq_init = (
            torch.zeros(encoding.shape[0], 1, seq_dim).to(encoding.device)
            if self.config.get("autoregressive_lstm", False)
            else None
        )

        beam_width = self.config.get("beam_width", 3)

        if top_k > beam_width:
            self.logger.warn("top-k is larger than beam_width - fewer candidates than expected will be returned!")

        # print(encoding.shape, seq_init.shape)
        outputs = self.classifier(encoding, seq=seq_init)

        all_pred_codes = []
        for bix, logits in enumerate(outputs):
            joint_probs = [([], 0)]
            for h_ix in range(self.config.num_heads):
                new_hypotheses = []
                for i, (combo, prob) in enumerate(joint_probs):
                    if h_ix > 0 and self.transitions is not None:
                        prev_oh = onehot(torch.tensor(combo[-1]).to(logits.device), N=self.config.output_dim) * 1.0
                        curr_logits = logits[h_ix, :] + self.transitions[h_ix - 1](prev_oh).detach()
                    elif self.config.get("autoregressive_lstm", False):

                        seq_init = torch.zeros(1, 1, seq_dim).to(encoding.device)
                        seq_embedded = [
                            embed(torch.tensor(x).to(encoding.device)).unsqueeze(0).unsqueeze(1).detach()
                            for x, embed in zip(combo, self.embeddings)
                        ]
                        seq = torch.cat([seq_init, *seq_embedded], dim=1)
                        if self.config.get("additive", False):
                            seq = torch.cumsum(seq, dim=1)
                            # for j in range(h_ix+1):
                            #     seq_cumul.append(torch.sum(seq[:, : (j + 1)], dim=1, keepdim=True))
                            #     use torch.cumsum!!
                            # seq = torch.cat(seq_cumul, dim=1)

                        curr_logits = self.classifier(encoding[bix].unsqueeze(0), seq)[0, h_ix, :]
                    else:
                        curr_logits = logits[h_ix, :]
                    probs = torch.softmax(curr_logits, -1)
                    if self.config.get("blocking_weight", 0) > 0 and outputs_to_block is not None:
                        block_ix = outputs_to_block[bix, h_ix]
                        probs[block_ix] *= 1 - self.config.get("blocking_weight", 0)
                    probs, predicted = torch.topk(probs, beam_width, -1)
                    for k in range(beam_width):
                        new_hyp = [copy.copy(combo), copy.copy(prob)]
                        new_hyp[0].append(predicted[k].item())
                        new_hyp[1] += torch.log(probs[k]).item()

                        new_hypotheses.append(new_hyp)

                joint_probs = new_hypotheses
                joint_probs = sorted(joint_probs, key=lambda x: x[1], reverse=True)[:beam_width]
            pred_codes = [x[0] for x in sorted(joint_probs, key=lambda x: x[1], reverse=True)[:beam_width]]
            all_pred_codes.append(pred_codes)

        if top_k == 1:
            top_1_codes = torch.IntTensor(all_pred_codes)[:, 0, :].to(encoding.device)
            return top_1_codes
        else:
            top_k_codes = torch.IntTensor(all_pred_codes)[:, :top_k, :].to(encoding.device)
            return top_k_codes

    def train_step(self, encoding, code_mask, take_step=True):
        # Encoding should be shape: bsz x dim
        # code_mask should be a n-hot vector, shape: bsz x codebook
        self.classifier.train()

        self.optimizer.zero_grad()
        if self.config.get("autoregressive_lstm", False):
            seq_dim = (
                self.config.hidden_dim
                if self.config.get("lstm_seq_dim", None) is None
                else self.config.get("lstm_seq_dim", None)
            )
            seq_init = torch.zeros(encoding.shape[0], 1, seq_dim).to(encoding.device)
            seq_embedded = [
                torch.matmul(code_mask[:, hix, :].float(), self.embeddings[hix].weight.detach()).unsqueeze(1)
                for hix in range(self.config.num_heads - 1)
            ]

            seq = torch.cat([seq_init, *seq_embedded], dim=1)
            if self.config.get("additive", False):
                seq = torch.cumsum(seq, dim=1)
                # for j in range(self.config.num_heads):
                #     seq_cumul.append(torch.sum(seq[:, : (j + 1)], dim=1, keepdim=True))
                # seq = torch.cat(seq_cumul, dim=1)
        else:
            seq = None
        outputs = self.classifier(encoding, seq=seq)

        logits = [outputs[:, 0, :].unsqueeze(1)]

        # Use teacher forcing to train the subsequent heads
        for head_ix in range(1, self.config.num_heads):
            if self.transitions is not None:
                logits.append(
                    outputs[:, head_ix, :].unsqueeze(1)
                    + self.transitions[head_ix - 1](code_mask[:, head_ix - 1, :]).detach().unsqueeze(1)
                )
            else:
                logits.append(outputs[:, head_ix, :].unsqueeze(1))
        logits = torch.cat(logits, dim=1)

        loss = torch.sum(
            -1 * torch.nn.functional.log_softmax(logits, dim=-1) * code_mask / code_mask.sum(dim=-1, keepdims=True),
            dim=-1,
        ).mean()  #
        if take_step:
            loss.backward()
            self.optimizer.step()

        return loss.detach()

    def forward(self, encoding):
        raise Exception(
            "fwd() shouldn't be called on the code predictor! Use either the training or inference methods"
        )
        logits = self.classifier(encoding)

        return logits
