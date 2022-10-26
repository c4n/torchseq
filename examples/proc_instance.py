import pdb
import json
import torch
import jsonlines
import sacrebleu
import torch.nn.functional as F
from collections import defaultdict

# Load config library
from torchseq.utils.config import Config
# Load dataset library
from torchseq.utils.tokenizer import Tokenizer
from torchseq.datasets.json_dataset import JsonDataset
from torchseq.datasets.json_loader import JsonDataLoader
# Load model library
from torchseq.agents.seq2seq_agent import Seq2SeqAgent

def load_model_config(path):
    # Change the config to use the custom dataset
    with open(path) as f:
        cfg_dict = json.load(f)
    cfg_dict["beam_search"]["beam_width"] = 3 # set beam
    cfg_dict["dataset"] = "json"
    cfg_dict["json_dataset"] = {
        "path": None,
        "field_map": [
            {"type": "copy", "from": "input", "to": "target"},
            {"type": "copy", "from": "input", "to": "source"},],}
    return Config(cfg_dict)


def load_dataset(config, path_data):
    # Load the data
    with jsonlines.open(path_data + "/mscoco-eval/test.jsonl") as f:
        rows = [row for row in f]
        dataset = [{'input': row['sem_input']} for row in rows]
    # Dataloder
    dataset = JsonDataLoader(config, test_samples=dataset, data_path=path_data)
    return dataset


class InstanceProcessing:
    def __init__(self, model, config, path_vocab, device) -> None:
        self.device = device
        self.tok_window = config.prepro.tok_window
        self.fields = config.json_dataset.data["field_map"]
        self.input_tokenizer = Tokenizer(config.prepro.get_first(["input_tokenizer", "tokenizer"]), path_vocab)
        self.output_tokenizer = Tokenizer(config.prepro.get_first(["output_tokenizer", "tokenizer"]), path_vocab)
        self.include_lang_codes = config.prepro.data.get("include_lang_codes", False)
        self.drop_target_lang_codes = config.prepro.data.get("drop_target_lang_codes", False)
        self.mask_prob = config.prepro.data.get("token_mask_prob", 0)
        self.pad_id = self.input_tokenizer.pad_id
        self.model = model

    def __call__(self, data):
        batch = []
        for index, instance in enumerate(data):
            instance = self._instance_input(instance)
            temp_feature = {}
            for key, val in instance.items():
                val = self._preproc(key, val)
                temp_feature[key] = val
            batch.append(temp_feature)
        
        batch_input = self._pad_and_order_sequences(batch)
        output = self.model.get_vector_batch(batch_input)
        return output.squeeze(dim=1)

    def _pad_and_order_sequences(self, batch):
        keys = batch[0].keys()
        max_lens = {k: max(len(x[k]) for x in batch) for k in keys}
        for x in batch:
            for k in keys:
                if k[0] == "_":
                    continue
                if k == "a_pos":
                    x[k] = F.pad(x[k], (0, max_lens[k] - len(x[k])), value=0)
                elif k[-5:] != "_text":
                    x[k] = F.pad(x[k], (0, max_lens[k] - len(x[k])), value=self.pad_id)

        tensor_batch = {}
        for k in keys:
            if k[-5:] != "_text" and k[0] != "_":
                tensor_batch[k] = torch.stack([x[k] for x in batch], 0)
                if k[-4:] == "_len":
                    tensor_batch[k] = tensor_batch[k].squeeze(dim=1)
            else:
                tensor_batch[k] = [x[k] for x in batch]
        return tensor_batch

    
    def _instance_input(self, tokens):
        """ 
        Input: 
            text: A bla xxx
        Output: 
            features dict:
                keys          |  Values
                -----------------------
                target_text   | A bla xxx
                source_text   | A bla xxx
                target     	  | tensor([101, 1037, 2304, 11990,  9055]) 
                target_len 	  | tensor([13]) 
                source     	  | tensor([101, 1037, 2304, 11990,  9055]) 
                source_len 	  | tensor([13]) e', 'source_len'
        """
        
        sample = {"input": tokens}
        output = JsonDataset.to_tensor(
            sample, 
            self.fields,
            self.input_tokenizer,
            self.output_tokenizer,
            tok_window=self.tok_window,
            include_lang_codes=self.include_lang_codes,
            drop_target_lang_codes=self.drop_target_lang_codes,
            mask_prob=self.mask_prob)
        return output

    def _preproc(self, key, val):
        if key[-5:] != "_text" and key[0] != "_":
            val = val.to(self.device)
        return (val)


if __name__ == "__main__":
        
    device = 'cuda'
    PATH_DATA = '../data/'
    PATH_PRETRAINED = "../data/"
    PATH_CHECKPOINT = '../runs/vae/20220920_232720_paraphrasing_vae_mscoco_789'

    # Load config
    config = load_model_config(PATH_CHECKPOINT + '/config.json')

    # Initial model instance
    model = Seq2SeqAgent(
        config=config, run_id=None,  
        output_path=None, data_path=PATH_PRETRAINED, 
        silent=False, verbose=False, 
        training_mode=False)

    # Load the checkpoint
    model.load_checkpoint(PATH_CHECKPOINT + '/model/checkpoint.pt')
    model.model.eval()

    # Create args and load data based on the config 
    data = load_dataset(config, PATH_DATA)

    get_vectors = InstanceProcessing(model, config, PATH_PRETRAINED, device)

    # Input instance
    input_instances =  [
        'Test1: A black Honda motorcycle parked in front of a garage.',
        'test11: A black Honda motorcycle parked in front of a garage.']

    sent_emb = get_vectors(input_instances)
    print(sent_emb)
    # pdb.set_trace()