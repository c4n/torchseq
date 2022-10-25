import pdb
import json
import torch
import jsonlines
import sacrebleu
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
    def __init__(self, config, path_vocab) -> None:
        self.tok_window = config.prepro.tok_window
        self.fields = config.json_dataset.data["field_map"]
        self.input_tokenizer = Tokenizer(config.prepro.get_first(["input_tokenizer", "tokenizer"]), path_vocab)
        self.output_tokenizer = Tokenizer(config.prepro.get_first(["output_tokenizer", "tokenizer"]), path_vocab)
        self.include_lang_codes = config.prepro.data.get("include_lang_codes", False)
        self.drop_target_lang_codes = config.prepro.data.get("drop_target_lang_codes", False)
        self.mask_prob = config.prepro.data.get("token_mask_prob", 0)
    

    def __call__(self, data, device):
        batch = defaultdict(lambda: [])
        for index, instance in enumerate(data):
            instance = self._instance_input(instance)
            for key, val in instance.items():
                batch[key].append(val)
        batch = {key:self._preproc(key, val, device) 
            for key,val in batch.items()}
        return batch

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

    def _preproc(self, key, val, device="cuda"):
        if key[-5:] != "_text" and key[0] != "_":
            val = torch.stack(val)
            val = val.to(device)
        return (val)


if __name__ == "__main__":
        
    PATH_DATA = '../data/'
    PATH_PRETRAINED = "../data/"
    PATH_CHECKPOINT = '../runs/vae/20220920_232720_paraphrasing_vae_mscoco_789'

    # Load config
    config = load_model_config(PATH_CHECKPOINT + '/config.json')
    # Create args and load data based on the config 
    data = load_dataset(config, PATH_DATA)


    ################################################################################
    ############################ Batch input features ##############################
    batch_proc = InstanceProcessing(config, PATH_PRETRAINED)

    # Batch input: list of text
    input_instance = [
        'A black Honda motorcycle parked in front of a garage.',
        'A black Honda motorcycle parked in front of a garage.',]
    # pre-processing 
    output_instance = batch_proc(input_instance, device='cuda')
    # show results
    [print(f"Key: {k:10s} \tValues: {v[0][:10]} ") # only value from index zero
               for k,v in output_instance.items()];


    ################################################################################
    ############################# get batch embedding ##############################
    # Initial model instance
    model = Seq2SeqAgent(
        config=config, run_id=None,  
        output_path=None, data_path=PATH_PRETRAINED, 
        silent=False, verbose=False, 
        training_mode=False)

    # Load the checkpoint
    model.load_checkpoint(PATH_CHECKPOINT + '/model/checkpoint.pt')
    model.model.eval();

    #### Todo ####
    pdb.set_trace()
    encoding_pooled_list = model.get_instance_vector({"_text":output_instance})
    print(encoding_pooled_list)