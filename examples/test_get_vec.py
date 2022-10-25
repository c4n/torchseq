import json, jsonlines, sacrebleu
from torchseq.agents.para_agent import ParaphraseAgent
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config

from torchseq.agents.seq2seq_agent import Seq2SeqAgent

import torch

# Which checkpoint should we load?
path_to_model = '/ist-project/scads/can/disentaglement_projects/torchseq/runs/vae/20220920_232720_paraphrasing_vae_mscoco_789'

DATA_PATH = '../data/'

# Load the data
with jsonlines.open(DATA_PATH + 'mscoco-eval/test.jsonl') as f:
    rows = [row for row in f]

examples = [{'input': row['sem_input']} for row in rows]


# Change the config to use the custom dataset
with open(path_to_model + "/config.json") as f:
    cfg_dict = json.load(f)
cfg_dict["dataset"] = "json"
cfg_dict["json_dataset"] = {
    "path": None,
    "field_map": [
        {"type": "copy", "from": "input", "to": "target"},
        {"type": "copy", "from": "input", "to": "source"},
#         {"type": "copy", "from": "input", "to": "template"},
    ],
}
cfg_dict["beam_search"]["beam_width"] = 3 # set beam
# Enable the code predictor
# cfg_dict["bottleneck"]["code_predictor"]["infer_codes"] = True

# Create the dataset and model
config = Config(cfg_dict)

checkpoint_path = path_to_model + "/model/checkpoint.pt"
# instance = ParaphraseAgent(config=config, run_id=None,  output_path=None, data_path=DATA_PATH, silent=False, verbose=False, training_mode=False)
instance = Seq2SeqAgent(config=config, run_id=None,  output_path=None, data_path=DATA_PATH, silent=False, verbose=False, training_mode=False)
# Load the checkpoint
instance.load_checkpoint(checkpoint_path)
instance.model.eval()
    
# Finally, run inference
data_loader = JsonDataLoader(config, test_samples=examples, data_path=DATA_PATH)
encoding_pooled_list = instance.get_vector(data_loader.test_loader)
print(encoding_pooled_list)