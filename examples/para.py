import json
from torchseq.agents.para_agent import ParaphraseAgent
# from torchseq.agents.aq_agent import AQAgent
from torchseq.agents.seq2seq_agent import Seq2SeqAgent
from torchseq.datasets.json_loader import JsonDataLoader
from torchseq.utils.config import Config
from torchseq.metric_hooks.textual import TextualMetricHook
import torch

model_path = '/raid/can/disentaglement_projects/torchseq/runs/examples/20220901_125037_paraphrasing_vae/'
# model_path = '../models/examples/20210503_184659_paraphrasing_vqvae/'
# model_path = '../models/examples/20210225_112226_paraphrasing_ae/'


# Load the config
with open(model_path + 'config.json') as f:
    cfg_dict = json.load(f)
cfg_dict["env"]["data_path"] = "../data/"


config = Config(cfg_dict)

# Load the model
instance = Seq2SeqAgent(config=config, run_id=None, output_path="./runs/examples/paraphrasing_eval", silent=False, verbose=False, training_mode=False, data_path='/raid/can/disentaglement_projects/torchseq/data')
# instance = AQAgent(config=config, run_id=None, output_path="./runs/examples/paraphrasing_eval", silent=False, verbose=False, training_mode=False, data_path='/raid/can/disentaglement_projects/torchseq/data')
instance.load_checkpoint(model_path + 'model/checkpoint.pt')
instance.model.eval()

# Create a dataset
data_loader = JsonDataLoader(config)

cfg_dict["json_dataset"] = {
    "path": None,
    "field_map": [
            {
                "type": "copy",
                "from": "q",
                "to": "source"
            },
            {
                "type": "copy",
                "from": "q",
                "to": "target"
            }
        ]
}

config = Config(cfg_dict)

# Run inference on the test split
test_loss, all_metrics, (pred_output, gold_output, gold_input), memory_values_to_return = instance.inference(data_loader.test_loader, metric_hooks=[TextualMetricHook(config, instance.input_tokenizer,'source', 'target')])

# Done!
print(all_metrics)
############

examples = [
    {'q': 'Who was the oldest cat in the world?'},
]



data_loader_custom = JsonDataLoader(config, test_samples=examples)

test_loss, all_metrics, (pred_output, gold_output, gold_input), memory_values_to_return = instance.inference(data_loader_custom.test_loader)

print(pred_output)
