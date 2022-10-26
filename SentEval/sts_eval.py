# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# Paper: https://arxiv.org/pdf/1803.05449.pdf
# Github: https://github.com/facebookresearch/SentEval
import sys
import io
import logging
import numpy as np

# import SentEval
sys.path.insert(0, '..')
import senteval

class STSEval():
    def __init__(self, proc_vecstors, path_data) -> None:
        self.path_data = path_data
        self.proc_vecstors = proc_vecstors
        # Set params for SentEval
        self.params_senteval = {
            'task_path': self.path_data, 'usepytorch': True, 'kfold': 5}
        self.params_senteval['classifier'] = {
            'nhid': 0, 'optim': 'rmsprop', 'batch_size': 8, 
            'tenacity': 3, 'epoch_size': 2 }
        # Set up logger
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
        self.se = senteval.engine.SE(self.params_senteval, self.batcher, self.prepare)

    def __call__(self, transfer_tasks):
        return self.se.eval(transfer_tasks)

    # SentEval prepare and batcher
    def prepare(self, params, samples):
        pass

    def batcher(self, params, batch):
        batch = [" ".join(tokens) for tokens in batch]
        return self.proc_vecstors(batch).cpu().numpy()