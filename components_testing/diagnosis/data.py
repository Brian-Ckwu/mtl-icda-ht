import json
from multiprocessing.sharedctypes import Value
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import pandas as pd
import numpy as np

class MedicalDiagnosisDataset(Dataset):
    def __init__(self, emrs: list[str], dx_labels: list[int], tokenizer: BertTokenizerFast):
        self.x = emrs
        self.y = dx_labels
        assert len(self.x) == len(self.y)
        self.tokenizer=tokenizer
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # find the indexed data
        emr = self.x[idx]
        dx = self.y[idx]
        # transform
        x = self.tokenizer(emr, truncation=True, verbose=False).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        # target transform
        y = torch.LongTensor(dx)        
        return x, y

def even_split_by_labels(data, mode):
    data_l = list()
    for i, row in enumerate(data):
        if mode == "train":
            if i % 5 != 0:
                data_l.append(row)
        elif mode == "val":
            if i % 5 == 0:
                data_l.append(row)
        else:
            raise ValueError("mode should be either train or val")
    return np.array(data_l)