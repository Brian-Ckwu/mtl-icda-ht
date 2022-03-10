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

def convert_icds_to_indices(icds: list[str], full_code: bool = True) -> list[int]:
    # utility functions
    def get_converted_icd(icd):
        return str(icd) if full_code else str(icd)[:3]

    def get_icd_idx_mapping():
        icd2idx = dict()
        for icd in icds:
            icd = get_converted_icd(icd)
            if icd not in icd2idx:
                icd2idx[icd] = len(icd2idx)
        return icd2idx
    
    # conversion
    icd2idx = get_icd_idx_mapping()
    indices = list()
    for icd in icds:
        icd = get_converted_icd(icd)
        idx = icd2idx[icd]
        indices.append(idx)

    return indices

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