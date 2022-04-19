from typing import Iterable
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

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
        x = self.tokenizer(emr, truncation=True).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        # target transform
        y = torch.LongTensor([dx])        
        return x, y

class DxBatchCollator(object):
    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        batch_x = list()
        batch_y = torch.LongTensor()

        for x, y in batch:
            batch_x.append(x)
            batch_y = torch.cat(tensors=(batch_y, y), dim=0)
        batch_x = self.tokenizer.pad(batch_x)

        return batch_x, batch_y

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

def split_by_div(data: Iterable, fold: int, remainder: int, mode: str) -> list:
    data_l = list()
    for i, item in enumerate(data):
        if mode == "train":
            if i % fold != remainder:
                data_l.append(item)
        elif mode == "val":
            if i % fold == remainder:
                data_l.append(item)
        else:
            raise ValueError("mode should be either train or val")
    return data_l