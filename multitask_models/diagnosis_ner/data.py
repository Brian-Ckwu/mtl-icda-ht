import torch
from typing import Iterable
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

"""
    EMR Dataset for Diagnosis and Medical NER
        Input: EMRs (list of strings), diagnosis labels (list of indices), NER labels (sets of indices)
        Output: tokenized EMRs (BatchEncoding), diagnosis labels (index), NER labels (tensor of indices)
"""

class MTLDiagnosisNERDataset(Dataset):
    def __init__(self, emrs: list[str], dx_labels: list[int], ner_labels: list[set[int]], tokenizer: BertTokenizerFast, ignore_index: int = -100):
        # data
        self.x = emrs
        self.y_dx = dx_labels
        self.y_ner = ner_labels
        assert len(self.x) == len(self.y_dx) == len(self.y_ner) # check equal length
        # tokenizer
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # find the indexed data
        emr = self.x[idx]
        dx = self.y_dx[idx]
        ner_labels = self.y_ner[idx]
        # transform
        encoded_emr = self.tokenizer(emr, truncation=True, return_offsets_mapping=True, verbose=False).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        offset_mapping = encoded_emr.pop("offset_mapping")
        # target transform - dx
        y_dx = torch.LongTensor([dx])
        # target transform - ner
        y_ner = torch.LongTensor(bert_tokens_to_ner_labels(offset_mapping, target_indices=ner_labels, ignore_index=self.ignore_index))

        return encoded_emr, y_dx, y_ner

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

def split_by_div(data: Iterable, fold: int, mode: str) -> list:
    data_l = list()
    for i, item in enumerate(data):
        if mode == "train":
            if i % fold != 0:
                data_l.append(item)
        elif mode == "val":
            if i % fold == 0:
                data_l.append(item)
        else:
            raise ValueError("mode should be either train or val")
    return data_l

def bert_tokens_to_ner_labels(offset_mapping, target_indices: set[int], ignore_index: int = -100) -> list[int]:
    offsets = offset_mapping
    ner_labels = list()
    for offset in offsets:
        start = offset[0].item()
        end = offset[1].item()
        ner_label = 1 if (end - start) > 0 else ignore_index
        for i in range(start, end):
            if i not in target_indices:
                ner_label = 0
                break
        ner_labels.append(ner_label)
    
    return ner_labels