from typing import Iterable
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

# Dataset classes
class MedicalDxDataset(Dataset):
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
        y = torch.LongTensor([dx])        
        return x, y

class MedicalNERDataset(Dataset):
    def __init__(self, emrs: list[str], labels: list[set[int]], tokenizer: BertTokenizerFast, ignore_index: int = -100):
        self.x = emrs
        self.y = labels
        assert len(self.x) == len(self.y)
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        emr = self.x[idx]
        label = self.y[idx]
        # transform
        tokenized_x = self.tokenizer(emr, truncation=True, return_offsets_mapping=True, verbose=False).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        offset_mapping = tokenized_x.pop("offset_mapping")
        # target transform
        ner_y = torch.LongTensor(bert_tokens_to_ner_labels(offset_mapping, label, self.ignore_index))
        return tokenized_x, ner_y

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

# Batch collators
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

class NERBatchCollator(object):
    def __init__(self, tokenizer, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
    
    def __call__(self, batch):
        batch_x, batch_y = list(), list()

        for x, y in batch:
            batch_x.append(x)
            batch_y.append(y)

        batch_x = self.tokenizer.pad(batch_x)

        dim_after_padding = batch_x["input_ids"].shape[1]
        for i in range(len(batch_y)):
            to_fill_length = dim_after_padding - batch_y[i].shape[0]
            padding = torch.ones(to_fill_length, dtype=torch.long) * self.ignore_index
            batch_y[i] = torch.cat((batch_y[i], padding), dim=0)
        batch_y = torch.stack(batch_y)

        return batch_x, batch_y

class DxNERBatchCollator(object):
    def __init__(self, tokenizer: BertTokenizerFast, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
    
    def __call__(self, batch):
        batch_x = list()
        batch_y_dx = list()
        batch_y_ner = list()

        # make the batch
        for x, y_dx, y_ner in batch:
            batch_x.append(x)
            batch_y_dx.append(y_dx.cpu().detach().item())
            batch_y_ner.append(y_ner)
        
        # handle x & y_dx
        batch_x = self.tokenizer.pad(batch_x)
        batch_y_dx = torch.LongTensor(batch_y_dx)

        # handle y_ner
        dim_after_padding = batch_x["input_ids"].shape[1]
        for i in range(len(batch_y_ner)):
            to_fill_length = dim_after_padding - batch_y_ner[i].shape[0]
            padding = torch.ones(to_fill_length, dtype=torch.long) * self.ignore_index
            batch_y_ner[i] = torch.cat((batch_y_ner[i], padding), dim=0)

        batch_y_ner = torch.stack(batch_y_ner)

        return batch_x, batch_y_dx, batch_y_ner

# Utility functions
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

def bert_tokens_to_ner_labels(offset_mapping, target_indices, ignore_index: int = -100) -> list[int]:
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