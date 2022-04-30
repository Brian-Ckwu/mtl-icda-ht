from typing import Iterable, List, Set, Tuple, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, AutoTokenizer

# Dataset classes
class MedicalDxDataset(Dataset):
    def __init__(self, emrs: List[str], dx_labels: List[int], tokenizer: BertTokenizerFast):
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

class MedicalNERDataset(Dataset):
    def __init__(self, emrs: List[str], labels: List[Set[int]], tokenizer: BertTokenizerFast, ignore_index: int = -100):
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

class MedicalNERIOBDataset(Dataset):
    def __init__(self, emrs: List[str], spans_tuples: List[List[Tuple[int]]], tokenizer: BertTokenizerFast, ignore_index: int = -100):
        assert len(emrs) == len(spans_tuples)
        self.emrs = emrs
        self.spans_tuples = spans_tuples
        self.span_dicts = [{span: 'B' for span in spans} for spans in spans_tuples] # for converting spans to IOB labels
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        # for label / index conversion
        self._iob2idx = {
            'O': 0,
            'B': 1,
            'I': 2,
            '[PAD]': ignore_index
        }
        self._idx2iob = {idx: label for label, idx in self._iob2idx.items()}
    
    def __len__(self):
        return len(self.emrs)
    
    def __getitem__(self, idx):
        emr = self.emrs[idx]
        spans = self.spans_tuples[idx]
        span_dict = self.span_dicts[idx].copy()
        # TODO: convert spans to IOB labels of BERT tokens
        emr_be = self.tokenizer(emr, truncation=True, return_offsets_mapping=True, verbose=False).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        offsets = emr_be.pop("offset_mapping").tolist()
        iob_labels = self.bert_offsets_to_iob_labels(offsets, spans, span_dict)
        return emr_be, iob_labels
    
    @property
    def iob2idx(self):
        return self._iob2idx
    
    @property
    def idx2iob(self):
        return self._idx2iob
    
    @property
    def num_tags(self):
        return len(self._iob2idx) - 1
    
    def bert_offsets_to_iob_labels(self, offsets: List[List[int]], spans: List[Tuple[int]], span_dict: Dict[tuple, int]):
        iob_labels = list()
        spans.append((int(1e8), int(1e8))) # add dummy span for convenience
        offsets_it = iter(offsets)
        spans_it = iter(spans)
        offset = next(offsets_it)
        span = next(spans_it)
        while True:
            start, end = offset
            if (start == 0) and (end == 0):  # [CLS] or [SEP] token
                label = '[PAD]'
            elif start < span[0]: # offset is to the left of span
                label = 'O'
            elif end <= span[1]: # in: i.e. start >= span[0] & end <= span[1]
                label = span_dict[span] # default 'B'
                span_dict[span] = 'I'
            else: # i.e. end > span[1]
                span = next(spans_it) # move to the next span
                continue
            # if the span is the same then execute the following codes
            iob_labels.append(label)
            try:
                offset = next(offsets_it)
            except StopIteration:
                break
        
        iob_labels = [self.iob2idx[label] for label in iob_labels]
        return iob_labels
    
    def collate_fn(self, batch):
        batch_x, batch_y = list(), list()
        for x, y in batch:
            batch_x.append(x)
            batch_y.append(y)

        batch_x = self.tokenizer.pad(batch_x)

        dim_after_padding = batch_x["input_ids"].shape[1]
        for i in range(len(batch_y)):
            to_fill_length = dim_after_padding - len(batch_y[i])
            padding = torch.ones(to_fill_length, dtype=torch.long) * self.ignore_index
            batch_y[i] = torch.cat((torch.LongTensor(batch_y[i]), padding), dim=0)
        batch_y = torch.stack(batch_y)

        return batch_x, batch_y

class MedicalIOBPOLDataset(Dataset):
    def __init__(
            self, 
            text_l: List[str], 
            ner_spans_l: List[Dict[Tuple, int]], 
            tokenizer: AutoTokenizer, 
            ignore_index: int = -100
        ):
        assert len(text_l) == len(ner_spans_l)

        self.text_l = text_l
        self.ner_spans_l = ner_spans_l
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        # for label / index conversion
        self.span_labels = {0: None, 1: "POS", 2: "NEG"}
        self.iob2idx = {
            "O": 0,
            "B-POS": 1,
            "I-POS": 2,
            "B-NEG": 3,
            "I-NEG": 4,
            "[PAD]": -100
        }
        self.idx2iob = {idx: label for label, idx in self.iob2idx.items()}
    
    def __len__(self):
        return len(self.text_l)
    
    def __getitem__(self, idx):
        # get x & y
        text = self.text_l[idx]
        span_pol_dict = self.ner_spans_l[idx]
        # convert span labels to token labels
        text_be = self.tokenizer(text, truncation=True, return_offsets_mapping=True).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        offsets = text_be.pop("offset_mapping").tolist()
        iob_labels = self.bert_offsets_to_iob_labels(offsets, span_pol_dict)
        return text_be, iob_labels
    
    @property
    def num_tags(self):
        return len(self.iob2idx) - 1
    
    def bert_offsets_to_iob_labels(self, offsets: List[List[int]], span_pol_dict: Dict[tuple, int]):
        seen_spans = set()
        iob_labels = list()
        span_pol_dict[(int(1e8), int(1e8))] = 0 # add dummy span for convenience
        span_pols = span_pol_dict.items()
        
        offsets_it = iter(offsets)
        span_pols_it = iter(span_pols)
        offset = next(offsets_it)
        span, pol = next(span_pols_it)
        while True:
            start, end = offset
            if (start == 0) and (end == 0):  # [CLS] or [SEP] token
                label = '[PAD]'
            elif start < span[0]: # offset is to the left of span
                label = 'O'
            elif end <= span[1]: # in: i.e. start >= span[0] & end <= span[1]
                if span not in seen_spans: # 'B'eginning of span
                    seen_spans.add(span)
                    label = f"B-{self.span_labels[pol]}"
                else: # 'I' span
                    label = f"I-{self.span_labels[pol]}"
            else: # i.e. end > span[1]
                span, pol = next(span_pols_it) # move to the next span
                continue
            # if the span is the same then execute the following codes
            iob_labels.append(label)
            try:
                offset = next(offsets_it)
            except StopIteration:
                break
        
        iob_labels = [self.iob2idx[label] for label in iob_labels]
        return iob_labels
    
    def collate_fn(self, batch):
        batch_x, batch_y = list(), list()
        for x, y in batch:
            batch_x.append(x)
            batch_y.append(y)

        batch_x = self.tokenizer.pad(batch_x)

        dim_after_padding = batch_x["input_ids"].shape[1]
        for i in range(len(batch_y)):
            to_fill_length = dim_after_padding - len(batch_y[i])
            padding = torch.ones(to_fill_length, dtype=torch.long) * self.ignore_index
            batch_y[i] = torch.cat((torch.LongTensor(batch_y[i]), padding), dim=0)
        batch_y = torch.stack(batch_y)

        return batch_x, batch_y

class MedicalDxNERIOBDataset(Dataset):
    def __init__(self, emrs: List[str], dx_labels: List[int], spans_tuples: List[List[Tuple[int]]], tokenizer: BertTokenizerFast, ignore_index: int = -100):
        assert len(emrs) == len(dx_labels) == len(spans_tuples)
        self.emrs = emrs
        self.dx_labels = dx_labels
        self.spans_tuples = spans_tuples
        self.span_dicts = [{span: 'B' for span in spans} for spans in spans_tuples] # for converting spans to IOB labels
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        # for label / index conversion
        self._iob2idx = {
            'O': 0,
            'B': 1,
            'I': 2,
            '[PAD]': ignore_index
        }
        self._idx2iob = {idx: label for label, idx in self._iob2idx.items()}
    
    def __len__(self):
        return len(self.emrs)
    
    def __getitem__(self, idx):
        emr = self.emrs[idx]
        dx = self.dx_labels[idx]
        spans = self.spans_tuples[idx]
        span_dict = self.span_dicts[idx].copy()
        # TODO: convert spans to IOB labels of BERT tokens
        emr_be = self.tokenizer(emr, truncation=True, return_offsets_mapping=True, verbose=False).convert_to_tensors(tensor_type="pt", prepend_batch_axis=False)
        offsets = emr_be.pop("offset_mapping").tolist()
        iob_labels = self.bert_offsets_to_iob_labels(offsets, spans, span_dict)
        return emr_be, dx, iob_labels
    
    @property
    def iob2idx(self):
        return self._iob2idx
    
    @property
    def idx2iob(self):
        return self._idx2iob
    
    @property
    def num_dx_labels(self):
        return len(Counter(self.dx_labels))

    @property
    def num_ner_labels(self):
        return len(self._iob2idx) - 1
    
    def bert_offsets_to_iob_labels(self, offsets: List[List[int]], spans: List[Tuple[int]], span_dict: Dict[tuple, int]):
        iob_labels = list()
        spans.append((int(1e8), int(1e8))) # add dummy span for convenience
        offsets_it = iter(offsets)
        spans_it = iter(spans)
        offset = next(offsets_it)
        span = next(spans_it)
        while True:
            start, end = offset
            if (start == 0) and (end == 0):  # [CLS] or [SEP] token
                label = '[PAD]'
            elif start < span[0]: # offset is to the left of span
                label = 'O'
            elif end <= span[1]: # in: i.e. start >= span[0] & end <= span[1]
                label = span_dict[span] # default 'B'
                span_dict[span] = 'I'
            else: # i.e. end > span[1]
                span = next(spans_it) # move to the next span
                continue
            # if the span is the same then execute the following codes
            iob_labels.append(label)
            try:
                offset = next(offsets_it)
            except StopIteration:
                break
        
        iob_labels = [self.iob2idx[label] for label in iob_labels]
        return iob_labels
    
    def collate_fn(self, batch):
        batch_x, batch_ydx, batch_yner = list(), list(), list()
        for x, ydx, yner in batch:
            batch_x.append(x)
            batch_ydx.append(ydx)
            batch_yner.append(yner)

        batch_x = self.tokenizer.pad(batch_x)
        batch_ydx = torch.LongTensor(batch_ydx)

        dim_after_padding = batch_x["input_ids"].shape[1]
        for i in range(len(batch_yner)):
            to_fill_length = dim_after_padding - len(batch_yner[i])
            padding = torch.ones(to_fill_length, dtype=torch.long) * self.ignore_index
            batch_yner[i] = torch.cat((torch.LongTensor(batch_yner[i]), padding), dim=0)
        batch_yner = torch.stack(batch_yner)

        return batch_x, batch_ydx, batch_yner

class MTLDiagnosisNERDataset(Dataset):
    def __init__(self, emrs: List[str], dx_labels: List[int], ner_labels: List[Set[int]], tokenizer: BertTokenizerFast, ignore_index: int = -100):
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
def convert_icds_to_indices(icds: List[str], full_code: bool = True) -> List[int]:
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
        elif mode == "valid":
            if i % fold == remainder:
                data_l.append(item)
        else:
            raise ValueError("mode should be either train or valid")
    return data_l

def bert_tokens_to_ner_labels(offset_mapping, target_indices, ignore_index: int = -100) -> List[int]:
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