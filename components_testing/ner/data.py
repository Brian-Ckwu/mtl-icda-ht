import torch
from typing import Iterable
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

# input: emrs (strings), labels (sets of indices)
# output: emrs (tokenized tensors), labels (NER tensors)
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

class BertBatchCollator(object):
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