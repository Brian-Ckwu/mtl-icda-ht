import json
from multiprocessing.sharedctypes import Value
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, DataCollatorWithPadding
import pandas as pd
import numpy as np

class EMRDataset(Dataset):
    def __init__(self, emr_file, mode="train"):
        df = pd.read_csv(emr_file).dropna()
        data = get_text_label_pairs(df) # --> ndarray: (text, label)
        data = even_split_by_labels(data, mode) # --> train or val data
        self.x = data[:, 0]
        self.y = data[:, 1]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# load utility files
with open("./icd2idx.json") as f:
    icd2idx = json.load(f)

def get_text_label_pairs(df):
    pairs = df[["PRESENTILLNESS", "In_ICD"]]
    pairs.loc[:, "In_ICD"] = pairs.loc[:, "In_ICD"].apply(lambda icd: icd2idx[icd])
    return pairs.values

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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def bert_batch_collator(batch):
    texts, labels = list(), list()
    for text, label in batch:
        texts.append(text)
        labels.append(label)
    features = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels)
    return features, labels