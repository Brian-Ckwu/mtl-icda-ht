"""
    Import Packages
"""
import warnings
import torch
import torch.nn as nn
import gc
import json
import jsonlines
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from model import BertTokenClassification
from data import MedicalNERDataset, BertBatchCollator, split_by_div
from utils import *

warnings.filterwarnings("ignore")

"""
    Configuration
"""
with open("./config.json") as f:
    config = json.load(f)

assert torch.cuda.is_available()
device = "cuda:0"

same_seeds(config["seed"])

for remainder in range(config["fold"]):
    config["model_save_name"] = f"nepochs-{config['n_epochs']}_fold-{config['fold']}_remainder-{remainder}"
    print(f"\n\n*** Training schedule {remainder + 1} ***")
    """
        Data
    """
    # load data
    # x: EMR
    emr_file = "/nfs/nas-7.1/ckwu/datasets/emr/6000/docs_0708.json"
    emrs = list()
    with jsonlines.open(emr_file) as f:
        for doc in f:
            if doc["annotations"]: # important
                emrs.append(doc["text"])
    # y: NER labels
    label_file = "/nfs/nas-7.1/ckwu/datasets/emr/6000/ner_labels_in_indices.txt"
    whole_indices = list()
    with open(label_file) as f:
        for line in f:
            indices = set(map(lambda i: int(i), line.split()))
            whole_indices.append(indices)

    # train / val split
    x_train, y_train = [split_by_div(data, config["fold"], remainder, "train") for data in [emrs, whole_indices]]
    x_val, y_val = [split_by_div(data, config["fold"], remainder, "val") for data in [emrs, whole_indices]]
    print(f"Training samples: {len(x_train)}; Validation samples: {len(x_val)}\n\n")

    # prepare dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataset = MedicalNERDataset(emrs=x_train, labels=y_train, tokenizer=tokenizer, ignore_index=-100)
    val_dataset = MedicalNERDataset(emrs=x_val, labels=y_val, tokenizer=tokenizer, ignore_index=-100)

    # prepare dataloader
    bert_batch_collator = BertBatchCollator(tokenizer, ignore_index=-100)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, collate_fn=bert_batch_collator)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, collate_fn=bert_batch_collator)

    """
        Model
    """
    model = BertTokenClassification(**config["model_config"]).to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    """
        Optimization
    """
    record = trainer(train_loader, val_loader, model, criterion, config, device)
    # save evaluation results
    with open("./eval_results/{}.json".format(config["model_save_name"]), "wt") as f:
        json.dump(record, f)

    ## collect garbage
    del x_train, y_train, x_val, y_val, train_dataset, val_dataset, train_loader, val_loader, model, record
    gc.collect()