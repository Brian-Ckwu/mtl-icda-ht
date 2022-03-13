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

from model import BERTClassification
from data import MedicalDiagnosisDataset, DxBatchCollator, convert_icds_to_indices, split_by_div
from utils import *

warnings.filterwarnings("ignore")

"""
    Configuration
"""
with open("./config.json") as f:
    config = json.load(f)

assert torch.cuda.is_available()
device = "cuda"

same_seeds(config["seed"])

"""
    Training Loop of Target Hyper-Parameters
"""
for remainder in range(5):
    config["model_save_name"] = "labelsize-{}_remainder-{}".format(config["model_config"]["label_size"], remainder)
    """
        Data
    """
    emr_file = "/nfs/nas-7.1/ckwu/datasets/emr/6000/docs_0708.json"
    emrs, dxs = list(), list()
    with jsonlines.open(emr_file) as f:
        for doc in f:
            if doc["annotations"]: # leave out un-annotated files
                emrs.append(doc["text"])
                dxs.append(doc["ICD"])

    dx_labels = convert_icds_to_indices(dxs, full_code=False)
    del dxs

    # train/val split
    x_train, y_train = [split_by_div(data, config["fold"], remainder, mode="train") for data in [emrs, dx_labels]]
    x_val, y_val = [split_by_div(data, config["fold"], remainder, mode="val") for data in [emrs, dx_labels]]

    # make dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataset = MedicalDiagnosisDataset(x_train, y_train, tokenizer)
    val_dataset = MedicalDiagnosisDataset(x_val, y_val, tokenizer)

    # make dataloader
    batch_collator = DxBatchCollator(tokenizer)
    train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True, pin_memory=True, collate_fn=batch_collator)
    val_loader = DataLoader(val_dataset, config["batch_size"], shuffle=False, pin_memory=True, collate_fn=batch_collator)
    del x_train, y_train, x_val, y_val
    gc.collect()

    """
        Model
    """
    model = BERTClassification(**config["model_config"]).to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    """
        Optimization
    """
    record = trainer(train_loader, val_loader, model, criterion, config, device)
    # save eval result
    with open("./eval_results/{}.json".format(config["model_save_name"]), mode="wt") as f:
        json.dump(record, f)
