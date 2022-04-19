"""
    Import Packages
"""
import warnings
import sys
import gc
import json
import jsonlines
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import BERTClassification
from data import MedicalDiagnosisDataset, DxBatchCollator, convert_icds_to_indices, split_by_div
from utils import *

warnings.filterwarnings("ignore")

"""
    Configuration
"""
with open("./config.json") as f:
    config = json.load(f)
args = Namespace(**config)

assert torch.cuda.is_available()
device = args.device

same_seeds(args.seed)

"""
    Training Loop of Target Hyper-Parameters
"""
def render_model_name(encoder_type, config):
    config_fields = [
        f"encoder-{encoder_type}",
        f"dx-{config['model_config']['label_size']}",
        f"lr-{config['optimizer_hparams']['lr']}"
    ]
    return '_'.join(config_fields)

for remainder in range(10):
    config["model_save_name"] = f"{render_model_name('Bio_ClinicalBERT', config)}_remainder-{remainder}"
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
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
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
