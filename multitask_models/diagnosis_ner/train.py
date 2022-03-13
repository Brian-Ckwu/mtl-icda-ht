"""
    Import Packages
"""
import warnings
import torch
import gc
import json
import jsonlines
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from model import BERTDiagnosisNER
from data import MTLDiagnosisNERDataset, DxNERBatchCollator, convert_icds_to_indices, split_by_div
from utils import *

warnings.filterwarnings("ignore")

"""
    Configuration
"""
config_file = "./config.json"
with open(config_file) as f:
    config = json.load(f)

assert torch.cuda.is_available()
device = "cuda"

same_seeds(config["seed"])

"""
    Training Loop
"""

for remainder in range(5):
    print(f"\n-------------- Training schedule {remainder + 1} --------------\n")
    config["model_save_name"] = "mtl_lw-1.0-8.0_remainder-{}".format(remainder)
    """
        Data
    """
    # EMRs & Diagnosis
    emr_file = "/nfs/nas-7.1/ckwu/datasets/emr/6000/docs_0708.json"
    emrs, dxs = list(), list()
    with jsonlines.open(emr_file) as f:
        for doc in f:
            if doc["annotations"]: # leave out un-annotated files
                emrs.append(doc["text"])
                dxs.append(doc["ICD"])

    dx_labels = convert_icds_to_indices(dxs, full_code=False)
    del dxs
    # NER labels
    label_file = "/nfs/nas-7.1/ckwu/datasets/emr/6000/ner_labels_in_indices.txt"
    whole_indices = list()
    with open(label_file) as f:
        for line in f:
            indices = set(map(lambda i: int(i), line.split()))
            whole_indices.append(indices)

    # train / val split
    x_train, y_dx_train, y_ner_train = [split_by_div(data, config["fold"], remainder, mode="train") for data in [emrs, dx_labels, whole_indices]]
    x_val, y_dx_val, y_ner_val = [split_by_div(data, config["fold"], remainder, mode="val") for data in [emrs, dx_labels, whole_indices]]
    del emrs, dx_labels, whole_indices

    # make dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataset = MTLDiagnosisNERDataset(x_train, y_dx_train, y_ner_train, tokenizer, ignore_index=-100)
    val_dataset = MTLDiagnosisNERDataset(x_val, y_dx_val, y_ner_val, tokenizer, ignore_index=-100)

    # make dataloader
    batch_collator = DxNERBatchCollator(tokenizer, ignore_index=-100)
    train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True, pin_memory=True, collate_fn=batch_collator)
    val_loader = DataLoader(val_dataset, config["batch_size"], shuffle=False, pin_memory=True, collate_fn=batch_collator)

    del x_train, y_dx_train, y_ner_train, x_val, y_dx_val, y_ner_val
    gc.collect()

    """
        Model
    """
    model = BERTDiagnosisNER(**config["model_config"]).to(device)

    """
        Optimization
    """
    record = trainer(train_loader, val_loader, model, config, device)
    # save evaluation results
    with open("./eval_results/{}.json".format(config["model_save_name"]), mode="wt") as f:
        json.dump(record, f)

    del model, record
    gc.collect()

