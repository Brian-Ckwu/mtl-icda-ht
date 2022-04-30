from turtle import update
import warnings
import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

import numpy as np
import torch
import torch.nn as nn
import random
import json
import pickle
import gc

from argparse import Namespace
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from utilities.data import MedicalNERIOBDataset, split_by_div
from utilities.utils import set_seeds, load_config, render_exp_name, move_bert_input_to_device
from utilities.model import BertNERModel, encoder_names_mapping

warnings.filterwarnings("ignore")

def trainer(args: Namespace):
    print(f"Start training with args:\n{args}")
    # Configuration
    set_seeds(args.seed)

    # Data
    # x: EMR
    emr_path = Path(args.emr_path)
    emrs = pickle.loads(emr_path.read_bytes())
    # y: NER labels
    spans_tuples_path = Path(args.ner_spans_tuples_path)
    spans_tuples = pickle.loads(spans_tuples_path.read_bytes())
    # train/val split
    train_emrs, train_labels = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="train") for data in [emrs, spans_tuples]]
    valid_emrs, valid_labels = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="valid") for data in [emrs, spans_tuples]]
    # dataset
    tokenizer = BertTokenizerFast.from_pretrained(encoder_names_mapping[args.tokenizer])
    train_set = MedicalNERIOBDataset(emrs=train_emrs, spans_tuples=train_labels, tokenizer=tokenizer)
    valid_set = MedicalNERIOBDataset(emrs=valid_emrs, spans_tuples=valid_labels, tokenizer=tokenizer)
    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.bs, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)

    # Model, Loss, Optimizer, and Scheduler
    model = BertNERModel(encoder=encoder_names_mapping[args.encoder], num_tags=train_set.num_tags).to(args.device)
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=train_set.ignore_index)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # Optimization
    # trackers
    record = {"acc": list(), "loss": list()}
    best_val_acc = 0
    step = 0
    for epoch in range(1, args.nepochs + 1):
        for x, y in train_loader:
            model.train()
            # move data
            x = move_bert_input_to_device(x, args.device)
            y = y.to(args.device)
            # inference
            scores = model(x)
            loss = criterion(scores.transpose(1, 2), y)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evaluate model at every ckpt
            if step % args.ckpt_steps == 0:
                print(f"Evaluating model at step {step}...")
                best_val_acc = update_evaluation(valid_loader, model, criterion, args, record, best_val_acc)
            step += 1
        
        # evaluate model at each epoch
        print(f"===== Evaluating model at epoch {epoch} =====")
        best_val_acc = update_evaluation(valid_loader, model, criterion, args, record, best_val_acc)

    record["best_val_acc"] = best_val_acc
    return best_val_acc, record


def update_evaluation(data_loader, model, criterion, args, record, best_acc):
    # utility function
    def update_record(record, acc, loss):
        record["acc"].append(acc)
        record["loss"].append(loss)
        return record
    # update metrics
    acc = evaluate_model_acc(data_loader, model, args.device)
    loss = evaluate_model_loss(data_loader, model, criterion, args.device)
    record = update_record(record, acc, loss)
    print(f"Acc: {acc:.4f} / Loss: {loss:.4f}")
    if acc > best_acc:
        best_acc = acc
        if args.exp_name:
            torch.save(model.state_dict(), "./models/{}.pth".format(args.exp_name))
            print("Best model saved.")

    return best_acc

def evaluate_model_loss(data_loader, model, criterion, device):
    model.eval()
    total_val_loss = 0

    for x, y in data_loader:
        # move data to device
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x).transpose(1, 2) # transpose for calculating cross entropy loss
            loss = criterion(pred, y)
        total_val_loss += loss.detach().cpu().item() * y.shape[0]
    
    mean_val_loss = total_val_loss / len(data_loader.dataset)
    return mean_val_loss

def evaluate_model_acc(data_loader, model, device):
    total_tokens = 0
    total_correct = 0
    
    model.eval()
    for x, y in data_loader:
        # inference
        x = move_bert_input_to_device(x, device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x) # no need to transpose in this case
        # calculate target metric (acc)
        total_tokens += (y != -100).sum().cpu().item()
        total_correct += (pred.argmax(dim=-1) == y).sum().cpu().item()
    
    acc = total_correct / total_tokens
    return acc

if __name__ == "__main__":
    config = load_config()
    args = Namespace(**config)
    if "cuda" in args.device:
        assert torch.cuda.is_available()
    
    remainders = range(args.fold) # 10
    for remainder in remainders:
        args.remainder = remainder
        args.exp_name = render_exp_name(args, hparams=["encoder", "nepochs", "bs", "lr", "fold", "remainder"])
        best_metric, train_log = trainer(args)
        # save best metric
        metric_path = Path(f"./eval_results/{args.exp_name}.txt")
        metric_path.write_text(data=str(best_metric))
        gc.collect()