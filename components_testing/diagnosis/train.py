"""
    Import Packages
"""
import sys
sys.path.append("../../")

import json
import pickle
from typing import Tuple
from pathlib import Path
from argparse import Namespace
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utilities.data import MedicalDxDataset, DxBatchCollator, convert_icds_to_indices, split_by_div
from utilities.model import BertDxModel, encoder_names_mapping
from utilities.utils import set_seeds, render_exp_name, move_bert_input_to_device

import warnings
warnings.filterwarnings("ignore")

def trainer(args: Namespace):
    # Configuration
    # save args
    args.ckpt_path = Path(args.save_dir) / args.exp_name
    args.ckpt_path.mkdir(parents=True, exist_ok=True)
    (args.ckpt_path / "args.pickle").write_bytes(pickle.dumps(args))
    if "cuda" in args.device:
        assert torch.cuda.is_available()

    set_seeds(args.seed)
    print("Finish configuration setup.")

    # Data
    # load data
    emrs = pickle.loads(Path(args.emr_path).read_bytes())
    icds = pickle.loads(Path(args.dx_path).read_bytes())
    dx_labels = convert_icds_to_indices(icds, full_code=args.fc)
    print("Finish data loading.")

    # train / valid split
    data_l = [emrs, dx_labels]
    train_emrs, train_dxs = [split_by_div(data, args.fold, args.remainder, "train") for data in data_l]
    valid_emrs, valid_dxs = [split_by_div(data, args.fold, args.remainder, "valid") for data in data_l]
    print("Finish train / valid split.")

    # dataset
    tokenizer = AutoTokenizer.from_pretrained(encoder_names_mapping[args.tokenizer], use_fast=True)
    train_set = MedicalDxDataset(emrs=train_emrs, dx_labels=train_dxs, tokenizer=tokenizer)
    valid_set = MedicalDxDataset(emrs=valid_emrs, dx_labels=valid_dxs, tokenizer=tokenizer)
    print(f"Finish dataset construction.")

    # dataloader
    batch_collator = DxBatchCollator(tokenizer)
    train_loader = DataLoader(train_set, args.bs // args.grad_accum_steps, shuffle=True, pin_memory=True, collate_fn=batch_collator)
    valid_loader = DataLoader(valid_set, args.bs // args.grad_accum_steps, shuffle=False, pin_memory=True, collate_fn=batch_collator)
    print(f"Finish dataloader construction.")

    # Model
    model = BertDxModel(
        encoder_name=encoder_names_mapping[args.encoder], 
        num_dxs=len(Counter(dx_labels))
    ).to(args.device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    tokenizer.model_max_length = model.bert.embeddings.position_embeddings.num_embeddings 
    print(f"Finish model construction. Max sequence length = {tokenizer.model_max_length}")

    # Optimization
    print(f"Staring training...")
    train_log = {
        "acc": list(),
        "loss": list(),
        "steps": list()
    }
    best_acc = 0
    nsteps = 0
    for epoch in range(args.nepochs):
        print(f"\n===== Training at epoch {epoch + 1} =====\n")
        for loader_idx, batch in enumerate(train_loader):
            eval_flag = False
            x, y_dx = batch
            model.train()

            # move to device
            x = move_bert_input_to_device(x, args.device)
            y_dx = y_dx.to(args.device)

            # forward & backward
            scores = model(x)
            loss = model.calc_loss(scores, y_dx) / args.grad_accum_steps
            loss.backward()

            # update model parameters
            if (loader_idx % args.grad_accum_steps == 0) or (loader_idx == len(train_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                nsteps += 1
                eval_flag = True

            if eval_flag and ((nsteps % args.ckpt_steps == 0) or (loader_idx == len(train_loader) - 1)):
                print(f"Evaluating model at step {nsteps} -> ", end='')
                acc, loss = evaluate(valid_loader, model, args)
                print(f"ACC = {acc:.3f}; loss = {loss:.3f}")
                # update train_log
                for k, v in zip(["acc", "loss", "steps"], [acc, loss, nsteps]):
                    train_log[k].append(v)
                Path(args.ckpt_path / "train_log.json").write_text(json.dumps(train_log))
                
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), args.ckpt_path / "best_model.pth")
                    (args.ckpt_path / "best_metric.txt").write_text(str(best_acc))
                    print("Best model saved.")

        print(f"===== Finish training epoch {epoch + 1} =====")

    return best_acc

def evaluate(data_loader: DataLoader, model: BertDxModel, args: Namespace) -> Tuple[float, float]:
    ncorrect = 0
    total_loss = 0
    model.eval()
    for x, y in data_loader:
        x = move_bert_input_to_device(x, args.device)
        y = y.to(args.device)
        
        with torch.no_grad():
            scores = model(x)
            loss = model.calc_loss(scores, y)

        ncorrect += (scores.argmax(dim=-1) == y).sum().detach().cpu().item()
        total_loss += loss.detach().cpu().item() * len(y)

    acc = ncorrect / len(data_loader.dataset)
    mean_loss = total_loss / len(data_loader.dataset)
    return acc, mean_loss

if __name__ == "__main__":
    config = json.loads(Path("./config.json").read_bytes())
    args = Namespace(**config)
    assert args.bs % args.grad_accum_steps == 0

    for r in range(1, 10):
        args.remainder = r
        args.exp_name = render_exp_name(args, hparams=["encoder", "fc", "optimizer", "lr", "nepochs", "bs", "fold", "remainder"])
        print(f"Experiment details: {args.exp_name}")
        best_acc = trainer(args)
        print(f"\n*** Finish training (fold = {args.remainder}): best_acc = {best_acc:.4f} ***\n")