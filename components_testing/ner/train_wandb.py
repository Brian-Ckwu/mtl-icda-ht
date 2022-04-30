import warnings
import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

import gc
import json
import pickle
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from argparse import Namespace
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from utilities.data import MedicalNERIOBDataset, split_by_div
from utilities.utils import set_seeds, load_config, render_exp_name, move_bert_input_to_device
from utilities.model import BertNERModel, encoder_names_mapping
from utilities.evaluation import ids_to_iobs, calc_seqeval_metrics

warnings.filterwarnings("ignore")
MODEL_MAX_LENGTH = 512

def trainer(args: Namespace):
    print(f"Start training with args:\n{args}")
    # Configuration
    args.ckpt_path = Path(args.save_dir) / args.exp_name
    args.ckpt_path.mkdir(parents=True, exist_ok=True)
    (args.ckpt_path / "args.pickle").write_bytes(pickle.dumps(args))

    set_seeds(args.seed)
    print("Finish configuration setup.")

    # Data
    emrs = pickle.loads(Path(args.emr_path).read_bytes())
    ner_spans_l = pickle.loads(Path(args.ner_spans_l_path).read_bytes())
    print("Finish data loading.")

    # train/val split
    train_emrs, train_labels = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="train") for data in [emrs, ner_spans_l]]
    valid_emrs, valid_labels = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="valid") for data in [emrs, ner_spans_l]]
    print("Finish train / valid split.")

    # dataset
    tokenizer = BertTokenizerFast.from_pretrained(encoder_names_mapping[args.tokenizer])
    train_set = MedicalNERIOBDataset(emrs=train_emrs, spans_tuples=train_labels, tokenizer=tokenizer)
    valid_set = MedicalNERIOBDataset(emrs=valid_emrs, spans_tuples=valid_labels, tokenizer=tokenizer)
    print(f"Finish dataset construction.")

    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.bs // args.grad_accum_steps, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.bs // args.grad_accum_steps, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)
    print(f"Finish dataloader construction.")

    # Model, Loss, Optimizer, and Scheduler
    model = BertNERModel(encoder=encoder_names_mapping[args.encoder], num_tags=train_set.num_tags).to(args.device)
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=train_set.ignore_index)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    print(f"Finish model construction. Max sequence length = {tokenizer.model_max_length}")

    # Optimization
    print(f"Staring training...")
    # trackers
    train_log = {
        "token_acc": list(),
        "precision": list(),
        "recall": list(),
        "f1-score": list(),
        "valid_loss": list(),
        "step": list()
    }
    best_token_acc = 0
    whole_batch_loss = 0
    step = 0
    for epoch in range(args.nepochs):
        print(f"\n===== Training at epoch {epoch + 1} =====\n")
        for loader_idx, batch in enumerate(train_loader):
            eval_flag = False
            x, y_ner = batch
            model.train()

            # move data
            x = move_bert_input_to_device(x, args.device)
            y_ner = y_ner.to(args.device)

            # forward & backward pass
            scores = model(x)
            loss = criterion(scores.transpose(1, 2), y_ner) / args.grad_accum_steps
            loss.backward()
            whole_batch_loss += loss.detach().cpu().item()

            # update model parameters
            if (loader_idx % args.grad_accum_steps == args.grad_accum_steps - 1) or (loader_idx == len(train_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                eval_flag = True
                wandb.log({"train_loss": whole_batch_loss})
                whole_batch_loss = 0

            # evaluate model at ckpt_steps and end of epoch
            if eval_flag and ((step % args.ckpt_steps == 0) or (loader_idx == len(train_loader) - 1)):
                print(f"Evaluating model at step {step} -> ")
                y_pred_raw, y_true_raw, valid_loss = predict_whole_set_ner(valid_loader, model, criterion, args.device)
                y_pred, y_true = ids_to_iobs(y_pred_raw, y_true_raw, valid_set)
                token_acc, p, r, f1 = calc_seqeval_metrics(y_true, y_pred)
                print(f"\bToken_acc = {token_acc:.3f}; Precision = {p:.3f}; Recall = {r:.3f}; F1-score = {f1:.3f}")

                # update train_log
                for k, v in zip(list(train_log.keys()), [token_acc, p, r, f1, valid_loss, step]):
                    train_log[k].append(v)
                Path(args.ckpt_path / "train_log.json").write_text(json.dumps(train_log))

                # update wandb
                wandb.log({k: v for k, v in zip(list(train_log.keys()), [token_acc, p, r, f1, valid_loss, step])})

                # update best metrics
                if token_acc > best_token_acc:
                    best_token_acc = token_acc
                    torch.save(model.state_dict(), args.ckpt_path / "best_model.pth")
                    (args.ckpt_path / "best_metric.txt").write_text(str(best_token_acc))
                    print("Best model saved.")
        
        print(f"===== Finish training epoch {epoch + 1} =====")
            
    wandb.run.summary["best_token_acc"] = best_token_acc
    wandb.finish(exit_code=0)
    return best_token_acc

def predict_whole_set_ner(data_loader: DataLoader, model: BertNERModel, criterion: nn.CrossEntropyLoss, device: str) -> Tuple[list, list, float]:
    y_pred_raw = list()
    y_true_raw = list()
    total_loss = 0

    model.to(device)
    model.eval()
    for x, y_ner in data_loader:
        x = move_bert_input_to_device(x, device)
        y_ner = y_ner.to(device)
        
        with torch.no_grad():
            scores = model(x)
            loss = criterion(scores.transpose(1, 2), y_ner)

        # record loss
        total_loss += loss.detach().cpu().item() * y_ner.shape[0]
        # record predictions
        pred = scores.argmax(dim=-1).detach().cpu().tolist()
        y_pred_raw.append(pred)
        # record ground truth
        true = y_ner.detach().cpu().tolist()
        y_true_raw.append(true)

    mean_loss = total_loss / len(data_loader.dataset)
    return y_pred_raw, y_true_raw, mean_loss

if __name__ == "__main__":
    config = load_config()
    args = Namespace(**config)
    hparams = ["encoder", "optimizer", "lr", "nepochs", "bs", "fold", "remainder"]
    assert args.bs % args.grad_accum_steps == 0

    if args.wandb:
        import wandb
    
    for lr in [2e-5, 3e-5, 4e-5]:
        for r in range(10):
            args.remainder = r
            args.exp_name = render_exp_name(args, hparams)
            wandb.init(
                project="icda_ner",
                name=f"{args.encoder}_lr-{args.lr}_r-{args.remainder}",
                config={hparam: getattr(args, hparam) for hparam in hparams}
            )

            print(f"Experiment details: {args.exp_name}")
            best_metric = trainer(args)
            print(f"\n*** Finish training (fold = {args.remainder}): best metric = {best_metric:.4f} ***\n")