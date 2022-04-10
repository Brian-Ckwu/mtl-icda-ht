import sys
sys.path.append("../../")

import gc
import json
import pickle
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from utilities.data import MedicalDxNERIOBDataset, convert_icds_to_indices, split_by_div
from utilities.utils import set_seeds, render_exp_name, move_bert_input_to_device
from utilities.model import BertDxNERModel, encoder_names_mapping

def trainer(args: Namespace):
    # Configuration
    # save args
    args.ckpt_path = Path(args.save_dir) / args.exp_name
    args.ckpt_path.mkdir(parents=True, exist_ok=True)
    (args.ckpt_path / "args.pickle").write_bytes(data=pickle.dumps(args))
    # fix seed
    set_seeds(args.seed)
    # check device
    if "cuda" in args.device:
        assert torch.cuda.is_available()

    # Data
    # x: EMRs
    emrs = pickle.loads(Path(args.emr_path).read_bytes())
    # y: Dx
    icds = pickle.loads(Path(args.dx_path).read_bytes())
    dx_labels = convert_icds_to_indices(icds, full_code=args.fc)
    # y: NER
    spans_tuples = pickle.loads(Path(args.ner_spans_tuples_path).read_bytes())
    
    # train / valid split
    data_l = [emrs, dx_labels, spans_tuples]
    train_emrs, train_dxs, train_ners = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="train") for data in data_l]
    valid_emrs, valid_dxs, valid_ners = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="valid") for data in data_l]

    # dataset and dataloader
    tokenizer = BertTokenizerFast.from_pretrained(encoder_names_mapping[args.tokenizer])
    train_set = MedicalDxNERIOBDataset(train_emrs, train_dxs, train_ners, tokenizer)
    valid_set = MedicalDxNERIOBDataset(valid_emrs, valid_dxs, valid_ners, tokenizer)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.bs, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)

    del emrs, icds, dx_labels, spans_tuples, train_emrs, train_dxs, train_ners, valid_emrs, valid_dxs, valid_ners
    gc.collect()

    # Model
    model = BertDxNERModel(
        encoder=encoder_names_mapping[args.encoder],
        dx_label_size=train_set.num_dx_labels,
        ner_label_size=train_set.num_ner_labels,
        loss_weights=args.lw
    ).to(args.device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # Optimization
    train_log = Namespace(dx_acc=list(), dx_loss=list(), ner_acc=list(), ner_loss=list(), best_dx_acc=0, best_ner_acc=0)
    nsteps = 0

    # Training loop
    for epoch in range(args.nepochs):
        print(f"\n===== Training at epoch {epoch + 1} =====\n")
        for x, y_dx, y_ner in train_loader:
            model.train()

            # move to device
            x = move_bert_input_to_device(x, args.device)
            y_dx = y_dx.to(args.device)
            y_ner = y_ner.to(args.device)

            # inference
            o_dx, o_ner = model(x)
            dx_loss, ner_loss, total_loss = model.calc_loss(o_dx, y_dx, o_ner, y_ner)

            # back-prop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # evaluate model every k steps
            if nsteps % args.ckpt_steps == 0:
                print(f"Evaluating model at step {nsteps}...")
                train_log = evaluate_and_save(valid_loader, model, args, train_log)
            nsteps += 1

        print(f"----- Evaluating at epoch {epoch + 1} -----")
        train_log = evaluate_and_save(valid_loader, model, args, train_log)

    return train_log

def evaluate_and_save(data_loader, model, args, train_log):
    dx_correct, dx_predict, ner_correct, ner_predict = 0, 0, 0, 0
    total_dx_loss, total_ner_loss = 0.0, 0.0

    model.eval()
    for x, y_dx, y_ner in data_loader:
        x = move_bert_input_to_device(x, args.device)
        y_dx = y_dx.to(args.device)
        y_ner = y_ner.to(args.device)
        with torch.no_grad():
            o_dx, o_ner = model(x)
            dx_loss, ner_loss, _ = model.calc_loss(o_dx, y_dx, o_ner, y_ner)
            # dx acc
            dx_correct += (o_dx.argmax(dim=-1) == y_dx).sum().cpu().detach().item()
            dx_predict += len(y_dx)
            # ner acc (token acc)
            ner_correct += (o_ner.argmax(dim=-1) == y_ner).sum().cpu().detach().item()
            ner_predict += (y_ner != model.ner_ignore_index).sum().cpu().detach().item()
            total_dx_loss += dx_loss.cpu().detach().item() * y_dx.shape[0]
            total_ner_loss += ner_loss.cpu().detach().item() * y_ner.shape[0]

    dx_acc = dx_correct / dx_predict
    ner_acc = ner_correct / ner_predict
    dx_loss = total_dx_loss / len(data_loader.dataset)
    ner_loss = total_ner_loss / len(data_loader.dataset)
    print(f"Diagnosis: acc -> {dx_acc:.3f}; loss -> {dx_loss:.3f} / NER: acc -> {ner_acc:.3f}; loss -> {ner_loss:.3f}")

    # update train log
    for key, value in zip(["dx_acc", "dx_loss", "ner_acc", "ner_loss"], [dx_acc, dx_loss, ner_acc, ner_loss]):
        key_l = getattr(train_log, key)
        key_l.append(value)
    
    # update best metrics and save
    if dx_acc > train_log.best_dx_acc:
        train_log.best_dx_acc = dx_acc
        torch.save(model.state_dict(), args.ckpt_path / "best_model.ckpt")
        print(f"Best model saved. (Dx acc = {dx_acc:.3f}; NER token acc = {ner_acc:.3f})")
    if ner_acc > train_log.best_ner_acc:
        train_log.best_ner_acc = ner_acc

    # save train log
    (args.ckpt_path / "train_log.json").write_text(data=json.dumps(vars(train_log)))
    
    return train_log

if __name__ == "__main__":
    config = json.loads(Path("./config.json").read_bytes())
    args = Namespace(**config)


    for r in range(10):
        args.remainder = r
        print(f"\n\n******* Now training with remainder={r} *******\n\n")
        args.exp_name = render_exp_name(args, hparams=["encoder", "fc", "lw", "lr", "remainder"])
        train_log = trainer(args)
        print(f"\n\n******* Finish training remainder={r} (Best ACC: dx={train_log.best_dx_acc:.3f}; ner={train_log.best_ner_acc:.3f}) *******\n\n")