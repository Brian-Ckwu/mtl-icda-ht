import gc
import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

import json
import pickle
from time import time
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import BertNENDataset, KBEntities
from model import BiEncoder
from test import fullset_evaluate
from utilities.data import split_by_div
from utilities.model import encoder_names_mapping
from utilities.utils import set_seeds, render_exp_name, move_bert_input_to_device

def in_batch_evaluate(data_loader: DataLoader, model: BiEncoder, args: Namespace) -> Tuple[float, float]: # return loss & acc
    total_loss = 0 
    total_correct = 0
    total_predict = 0
    pbar = tqdm(total=len(data_loader), ncols=0, desc="Valid", unit=" steps")
    for emr_be, mention_indices_l, target_cuis, negative_cuis_l in data_loader:
        model.eval()
        emr_be = move_bert_input_to_device(emr_be, args.device)

        with torch.no_grad():
            # Encode mentions
            mentions = model.encode_mentions(emr_be, mention_indices_l)
            assert len(mentions) == len(mention_indices_l) == len(target_cuis) == len(negative_cuis_l)

            # Encode entities
            for mention, target_cui, negative_cuis in zip(mentions, target_cuis, negative_cuis_l):
                batch_cuis = [target_cui] + negative_cuis
                assert len(batch_cuis) == args.cuibs
                ents_be = data_loader.dataset.make_entities_be(cuis=batch_cuis).to(args.device)
                ents_labels = data_loader.dataset.make_entities_labels(target_cui, negative_cuis).to(args.device)
                assert ents_labels[0].item() == 1.0
                assert ents_labels[1:].sum().item() == 0.0

                y_ment = mention
                y_ents = model.encode_entities(ents_be)

                scores = model.calc_scores(y_ment, y_ents).squeeze()
                loss = model.calc_loss(scores, ents_labels)

                # Accumulate metrics
                total_loss += loss.detach().cpu().item()
                total_correct += 1 if (scores.argmax(dim=-1).item() == 0) else 0 # TODO: evaluate accuracy
                total_predict += 1
        
        pbar.update(n=1)
    
    pbar.close()
    eval_loss = total_loss / total_predict
    eval_acc = total_correct / total_predict
    return eval_loss, eval_acc

def trainer(args: Namespace):
    # Configuration
    args.save_dir = Path(args.save_dir)
    args.exp_name = render_exp_name(args, hparams=["encoder", "seed", "emrbs", "cuibs", "optimizer", "lr", "nepochs", "fold", "remainder"])
    args.ckpt_path = args.save_dir / args.exp_name
    args.ckpt_path.mkdir(parents=True, exist_ok=True)
    (args.ckpt_path / "args.pickle").write_bytes(pickle.dumps(args))

    set_seeds(args.seed)

    # Data
    emrs = pickle.loads(Path(args.emr_path).read_bytes())
    ner_spans_l = pickle.loads(Path(args.ner_spans_l_path).read_bytes())
    sm2cui = json.loads(Path(args.sm2cui_path).read_bytes())
    smcui2name = json.loads(Path(args.smcui2name_path).read_bytes())

    train_emrs, train_ner_spans_l = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="train") for data in [emrs, ner_spans_l]]
    valid_emrs, valid_ner_spans_l = [split_by_div(data, fold=args.fold, remainder=args.remainder, mode="valid") for data in [emrs, ner_spans_l]]

    tokenizer = AutoTokenizer.from_pretrained(encoder_names_mapping[args.tokenizer])

    train_set = BertNENDataset(
        emrs=train_emrs,
        ner_spans_l=train_ner_spans_l,
        mention2cui=sm2cui,
        cui2name=smcui2name,
        cui_batch_size=args.cuibs,
        tokenizer=tokenizer
    )
    valid_set = BertNENDataset(
        emrs=valid_emrs,
        ner_spans_l=valid_ner_spans_l,
        mention2cui=sm2cui,
        cui2name=smcui2name,
        cui_batch_size=args.cuibs,
        tokenizer=tokenizer    
    )
    entities_set = KBEntities(
        id2desc=smcui2name,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_set, batch_size=args.emrbs, shuffle=True, pin_memory=True, collate_fn=lambda batch: batch[0])
    valid_loader = DataLoader(valid_set, batch_size=args.emrbs, shuffle=False, pin_memory=True, collate_fn=lambda batch: batch[0])
    entities_loader = DataLoader(entities_set, batch_size=args.cuibs, shuffle=False, pin_memory=True, collate_fn=entities_set.collate_fn)

    # Model
    model = BiEncoder(encoder_name=encoder_names_mapping[args.encoder]).to(args.device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # Optimization
    nsteps = 0
    stale = 0
    best_fullset_acc = 0.0
    train_log = {"fullset_acc": list(), "best_fullset_acc": best_fullset_acc}

    for epoch in range(1, args.nepochs + 1):
        print(f"\n===== Start training at epoch {epoch} =====\n")
        pbar = tqdm(total=len(train_loader), ncols=0, desc="Train", unit=" steps")

        for emr_be, mention_indices_l, target_cuis, negative_cuis_l in train_loader:
            model.train()
            
            # Encode mentions
            emr_be = move_bert_input_to_device(emr_be, args.device)

            mentions = model.encode_mentions(emr_be, mention_indices_l)
            assert len(mentions) == len(mention_indices_l) == len(target_cuis) == len(negative_cuis_l)

            # Encode entities
            emr_loss = torch.tensor([0.0]).to(args.device)
            for mention, target_cui, negative_cuis in zip(mentions, target_cuis, negative_cuis_l):
                batch_cuis = [target_cui] + negative_cuis; assert len(batch_cuis) == args.cuibs
                ents_be = train_set.make_entities_be(cuis=batch_cuis).to(args.device)
                ents_labels = train_set.make_entities_labels(target_cui, negative_cuis).to(args.device)

                y_ment = mention
                y_ents = model.encode_entities(ents_be)

                # Calculate score & loss
                scores = model.calc_scores(y_ment, y_ents)
                loss = model.calc_loss(scores.squeeze(), ents_labels)

                # Accumulate loss
                emr_loss += loss
            
            # Update parameters
            optimizer.zero_grad()
            if emr_loss.requires_grad:
                emr_loss.backward()
            optimizer.step()
            
            # Evaluate every k steps
            if nsteps % args.ckpt_steps == 0:
                fullset_acc = fullset_evaluate(valid_loader, model, args, entities_loader=entities_loader)
                train_log["fullset_acc"].append(fullset_acc)
                print(f"Model evaluated at step {nsteps}: accuracy = {fullset_acc:.3f}")
                
                # Check best acc
                if fullset_acc > best_fullset_acc:
                    stale = 0
                    best_fullset_acc = fullset_acc
                    train_log["best_fullset_acc"] = best_fullset_acc
                    print("Saving best model.")
                    torch.save(model.state_dict(), args.ckpt_path / "best_model.ckpt")
                else:
                    stale += 1
                    print(f"Model stop improving for {int(stale * args.ckpt_steps)} steps.")
                    if stale >= args.patience:
                        print("Stop training because of early stopping.")
                
                # Save train log
                (args.ckpt_path / "train_log.json").write_text(json.dumps(train_log))

            mean_loss = (emr_loss.detach().cpu().item() / len(mentions)) if len(mentions) > 0 else emr_loss.detach().cpu().item()
            nsteps += 1
            pbar.update(n=1)
            pbar.set_postfix(
                loss=f"{mean_loss:.3f}",
                acc=f"{fullset_acc:.3f}",
                steps=nsteps
            )

            del emr_loss, mean_loss
            gc.collect()

        pbar.close()
    
    # TODO: save representations of fullset entities
    y_ents_all = model.encode_all_entities(entities_loader, args)
    torch.save(y_ents_all, args.ckpt_path / f"entity_embeddings_{len(y_ents_all)}.pt")

    return best_fullset_acc


if __name__ == "__main__":
    config = json.loads(Path("./config.json").read_bytes())
    args = Namespace(**config)

    start_time = time()
    best_acc = trainer(args)
    end_time = time()
    total_minutes = (end_time - start_time) / 60

    print(f"\n\n ===== Finish training: best acc = {best_acc:.4f}; time used = {total_minutes:.2f} minutes =====")
