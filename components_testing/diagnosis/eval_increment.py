import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

from typing import List, Dict
from pathlib import Path
from argparse import Namespace, ArgumentParser
from collections import Counter

import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utilities.utils import set_seeds, load_pickle, load_jsonl
from utilities.preprocess import select_labels_subset, build_label2id_mapping, augment_samples_with_partials
from utilities.data import MedicalDxDataset
from utilities.model import BertDxModel, encoder_names_mapping
from utilities.evaluation import evaluate_dx_model

import warnings
warnings.filterwarnings("ignore")

def main(args: Namespace):
    # Configuration
    set_seeds(args.seed)

    # Data
    # load data
    emrs = load_pickle(args.emr_path)
    dxs = load_pickle(args.dx_path)
    extracted_emrs = load_jsonl(args.extracted_emrs_path)
    target_dxs = load_pickle(args.target_dx_path)

    # select subset
    outputs, dxs = select_labels_subset(inputs=list(zip(emrs, extracted_emrs)), labels=dxs, target_labels=set(target_dxs))
    emrs, extracted_emrs = zip(*outputs)

    # preprocessing
    # TODO: preprocess EMRs
    # preprocess Dxs
    dx2id = build_label2id_mapping(labels=dxs)
    id2dx = {v: k for k, v in dx2id.items()}
    label_ids = list(map(lambda dx: dx2id[dx], dxs))

    # split
    assert len(emrs) == len(label_ids)
    _, X_valid, _, eX_valid, _, y_valid = train_test_split(emrs, extracted_emrs, label_ids, train_size=args.train_size, test_size=args.test_size, random_state=args.seed, stratify=label_ids)

    # build datasets
    tokenizer = AutoTokenizer.from_pretrained(encoder_names_mapping[args.tokenizer], use_fast=True)
    tokenizer.model_max_length = 512
    valid_set = MedicalDxDataset(emrs=X_valid, dx_labels=y_valid, tokenizer=tokenizer)

    # Model
    model = BertDxModel(
        encoder_name=encoder_names_mapping[args.encoder], 
        num_dxs=len(Counter(label_ids))
    )
    model.load_state_dict(torch.load(Path(args.exp_path) / "best_models" / f"best_{args.model_metric}.pth", map_location=args.device))

    # Incremental Evaluation
    # make evaluation partials
    valid_spans_l = [eX_valid[i]["spans"] for i in range(len(eX_valid))]
    X_valid_partials, y_valid_partials = augment_samples_with_partials(
        emrs=X_valid, 
        spans_l=valid_spans_l, 
        dxs=y_valid, 
        n_partials=args.eval_partials
    )

    # calculate partial metrics
    partial_metrics_l = list()
    assert len(X_valid_partials) % args.eval_partials == 0
    seg_len = len(X_valid_partials) // args.eval_partials

    for seg in range(1, args.eval_partials + 1):
        seg_start_idx = (seg % args.eval_partials) * seg_len
        X_valid_partial = X_valid_partials[seg_start_idx:seg_start_idx + seg_len]
        y_valid_partial = y_valid_partials[seg_start_idx:seg_start_idx + seg_len]
        valid_set_partial = MedicalDxDataset(emrs=X_valid_partial, dx_labels=y_valid_partial, tokenizer=tokenizer)
        valid_loader_partial = DataLoader(valid_set_partial, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, collate_fn=valid_set_partial.collate_fn)
        metrics = evaluate_dx_model(model, valid_loader_partial, args.device, verbose=True)
        partial_metrics_l.append(metrics)
        print(metrics)

    # merge results
    incremental_metrics = merge_dicts(partial_metrics_l)

    # Save Eval Results
    df = pd.DataFrame(data=incremental_metrics, index=[(i / args.eval_partials) for i in range(1, args.eval_partials + 1)]).rename_axis("prop", axis="index")
    save_path = Path(args.exp_path) / "eval_results"
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / f"incremental_{args.model_metric}.csv", index=True)

def merge_dicts(dicts: List[dict]) -> Dict[str, list]:
    merged = {k: list() for k in dicts[0].keys()}
    for d in dicts:
        for k, v in d.items():
            merged[k].append(v)

    return merged

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--args_path",
        type=str,
        help="Path to training args"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Evaluation device"
    )

    parser.add_argument(
        "--eval_partials",
        type=int,
        default=10,
        help="How many partials are used for evaluation?"
    )

    parser.add_argument(
        "--model_metric",
        type=str,
        default="hat3",
        help="Which model (saved for different best metrics during training) to use?"
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    cmd_args = parse_args()

    args = load_pickle(cmd_args.args_path)
    args.device = cmd_args.device
    args.eval_partials = cmd_args.eval_partials
    args.model_metric = cmd_args.model_metric

    main(args)