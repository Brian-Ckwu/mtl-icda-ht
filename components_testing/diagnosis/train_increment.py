import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

import json
import pickle
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from utilities.utils import set_seeds, render_exp_name, load_args, load_pickle, load_jsonl, get_logger
from utilities.preprocess import select_labels_subset, build_label2id_mapping, augment_samples_with_partials
from utilities.data import MedicalDxDataset
from utilities.model import BertDxModel, encoder_names_mapping
from utilities.trainer import ICDATrainer
from utilities.evaluation import evaluate_dx_model

import warnings
warnings.filterwarnings("ignore")

def main():
    # ## Configuration
    args = load_args("./config.json")
    set_seeds(args.seed)

    # logger
    logger = get_logger(name=str(__name__))

    # set up experiment
    args.exp_name = render_exp_name(args, hparams=args.exp_hparams, sep='__')
    args.exp_path = f"{args.save_dir}/{args.exp_name}"
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)

    # save args
    (Path(args.exp_path) / "config.json").write_text(json.dumps(vars(args), indent=4))
    (Path(args.exp_path) / "args.pickle").write_bytes(pickle.dumps(args))

    # ## Data
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
    (Path(args.exp_path) / "label_id2dx.json").write_text(json.dumps(id2dx, indent=4)) # save label mapping for inference
    label_ids = list(map(lambda dx: dx2id[dx], dxs))

    # split
    assert len(emrs) == len(label_ids)
    X_train, X_valid, eX_train, eX_valid, y_train, y_valid = train_test_split(emrs, extracted_emrs, label_ids, train_size=args.train_size, test_size=args.test_size, random_state=args.seed, stratify=label_ids)

    # partial augmentation of train_set
    if args.n_partials > 1:
        train_spans_l = [eX_train[i]["spans"] for i in range(len(eX_train))]
        X_train, y_train = augment_samples_with_partials(emrs=X_train, spans_l=train_spans_l, dxs=y_train, n_partials=args.n_partials)

    # build datasets
    tokenizer = AutoTokenizer.from_pretrained(encoder_names_mapping[args.tokenizer], use_fast=True)
    tokenizer.model_max_length = 512
    train_set = MedicalDxDataset(emrs=X_train, dx_labels=y_train, tokenizer=tokenizer)
    valid_set = MedicalDxDataset(emrs=X_valid, dx_labels=y_valid, tokenizer=tokenizer)

    # ## Model
    model = BertDxModel(
        encoder_name=encoder_names_mapping[args.encoder], 
        num_dxs=len(Counter(label_ids))
    )
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # ## Optimization
    trainer = ICDATrainer(
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        optimizer=optimizer,
        eval_func=evaluate_dx_model,
        logger=logger,
        args=args
    )
    trainer.train()

if __name__ == "__main__":
    main()
