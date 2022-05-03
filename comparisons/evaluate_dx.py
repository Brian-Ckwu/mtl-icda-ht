import sys
import re
import gc
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

import json
import pickle
from pathlib import Path
from argparse import Namespace
from collections import Counter

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import ttest_rel
from transformers import AutoTokenizer, logging
logging.set_verbosity_error()

from utilities.data import MedicalDxDataset, DxBatchCollator, MedicalDxNERIOBDataset, convert_icds_to_indices, split_by_div
from utilities.model import BertDxModel, BertDxNERModel, encoder_names_mapping
from utilities.utils import move_bert_input_to_device, set_seeds, load_config, load_pickle
from utilities.evaluation import predict_whole_set_dx, get_top_k_accuracies, get_evaluations

MODEL_MAX_LENGTH = 512

def evaluate_dx_models(args: Namespace) -> pd.DataFrame:
    # Configuration
    set_seeds(args.seed)
    tokenizer_name = encoder_names_mapping[args.tokenizer]
    encoder_name = encoder_names_mapping[args.encoder]

    # Data
    emrs = load_pickle(args.emr_path)
    icds = load_pickle(args.icd_path); dxs = convert_icds_to_indices(icds, full_code=args.fc)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = MODEL_MAX_LENGTH

    eval_emrs, eval_dxs = [split_by_div(data, args.fold, args.remainder, mode="valid") for data in [emrs, dxs]]
    eval_set = MedicalDxDataset(eval_emrs, eval_dxs, tokenizer)
    eval_loader = DataLoader(eval_set, args.bs, shuffle=False, pin_memory=True, collate_fn=DxBatchCollator(tokenizer))

    # Model
    model = BertDxModel(encoder_name=encoder_name, num_dxs=len(Counter(dxs)))
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))

    # Evaluation
    scores = predict_whole_set_dx(model, eval_loader, args.device).detach().cpu()
    preds = scores.argmax(dim=-1).tolist()
    eval_df = get_evaluations(
        y_true=eval_dxs, 
        y_pred=preds, 
        label_size=len(Counter(dxs)),
        model_outputs=scores,
        model_name=args.encoder
    )

    return eval_df

if __name__ == "__main__":
    config = load_config("./dx_config.json")
    args = Namespace(**config)

    eval_dfs = list()
    for k in range(args.fold):
        print(f"Start evaluating fold = {k}:\n")
        args.remainder = k
        args.ckpt_path = re.sub(pattern=r"remainder\-\d", repl=f"remainder-{k}", string=args.ckpt_path)
        eval_df = evaluate_dx_models(args)
        eval_dfs.append(eval_df)
        print(eval_df)
    
    eval_dfs_cat = pd.concat(objs=eval_dfs)

    eval_mean_df = eval_dfs_cat.mean(axis=0).to_frame().T.rename({0: "mean"})
    eval_std_df = eval_dfs_cat.std(axis=0).to_frame().T.rename({0: "std"})
    eval_mean_std_df = pd.concat(objs=[eval_mean_df, eval_std_df])
    print("Cross validation finished:")
    print(eval_mean_std_df)
    
    eval_save_path = Path(args.eval_save_dir) / f"{args.encoder}_{args.lr}_eval_mean_std.csv"
    eval_mean_std_df.to_csv(eval_save_path, index_label="index")
    print("Cross validation metrics saved.")