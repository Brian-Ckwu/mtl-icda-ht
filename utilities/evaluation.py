import torch
import pandas as pd
import sklearn.metrics
from colorama import Fore, Style
from tqdm import tqdm
from typing import List, Tuple

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast

from .utils import move_bert_input_to_device

def predict_whole_set_dx(model: BertModel, data_loader: DataLoader, device: str) -> list:
    preds = list()
    model.to(device)
    model.eval()
    print(f"Predicting whole dataset...")
    for batch in tqdm(data_loader):
        x = move_bert_input_to_device(batch[0], device)
        with torch.no_grad():
            pred = model(x)
            if len(pred) > 1:
                pred = pred[0]
            preds.append(pred)
    return torch.cat(preds, dim=0)

def predict_whole_set_ner(model: BertModel, data_loader: DataLoader, device: str) -> list:
    y_pred_raw = list()
    y_true_raw = list()
    model.to(device)
    model.eval()
    print(f"Predicting whole dataset...")
    for x, y in tqdm(data_loader):
        x = move_bert_input_to_device(x, device)
        with torch.no_grad():
            scores = model(x)
            pred = scores.argmax(dim=-1).detach().cpu().tolist()
            y_pred_raw.append(pred)
        true = y.detach().cpu().tolist()
        y_true_raw.append(true)

    return y_pred_raw, y_true_raw

def ids_to_iobs(y_pred_raw: List[List[List[int]]], y_true_raw: List[List[List[int]]], ner_dataset: Dataset) -> Tuple[List[List[str]], List[List[str]]]:
    y_pred = list()
    y_true = list()
    for batch_pred, batch_true in zip(y_pred_raw, y_true_raw):
        for emr_pred, emr_true in zip(batch_pred, batch_true):
            assert len(emr_pred) == len(emr_true)
            iob_preds = list()
            iob_trues = list()
            for idx_pred, idx_true in zip(emr_pred, emr_true):
                if idx_true != ner_dataset.ignore_index:
                    iob_pred = ner_dataset.idx2iob[idx_pred]
                    iob_true = ner_dataset.idx2iob[idx_true]
                    iob_preds.append(iob_pred)
                    iob_trues.append(iob_true)
            y_pred.append(iob_preds)
            y_true.append(iob_trues)
                
    return y_pred, y_true

def visualize_ner_labels(tokenizer: BertTokenizerFast, input_ids: List[int], ner_labels: List[int]):
    for i, token_id in enumerate(input_ids[0]):
        token = tokenizer.decode(token_id)
        if token[:2] == "##":
            token = token[2:]
            print('\b', end='')
        if ner_labels[i] == 0:
            print(Style.RESET_ALL + token, end=' ')
        else:
            print(Fore.RED + token, end=' ')

def get_top_k_accuracies(y_true, y_scores, k, labels):
    accs = list()
    for i in range(1, k + 1):
        acc = sklearn.metrics.top_k_accuracy_score(y_true, y_scores, k=i, labels=labels)
        accs.append((i, acc))
    return pd.DataFrame(accs, columns=['k', "acc"]).set_index('k')

def get_evaluations(y_true, y_pred, label_size, model_outputs, model_name: str) -> pd.DataFrame: # macro/micro-f1; Cohen's kappa; Matthew’s correlation coefficient; h@1,3,6,9
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    cohen_kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    hak = get_top_k_accuracies(y_true, model_outputs, k=10, labels=range(label_size))
    return pd.DataFrame(
        {
            "macro_f1": [macro_f1],
            "micro_f1": [micro_f1],
            "cohen_kappa": [cohen_kappa],
            "mcc": [mcc],
            "h@3": hak.loc[3].acc,
            "h@6": hak.loc[6].acc,
            "h@9": hak.loc[9].acc
        },
        index=[model_name]
    )