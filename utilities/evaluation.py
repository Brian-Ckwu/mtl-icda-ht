import torch
import pandas as pd
import sklearn.metrics
from torch.utils.data import DataLoader
from colorama import Fore, Style
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from utils import move_bert_input_to_device

def predict_whole_set(model: BertModel, data_loader: DataLoader, device: str) -> list:
    preds = list()
    model.eval()
    print(f"Predicting whole dataset...")
    for batch in tqdm(data_loader):
        x = move_bert_input_to_device(batch[0], device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred)
    return preds
        
def visualize_ner_labels(tokenizer: BertTokenizerFast, input_ids: list[int], ner_labels: list[int]):
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
    return pd.DataFrame(accs, columns=["k", "acc"])

def get_evaluations(y_true, y_pred, label_size, model_outputs, model_name: str) -> pd.DataFrame: # macro/micro-f1; Cohen's kappa; Matthew’s correlation coefficient; h@1,3,6,9
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    cohen_kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    hak = get_top_k_accuracies(y_true, model_outputs, k=10)
    return pd.DataFrame(
        {
            "macro_f1": [macro_f1],
            "micro_f1": [micro_f1],
            "cohen_kappa": [cohen_kappa],
            "mcc": [mcc],
            "h@3": hak[hak.k == 3].acc.values,
            "h@6": hak[hak.k == 6].acc.values,
            "h@9": hak[hak.k == 9].acc.values
        },
        index=[model_name]
    )