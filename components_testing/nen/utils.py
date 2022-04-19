import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from data import KBEntities
from model import BiEncoder

from utilities.utils import move_bert_input_to_device

# Encode mention by [CLS] token representation, suitable for evalutating direct phrase-entity mapping models (may not be suitable to evaluate models trained on EMRs)
def evaluate_by_vocab(vocab_path: str, entities_set: Dataset, entity_embeddings: torch.FloatTensor, model: BiEncoder, tokenizer: BertTokenizerFast, args: Namespace):
    vocab = Path(vocab_path).read_text().split('\n')
    vocab_dict = {i: phrase for i, phrase in enumerate(vocab)}

    vocab_set = KBEntities(id2desc=vocab_dict, tokenizer=tokenizer)
    vocab_loader = DataLoader(vocab_set, batch_size=args.cuibs, shuffle=False, pin_memory=True, collate_fn=vocab_set.collate_fn)

    model.eval()

    all_desc_preds = list()
    for phr_ids, phr_be in tqdm(vocab_loader):
        phr_be = move_bert_input_to_device(phr_be, args.device)

        with torch.no_grad():
            h_cls = model.emr_encoder(**phr_be).last_hidden_state[:, 0, :]
            y_ments = h_cls.to(args.device)
            y_ents_all = entity_embeddings.to(args.device)

            scores = model.calc_scores(y_ments, y_ents_all)
            preds = scores.argmax(dim=-1).cpu().tolist()
            cui_preds = [entities_set._ids[pred] for pred in preds]
            desc_preds = [entities_set._id2desc[cui_pred] for cui_pred in cui_preds]
        
        all_desc_preds += desc_preds

    df = pd.DataFrame(data={"phrase": vocab, "desc_pred": all_desc_preds})    
    return df

def fullset_qualitative_evaluate():

    raise NotImplementedError