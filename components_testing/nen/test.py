import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

import time
from typing import Tuple, List
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from model import BiEncoder
from utilities.utils import move_bert_input_to_device

def fullset_evaluate(data_loader: DataLoader, model: BiEncoder, args: Namespace, entity_embeddings: torch.FloatTensor = None, entities_loader: DataLoader = None) -> Tuple[float, List[float]]:
    assert entities_loader
    # Load or construct entity embeddings
    if entity_embeddings != None:
        all_y_ents = entity_embeddings.to(args.device)
    else:
        all_y_ents = model.encode_all_entities(entities_loader, args).to(args.device)
    
    # Evaluation
    all_cuis = entities_loader.dataset._ids
    total_correct = 0
    total_predict = 0
    batch_exec_times = list()

    model.eval()
    for emr_be, mention_indices_l, target_cuis, _ in data_loader: # No need of negative_cuis_l
        start_time = time.time()
        emr_be = move_bert_input_to_device(emr_be, args.device)

        with torch.no_grad():
            # Encode mentions
            y_ments = model.encode_mentions(emr_be, mention_indices_l)
            assert len(y_ments) == len(mention_indices_l) == len(target_cuis)

            # Calculate scores
            scores = model.calc_scores(y_ments, all_y_ents)
            
            # Check correctness
            preds = scores.argmax(dim=-1).cpu().tolist()
            # Calculate num correct
            for pred, target_cui in zip(preds, target_cuis):
                pred_cui = all_cuis[pred]
                if pred_cui == target_cui:
                    total_correct += 1
            
            total_predict += len(preds)
        end_time = time.time()
        batch_exec_time = (end_time - start_time) * 1000
        batch_exec_times.append(batch_exec_time)
    
    fullset_acc = total_correct / total_predict
    return fullset_acc, batch_exec_times