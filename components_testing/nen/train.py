import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

from tqdm import tqdm
from typing import Tuple
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from model import BiEncoder

from utilities.utils import move_bert_input_to_device

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