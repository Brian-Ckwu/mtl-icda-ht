import sys
sys.path.append("/nfs/nas-7.1/ckwu/mtl-icda-ht")

from typing import List
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel

from utilities.utils import move_bert_input_to_device

class BiEncoder(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.emr_encoder = BertModel.from_pretrained(encoder_name)
        self.ent_encoder = BertModel.from_pretrained(encoder_name)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    
    def encode_mentions(self, be, mention_indices_l: List[List[int]]) -> List[torch.FloatTensor]: 
        H = self.emr_encoder(**be).last_hidden_state[0] # single instance batch
        
        mentions = list()
        for mention_indices in mention_indices_l:
            mention = H[mention_indices, :]
            mention = torch.mean(mention, dim=0) # TODO: change mean pooling to other kind of pooling
            mentions.append(mention)

        return torch.stack(mentions, dim=0) if mentions else mentions
    
    def encode_entities(self, be):
        H = self.ent_encoder(**be).last_hidden_state # (batch, seq_len, embed_dim)
        h_cls = H[:, 0, :]
        return h_cls
    
    def calc_scores(self, y_ment: torch.FloatTensor, y_ents: torch.FloatTensor) -> torch.FloatTensor:
        if y_ment.dim() == 1:
            y_ment = y_ment.unsqueeze(0)
        return torch.mm(y_ment, y_ents.transpose(1, 0))

    def calc_loss(self, scores: torch.FloatTensor, target: torch.LongTensor) -> torch.FloatTensor:
        return self.criterion(scores, target)

    def encode_all_entities(self, entities_loader: DataLoader, args: Namespace):
        self.eval()
        all_cuis = list()
        all_y_ents = list()
        for cuis, ents_be in entities_loader:
            with torch.no_grad():
                ents_be = move_bert_input_to_device(ents_be, args.device)
                y_ents = self.encode_entities(ents_be)

                all_cuis += cuis
                all_y_ents.append(y_ents.detach().cpu())

        assert all_cuis == entities_loader.dataset._ids
        all_y_ents = torch.cat(all_y_ents, dim=0)
        return all_y_ents