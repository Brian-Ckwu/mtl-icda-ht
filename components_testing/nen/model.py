from typing import List

import torch
import torch.nn as nn

from transformers import BertModel

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

        return mentions
    
    def encode_entities(self, be):
        H = self.ent_encoder(**be).last_hidden_state # (batch, seq_len, embed_dim)
        h_cls = H[:, 0, :]
        return h_cls
    
    def calc_scores(self, y_ment: torch.FloatTensor, y_ents: torch.FloatTensor):
        return torch.mm(y_ment.unsqueeze(0), y_ents.transpose(1, 0))

    def calc_loss(self, scores: torch.FloatTensor, target: torch.LongTensor):
        return self.criterion(scores, target)