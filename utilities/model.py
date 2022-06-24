from typing import List
from argparse import Namespace

import torch.nn as nn
from transformers import BertModel, AutoModel

class BertDxModel(nn.Module):
    def __init__(self, encoder_name, num_dxs):
        super(BertDxModel, self).__init__()
        self.bert = AutoModel.from_pretrained(encoder_name, local_files_only=True)
        self.embed_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.fc = nn.Linear(self.embed_dim, num_dxs)

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, x):
        h = self.bert(**x).last_hidden_state
        cls_h = h[:, 0, :]
        scores = self.fc(cls_h)
        return scores
    
    def calc_loss(self, scores, labels):
        return self.criterion(scores, labels)

class BertNERModel(nn.Module):
    def __init__(self, encoder: str, num_tags: int):
        super(BertNERModel, self).__init__()
        # model
        self.bert = BertModel.from_pretrained(encoder)
        self.embed_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.fc = nn.Linear(self.embed_dim, num_tags)
    
    def forward(self, x):
        h_last = self.bert(**x).last_hidden_state
        scores = self.fc(h_last)
        return scores

class BertDxNERModel(nn.Module):
    def __init__(self, encoder: str, dx_label_size: int, ner_label_size: int, loss_weights: List[float] = [1, 1], ner_ignore_index: int = -100):
        super(BertDxNERModel, self).__init__()
        # model
        self.bert = BertModel.from_pretrained(encoder)
        self.embed_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.dx_layer = nn.Linear(self.embed_dim, dx_label_size)
        self.ner_layer = nn.Linear(self.embed_dim, ner_label_size)
        # criterion
        assert len(loss_weights) == 2
        self.loss_weights = loss_weights
        self.ner_ignore_index = ner_ignore_index
        self.dx_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.ner_criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=ner_ignore_index)
    
    def forward(self, x):
        H = self.bert(**x).last_hidden_state
        h_cls = H[:, 0, :]
        o_dx = self.dx_layer(h_cls)
        o_ner = self.ner_layer(H)
        return o_dx, o_ner
    
    def calc_dx_loss(self, o_dx, y_dx):
        return self.dx_criterion(o_dx, y_dx)
    
    def calc_ner_loss(self, o_ner, y_ner):
        return self.ner_criterion(o_ner.transpose(1, 2), y_ner)
    
    def calc_loss(self, o_dx, y_dx, o_ner, y_ner):
        dx_loss = self.calc_dx_loss(o_dx, y_dx)
        ner_loss = self.calc_ner_loss(o_ner, y_ner)
        total_loss = self.loss_weights[0] * dx_loss + self.loss_weights[1] * ner_loss
        return dx_loss, ner_loss, total_loss

encoder_names_mapping = {
    "BERT": "bert-base-uncased",
    "BioBERT": "dmis-lab/biobert-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "LinkBERT": "michiyasunaga/LinkBERT-base",
    "BioLinkBERT": "michiyasunaga/BioLinkBERT-base",
    "RoBERTa": "roberta-base",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
}