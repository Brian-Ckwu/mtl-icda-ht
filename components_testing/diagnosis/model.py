import torch
import torch.nn as nn
from transformers import BertModel

class BERTClassification(nn.Module):
    def __init__(self, model_name, embed_size, label_size):
        super(BERTClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(embed_size, label_size)
    
    def forward(self, x):
        h = self.bert(**x).last_hidden_state
        cls_h = h[:, 0, :]
        pred = self.fc(cls_h)
        return pred
