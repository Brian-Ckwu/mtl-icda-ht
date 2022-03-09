import torch.nn as nn
from transformers import BertModel

class BertTokenClassification(nn.Module):
    def __init__(self, model_name, embed_size, label_size):
        super(BertTokenClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(embed_size, label_size)
    
    def forward(self, x):
        last_h = self.bert(**x).last_hidden_state
        pred = self.fc(last_h)
        return pred