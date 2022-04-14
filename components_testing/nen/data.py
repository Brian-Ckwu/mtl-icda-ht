import torch
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class BertNENDataset(Dataset):
    def __init__(self, emrs: List[str], ner_spans_l: List[List[Tuple[int]]], mention2cui: Dict[str, str], tokenizer: BertTokenizerFast):
        assert len(emrs) == len(ner_spans_l)
        # attributes
        self.emrs = emrs
        self.ner_spans_l = ner_spans_l
        self.mention2cui = mention2cui
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.emrs)
    
    def __getitem__(self, idx): # -> BE (EMR), token_indices_l (NER), entities (NEN)
        emr = self.emrs[idx]
        ner_spans = self.ner_spans_l[idx]
        # TODO: filter ner_spans according to mention2entity & construct entity labels

        be = self.tokenizer(emr, return_offsets_mapping=True)
        offsets = be.pop("offset_mapping")

        token_indices_l = self.spans_to_token_indices_l(ner_spans, offsets)
        
        be = be.convert_to_tensors("pt", prepend_batch_axis=True)
        return be, token_indices_l
    
    @staticmethod
    def spans_to_token_indices_l(spans: List[Tuple[int]], offsets: List[Tuple[int]]) -> List[List[int]]:
        token_idx = 1 # start with the first token (skip [CLS])
        token_indices_l = list()
        for span in spans:
            token_indices = list()
            while True:
                offset = offsets[token_idx]
                start, end = offset
                if (start == 0) and (end == 0): # [CLS] or [SEP] token
                    break
                elif (start < span[0]):
                    token_idx += 1
                elif (end <= span[1]): # start >= span[0]
                    token_indices.append(token_idx)
                    token_idx += 1
                else: # end > span[1]
                    break
            token_indices_l.append(token_indices)
        
        return token_indices_l
