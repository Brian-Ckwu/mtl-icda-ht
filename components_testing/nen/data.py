import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class BertNENDataset(Dataset):
    def __init__(
            self, 
            emrs: List[str], 
            ner_spans_l: List[List[Tuple[int]]], 
            mention2cui: Dict[str, str], 
            cui2name: Dict[str, str], 
            cui_batch_size: int,
            tokenizer: BertTokenizerFast
        ):

        assert len(emrs) == len(ner_spans_l)
        # attributes
        self.emrs = emrs
        self.ner_spans_l = ner_spans_l
        self.mention2cui = mention2cui
        self.cui2name = cui2name
        self.cuis = list(cui2name.keys())
        self.cui_batch_size = cui_batch_size
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.emrs)
    
    def __getitem__(self, idx): # -> BE (EMR), token_indices_l (NER), CUIs (NEN), negative samples (NEN)
        emr = self.emrs[idx]
        ner_spans = self.ner_spans_l[idx]
        # Filter ner_spans according to mention2cui & construct entity labels
        mapped_ner_spans = list()
        cuis = list()
        for ner_span in ner_spans:
            start, end = ner_span
            emr_span = emr[start:end].lower().strip()
            if emr_span in self.mention2cui:
                mapped_ner_spans.append(ner_span)
                cuis.append(self.mention2cui[emr_span])

        be = self.tokenizer(emr, return_offsets_mapping=True)
        offsets = be.pop("offset_mapping")
        token_indices_l = self.spans_to_token_indices_l(mapped_ner_spans, offsets)
        
        be = be.convert_to_tensors("pt", prepend_batch_axis=True)
        # Align token_indices_l & CUIs (remove empty list)
        assert len(token_indices_l) == len(cuis)
        token_indices_l_clean, cuis_clean = list(), list()
        for token_indices, cui in zip(token_indices_l, cuis):
            if token_indices != []:
                token_indices_l_clean.append(token_indices)
                cuis_clean.append(cui)
        # Prepare negative samples
        negative_cuis_l = list()
        for cui in cuis_clean:
            negative_cuis = self.random_negative_sampling(target_cui=cui, batch_size=self.cui_batch_size)
            negative_cuis_l.append(negative_cuis)

        assert len(token_indices_l_clean) == len(cuis_clean) == len(negative_cuis_l)
        return be, token_indices_l_clean, cuis_clean, negative_cuis_l
    
    def random_negative_sampling(self, target_cui: str, batch_size: int):
        random_cuis = random.sample(self.cuis, k=batch_size)
        negatives = list()
        for random_cui in random_cuis:
            if random_cui != target_cui:
                negatives.append(random_cui)
        return negatives[:batch_size - 1] # return 'batch_size - 1' (e.g. 15) negative samples
    
    def make_entities_be(self, cuis: List[str]):
        entities = [self.cui2name[cui] for cui in cuis]
        be = self.tokenizer(entities, padding=True, return_tensors="pt")
        return be

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
    
    @staticmethod
    def make_entities_labels(target_cui: str, negative_cuis: List[str]) -> torch.LongTensor:
        all_cuis = [target_cui] + negative_cuis
        labels = list()
        for cui in all_cuis:
            label = cui == target_cui
            labels.append(label)
        
        return torch.LongTensor(labels)