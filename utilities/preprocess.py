import math
from typing import List, Any, Set, Tuple, Dict

def select_labels_subset(inputs: List[Any], labels: List[Any], target_labels: Set[Any]) -> Tuple[list]:
    assert len(inputs) == len(labels)

    new_inputs = list()
    new_labels = list()
    for input_, label in zip(inputs, labels):
        if label in target_labels:
            new_inputs.append(input_)
            new_labels.append(label)

    return new_inputs, new_labels

def build_label2id_mapping(labels: list) -> Dict[Any, int]:
    label2id = dict()
    for label in labels:
        if label not in label2id:
            label2id[label] = len(label2id)

    return label2id

def make_partial(emr: str, spans: List[List[int]], prop: float) -> str:
    if len(spans) == 0:
        return emr
    
    end_span_idx = math.floor(len(spans) * prop)
    try:
        end_offset = spans[end_span_idx][1]
    except:
        print(emr)
        print(spans)
        print(end_span_idx)
    return emr[:end_offset]

def augment_samples_with_partials(emrs: List[str], spans_l: List[List[List[int]]], dxs: List[Any], n_partials: int) -> Tuple[List[str], List[Any]]:
    assert len(emrs) ==  len(spans_l) == len(dxs)
    augmented_emrs = emrs.copy()
    augmented_dxs = dxs.copy()

    for partial_idx in range(1, n_partials):
        prop = partial_idx / n_partials
        for emr, spans, dx in zip(emrs, spans_l, dxs):
            partial_emr = make_partial(emr, spans, prop)
            augmented_emrs.append(partial_emr)
            augmented_dxs.append(dx)
    
    assert len(augmented_emrs) == len(augmented_dxs) == len(dxs) * n_partials
    return augmented_emrs, augmented_dxs