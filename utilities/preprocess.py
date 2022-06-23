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