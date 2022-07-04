import json
import re
import ast

import pandas as pd
import dask.dataframe as dd
import torch
from torch.utils.data import Dataset
# from allennlp.modules.elmo import Elmo
# from allennlp.modules.elmo import batch_to_ids

# import model.constant as C

DIGIT_PATTERN = re.compile('\d')

def bio_to_bioes(labels):
    """Convert a sequence of BIO labels to BIOES labels.
    :param labels: A list of labels.
    :return: A list of converted labels.
    """
    label_len = len(labels)
    labels_bioes = []
    for idx, label in enumerate(labels):
        next_label = labels[idx + 1] if idx < label_len - 1 else 'O'
        if label == 'O':
            labels_bioes.append('O')
        elif label.startswith('B-'):
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('S-' + label[2:])
        else:
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('E-' + label[2:])
    return labels_bioes


def mask_to_distance(mask, mask_len, decay=.1):
    if 1 not in mask:
        return [0] * mask_len
    start = mask.index(1)
    end = mask_len - list(reversed(mask)).index(1)
    dist = [0] * mask_len
    for i in range(start):
        dist[i] = max(0, 1 - (start - i - 1) * decay)
    for i in range(end, mask_len):
        dist[i] = max(0, 1 - (i - end) * decay)
    return dist

class FetDataset(Dataset):
    def __init__(self,
                 preprocessor,
                 training_file_path,
                 tokens_field,
                 entities_field,
                 sentence_field,
                 label_stoi,
                 test = False,
                 gpu=False):
        self.preprocessor = preprocessor
        self.gpu = gpu
        self.test = test
        self.entities_field = entities_field
        self.tokens_field = tokens_field
        self.sentence_field = sentence_field
        self.label_stoi = label_stoi
        self.label_size = len(label_stoi)
        self.data = pd.read_parquet(training_file_path, engine="fastparquet") if training_file_path.endswith('.parquet') else pd.read_csv(training_file_path)  
    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        if type(record[self.tokens_field]) == str:
            record_dict = {"tokens":ast.literal_eval(record[self.tokens_field]),"entities":ast.literal_eval(record[self.entities_field]),"sentence":record[self.sentence_field]}
        else:
            record_dict = {"tokens":record[self.tokens_field],"entities":record[self.entities_field],"sentence":record[self.sentence_field]}
        instance = self.preprocessor.process_instance(record_dict,self.label_stoi,self.test)
        # instance = ast.literal_eval(record['instance'])
        return instance

    def __len__(self):
        return len(self.data)

