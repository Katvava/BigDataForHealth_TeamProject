import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

class MedicalDiagnosisDataset(Dataset):
    def __init__(self, seqs, labels, seq_len):
        self.seqs = seqs

        self.labels = []

        for i in labels:
            onehot = [0] * 2
            onehot[i] = 1
            self.labels.append(onehot)

        self.seq_len = seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index], self.seq_len[index]


def visit_collate_fn(batch):
    # the batch is a list of [(seq, label1, seq_len1), (seq2, label2, seq_len2), ... , (seqN, labelN, seq_lenN)]

    seqs = []
    labels = []
    lengths = []

    for i in range(len(batch)):
        lengths.append(batch[i][2])

    lengths = np.asarray(lengths)
    sorted_index = np.argsort(-1*lengths)
    lengths = np.sort(-1*lengths)*-1

    for j in range(len(sorted_index)):
        i = sorted_index[j]
        data = batch[i]

        seq = data[0]
        label = data[1]

        seqs.append(seq)
        labels.append(label)

    seqs = torch.tensor(np.asarray(seqs)).float()
    lengths = torch.tensor(np.asarray(lengths)).float()
    labels = torch.tensor(np.asarray(labels)).float()

    return (seqs, lengths), labels

