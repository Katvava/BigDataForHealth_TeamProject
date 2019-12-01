import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from mydatasets import MedicalDiagnosisDataset, visit_collate_fn

from models.my_rnn_elmo import GRU_ELMO
from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
import numpy as np

# Set a correct path to the data files that you preprocessed

PATH_TEST_SEQS = "../data/saved_features/test_data.pkl"
PATH_TEST_LEN = "../data/saved_features/test_l_before_padding.pkl"

PATH_WEIGHT = "./model_weights/MyRNNELMo.pth"

NUM_EPOCHS = 500
BATCH_SIZE = 128
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

learning_rate = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print('===> Loading entire datasets')
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_len = pickle.load(open(PATH_TEST_LEN, 'rb'))

test_dataset = MedicalDiagnosisDataset(test_seqs['X'], test_seqs['y'], test_len)

# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn,
                         num_workers=NUM_WORKERS)

ELMO_DIM = 1024
HIDDEN_DIM = 64
OUTPUT_DIM = 2
N_LAYERS = 1
BIDIRECTIONAL = False
DROPOUT = 0.4

model = torch.load(PATH_WEIGHT)
criterion = nn.BCEWithLogitsLoss()
model.to(device)
criterion.to(device)

best_val_acc = 0.0
test_losses, test_accuracies = [], []
all_valid_results = []

for epoch in range(NUM_EPOCHS):
    test_loss, test_accuracy, valid_results = evaluate(model, device, test_loader, criterion)
    all_valid_results.append(valid_results)

    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

print('Average loss: {}'.format(np.mean(test_losses)))
print('Average accuracy: {}'.format(np.mean(test_accuracies)))

zx = 1
