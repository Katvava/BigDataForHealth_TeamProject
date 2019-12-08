import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mydatasets import MedicalDiagnosisDataset, visit_collate_fn
from utils import evaluate

# Set a correct path to the data files that you preprocessed

PATH_TEST_SEQS = "../data/saved_features/test_data.pkl"
PATH_TEST_LEN = "../data/saved_features/test_l_before_padding.pkl"

PATH_WEIGHT = "./model_weights/MyRNNELMo.pth"

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
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn,
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
test_losses, test_accuracies, test_f1s, test_aucs = [], [], [], []
all_valid_results = []

test_loss, test_accuracy, test_f1, test_auc, valid_results = evaluate(model, device, test_loader, criterion)

test_losses.append(test_loss)
test_accuracies.append(test_accuracy)
test_f1s.append(test_f1)
test_aucs.append(test_auc)

all_valid_results.append(valid_results)

print('Average loss: {}'.format(np.mean(test_losses)))
print('Average accuracy: {}'.format(np.mean(test_accuracies)))
print('Average f1: {}'.format(np.mean(test_f1s)))
print('Average auc: {}'.format(np.mean(test_aucs)))


zx = 1
