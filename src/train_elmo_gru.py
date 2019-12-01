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

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "../data/saved_features/train_data.pkl"
PATH_VALID_SEQS = "../data/saved_features/valid_data.pkl"
PATH_TEST_SEQS = "../data/saved_features/test_data.pkl"

PATH_TRAIN_LEN = "../data/saved_features/train_l_before_padding.pkl"
PATH_VALID_LEN = "../data/saved_features/valid_l_before_padding.pkl"
PATH_TEST_LEN = "../data/saved_features/test_l_before_padding.pkl"

PATH_OUTPUT = "./model_weights"

NUM_EPOCHS = 500
BATCH_SIZE = 128
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

learning_rate = 0.00005

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_len = pickle.load(open(PATH_TRAIN_LEN, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_len = pickle.load(open(PATH_VALID_LEN, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_len = pickle.load(open(PATH_TEST_LEN, 'rb'))

train_dataset = MedicalDiagnosisDataset(train_seqs['X'], train_seqs['y'], train_len)
valid_dataset = MedicalDiagnosisDataset(valid_seqs['X'], valid_seqs['y'], valid_len)
test_dataset = MedicalDiagnosisDataset(test_seqs['X'], test_seqs['y'], test_len)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn,
                          num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn,
                          num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn,
                         num_workers=NUM_WORKERS)

ELMO_DIM = 1024
HIDDEN_DIM = 8
OUTPUT_DIM = 2
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.5

criterion = nn.BCEWithLogitsLoss()
model = GRU_ELMO(ELMO_DIM,
                 HIDDEN_DIM,
                 OUTPUT_DIM,
                 N_LAYERS,
                 BIDIRECTIONAL,
                 DROPOUT)
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-4)

model.to(device)
criterion.to(device)


best_val_acc = 0.0
train_losses, train_accuracies, train_f1s, train_aucs = [], [], [], []
valid_losses, valid_accuracies, valid_f1s, valid_aucs = [], [], [], []
all_valid_results = []

for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy, train_f1, train_auc = train(model, device, train_loader, criterion, optimizer, epoch)
    valid_loss, valid_accuracy, valid_f1, valid_auc, valid_results = evaluate(model, device, valid_loader, criterion)
    all_valid_results.append(valid_results)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    train_f1s.append(train_f1)
    valid_f1s.append(valid_f1)

    train_aucs.append(train_auc)
    valid_aucs.append(valid_auc)

    is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    if is_best and epoch > 20:
        best_val_acc = valid_accuracy
        torch.save(model, os.path.join(PATH_OUTPUT, "MyRNNELMo.pth"))

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyRNNELMo.pth"))

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1s, valid_f1s, train_aucs, valid_aucs)

zx = 1
