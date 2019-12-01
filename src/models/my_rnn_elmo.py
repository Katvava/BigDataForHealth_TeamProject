import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# class GRU_ELMO(nn.Module):
#     def __init__(self,
#                  hidden_dim,
#                  output_dim,
#                  n_layers,
#                  bidirectional,
#                  dropout):
#
#         super().__init__()
#         elmo_dim = 1024
#
#         self.rnn = nn.GRU(elmo_dim,
#                           hidden_dim,
#                           num_layers=n_layers,
#                           bidirectional=bidirectional,
#                           batch_first=True,
#                           dropout=0 if n_layers < 2 else dropout)
#
#         self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text):
#
#         # text = [batch size, sent len]
#
#         # embedded = [batch size, sent len, emb dim]
#
#         tmp, hidden = self.rnn(text)
#
#         # hidden = [n layers * n directions, batch size, emb dim]
#
#         if self.rnn.bidirectional:
#             hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         else:
#             hidden = self.dropout(hidden[-1, :, :])
#
#         # hidden = [batch size, hid dim]
#
#         output = self.out(hidden)
#
#         # output = [batch size, out dim]
#
#         return output

# class GRU_ELMO(nn.Module):
#     def __init__(self,
#                  dim_input,
#                  hidden_dim,
#                  output_dim,
#                  n_layers,
#                  bidirectional,
#                  dropout):
#         super(GRU_ELMO, self).__init__()
#
#         rnn_input_size = dim_input//16
#
#         self.rnn = nn.GRU(rnn_input_size,
#                           hidden_dim,
#                           n_layers,
#                           bidirectional=bidirectional,
#                           batch_first = True,
#                           dropout=0 if n_layers < 2 else dropout)
#
#         self.fc1 = nn.Linear(dim_input, dim_input//4)
#         self.fc2 = nn.Linear(dim_input//4, dim_input//16)
#         self.fc3 = nn.Linear(dim_input//16, rnn_input_size)
#
#         if bidirectional == True:
#             self.fc4 = nn.Linear(rnn_input_size * 2, output_dim)
#         else:
#             self.fc4 = nn.Linear(rnn_input_size, output_dim)
#
#         self.dropout = nn.Dropout(p=0.3)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_tuple):
#         seqs, lengths = input_tuple
#         x1 = self.relu(self.fc1(seqs))
#         x2 = self.relu(self.fc2(x1))
#         x3 = self.relu(self.fc3(x2))
#
#         x3 = pack_padded_sequence(x3, lengths, batch_first = True)
#         output_all_steps, h = self.rnn(x3)
#         output_all_steps = pad_packed_sequence(output_all_steps)
#
#         if self.rnn.bidirectional:
#             h = self.dropout(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))
#         else:
#             h = self.dropout(h[-1, :, :])
#
#         x4 = self.dropout(self.relu(self.fc4(h)))
#         return x4

class GRU_ELMO(nn.Module):
    def __init__(self,
                 dim_input,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        super(GRU_ELMO, self).__init__()

        rnn_input_size = dim_input

        self.rnn = nn.GRU(rnn_input_size,
                          hidden_dim,
                          n_layers,
                          bidirectional=bidirectional,
                          batch_first = True,
                          dropout=0 if n_layers < 2 else dropout)

        # self.fc1 = nn.Linear(dim_input, dim_input//4)
        # self.fc2 = nn.Linear(dim_input//4, dim_input//16)
        # self.fc3 = nn.Linear(dim_input//16, rnn_input_size)

        if bidirectional == True:
            self.fc4 = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, input_tuple):
        seqs, lengths = input_tuple

        x1 = pack_padded_sequence(seqs, lengths, batch_first = True)
        output_all_steps, h = self.rnn(x1)
        output_all_steps = pad_packed_sequence(output_all_steps)

        if self.rnn.bidirectional:
            h = self.dropout(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))
        else:
            h = self.dropout(h[-1, :, :])

        x2 = self.dropout(self.fc4(h))
        return x2

#
# X = torch.rand([32, 100, 1024])
# model = GRU_ELMO(1024, 64,  2, 2, True, 0.3)
# y = model([X, [32]*32])
