import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10003, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class Net2(Net1):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10003, 300, padding_idx=0)
        self.fc1 = nn.Linear(300, 128)

    def forward(self, x):
        x = self.emb(x)
        x = super().forward(x)
        return x


class Net3(Net2):
    def __init__(self, emb_matrix=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=(7,), padding='same')
        if emb_matrix is not None:
            self.emb.load_state_dict({"weight": emb_matrix})

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x



def load_embedding_matrix(vocab):
    glove = pd.read_csv('./glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}
    embedding_matrix = np.random.rand(len(vocab) + 3, 300)

    for word, index in enumerate(vocab):
        if word in glove_embedding:
            embedding_matrix[index] = glove_embedding[word]
    return torch.Tensor(embedding_matrix)
