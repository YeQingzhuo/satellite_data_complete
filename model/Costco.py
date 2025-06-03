
import torch as t
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io as scio
import torch as t
import sys
import h5py

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')


class CostCo(nn.Module):
    # from kdd 18
    # paper:CoSTCo: A Neural Tensor Completion Model for Sparse Tensors
    def __init__(self, i_size, j_size, k_size, embedding_dim, nc=100, is_history=False):
        super(CostCo, self).__init__()

        # （embedding layers）iembeddings、jembeddings and kembeddings。
        self.iembeddings = nn.Embedding(i_size, embedding_dim)
        self.jembeddings = nn.Embedding(j_size, embedding_dim)
        self.kembeddings = nn.Embedding(k_size, embedding_dim)

        self.conv1 = nn.Conv2d(1, nc, (1, embedding_dim))
        self.conv2 = nn.Conv2d(nc, nc, (3, 1))

        #  fc1
        self.fc1 = nn.Linear(nc, 1)

        self.is_history = is_history

        self.history_prob = nn.Parameter(torch.abs(torch.tensor(0.5)))

    
    def forward(self, i_input, j_input, k_input, is_history=False):
        # embedding
        if is_history:
            if t.rand(1).item() < self.history_prob.item():
                return t.zeros(i_input.size(0), 1).to(i_input.device)
                
        lookup_itensor = i_input.long()
        lookup_jtensor = j_input.long()
        lookup_ktensor = k_input.long()
        
        i_embeds = self.iembeddings(lookup_itensor)
        j_embeds = self.jembeddings(lookup_jtensor)
        k_embeds = self.kembeddings(lookup_ktensor)

        H = t.cat((i_embeds.unsqueeze(1), j_embeds.unsqueeze(1), k_embeds.unsqueeze(1)), 1)
        H = H.unsqueeze(1)

        x = t.relu(self.conv1(H))
        x = t.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        
        return x
    
