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
from model.layers import Attention, ModeProduct

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')


class NTM(nn.Module):
    def __init__(self, num_user, num_item, num_time, latent_dim, k=100, c=5, is_history=False):
        super(NTM, self).__init__()
        seed = 42
        if seed is not None:
            t.manual_seed(seed)

        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

       
        self.w = nn.Parameter(t.randn(latent_dim, k))
        assert not torch.isnan(self.w).any(), "w contains NaN after initialization"
        self.mode = ModeProduct(latent_dim, latent_dim, latent_dim, c, c, c)
        self.flat = nn.Flatten()
        self.FC = nn.Linear(in_features=c ** 3 + k, out_features=1)
        self.c = c

        self.is_history = is_history

        self.history_prob = nn.Parameter(torch.abs(torch.tensor(0.5)))

    def forward(self, i_input, j_input, k_input, is_history=False):
        if is_history:
            if t.rand(1).item() < self.history_prob.item():
                return t.zeros(i_input.size(0), 1).to(i_input.device)
        
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[k_input]

        assert not torch.isnan(i_embeds).any(), "i_embeds contains NaN"
        assert not torch.isnan(j_embeds).any(), "j_embeds contains NaN"
        assert not torch.isnan(k_embeds).any(), "k_embeds contains NaN"

        mul_ij = torch.mul(i_embeds, j_embeds)
        assert not torch.isnan(mul_ij).any(), "mul_ij contains NaN"

        mul_ijk = torch.mul(mul_ij, k_embeds)
        assert not torch.isnan(mul_ijk).any(), "mul_ijk contains NaN"

        assert not torch.isnan(self.w).any(), "w contains NaN"

        gcp = torch.mm(mul_ijk, self.w)
        assert not torch.isnan(gcp).any(), "gcp after matmul contains NaN"

        gcp = F.relu(gcp)
        assert not torch.isnan(gcp).any(), "gcp after ReLU contains NaN"
        
        # # GCP
        # gcp = F.relu(torch.mm(torch.mul(torch.mul(i_embeds, j_embeds), k_embeds), self.w))
        # assert not torch.isnan(gcp).any(), "gcp contains NaN"

        # output product
        x = torch.einsum('ni,nj,nk->nijk', i_embeds, j_embeds, k_embeds)
        # mode product
        x = self.mode(x)
        x = self.flat(x)
        x = torch.cat((gcp, x), 1)
        assert not torch.isnan(x).any(), "x before FC contains NaN"
        x = self.FC(x)
        assert not torch.isnan(x).any(), "x after FC contains NaN"
        return x
    