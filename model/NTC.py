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

class NTC(nn.Module):
    def __init__(self, num_user, num_item, num_time, latent_dim, channels, is_history=False):
        super(NTC, self).__init__()
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time, latent_dim)))

        self.Conv = nn.Sequential(  # input: [batch, channel, depth, height, width]
            nn.Conv3d(1, channels, 2, 2),  # in_channels, out_channels, kernel_size, stride, padding=0
            nn.ReLU(inplace=True),
            # nn.Dropout3d(0.2),
            nn.Conv3d(channels, channels, 2, 2),     # , kernel_size=2, stride=2
            nn.ReLU(inplace=True),
            # nn.Dropout3d(0.2),
            nn.Conv3d(channels, channels, 2, 2),
            nn.ReLU(inplace=True),
        )
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels*((latent_dim // (2 ** 3)) ** 3), 1)
            # nn.Linear(channels, 1)
        )

        self.is_history = is_history

        self.history_prob = nn.Parameter(torch.abs(torch.tensor(0.5)))

    def forward(self, i_input, j_input, k_input, ks_input=None, is_history=False):
        
        if is_history:
            if t.rand(1).item() < self.history_prob.item():
                return t.zeros(i_input.size(0), 1).to(i_input.device)

        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[k_input]
        
        xijk = torch.einsum('ni,nj,nk->nijk', i_embeds, j_embeds, k_embeds)
        xijk = xijk.unsqueeze(1)
        xijk = self.Conv(xijk)
        xijk = self.FC(xijk)
        xijk = torch.sigmoid(xijk)
        return xijk
