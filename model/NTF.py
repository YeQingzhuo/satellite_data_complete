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


class NTF(nn.Module):
    def __init__(self, num_user, num_item, num_time, latent_dim, period, is_history=False):
        super(NTF, self).__init__()
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_user, latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_item, latent_dim)))
        self.time_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_time+period, latent_dim)))

        self.LSTM = nn.LSTM(latent_dim, latent_dim, 1, batch_first=True)  # (input_size/feature/rank, latent_dim, num_layers)
        self.FC = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                nn.BatchNorm1d(latent_dim))
        self.MLP = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

        self.is_history = is_history

    
        self.history_prob = nn.Parameter(torch.abs(torch.tensor(0.5)))

    def forward(self, i_input, j_input, k_input, ks_input=None, is_history=False):
        if is_history:
            if t.rand(1).item() < self.history_prob.item():
                return t.zeros(i_input.size(0), 1).to(i_input.device)
                
        i_embeds = self.user_embeddings[i_input]
        j_embeds = self.item_embeddings[j_input]
        k_embeds = self.time_embeddings[ks_input]    
        
        _, (k_embeds, _) = self.LSTM(k_embeds)
        if k_embeds.size(1) ==1:
            k_embeds = torch.sigmoid(self.FC(k_embeds.squeeze(0)))
        else:
            k_embeds = torch.sigmoid(self.FC(k_embeds.squeeze()))

        inputs = torch.cat((i_embeds, j_embeds, k_embeds), dim=1)
        xijk = torch.sigmoid(self.MLP(inputs))
        return xijk