# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, data_size, device, forecast_moodel, args):
        super(RNN, self).__init__()
        self.init_lin_h = nn.Linear(args.noise_dim, args.latent_dim)
        self.init_lin_c = nn.Linear(args.noise_dim, args.latent_dim)
        self.init_input = nn.Linear(args.noise_dim, args.latent_dim)

        self.rnn = nn.LSTM(args.latent_dim, args.latent_dim, args.num_rnn_layer)

        self.forecast_moodel = forecast_moodel

        E_dim = 0
        for n, p in forecast_moodel.named_parameters():
            E_dim += torch.numel(p)
            torch.nn.init.normal_(p)

        # Transforming LSTM output to vector shape
        self.lin_transform_down = nn.Sequential(
                            nn.Linear(args.latent_dim, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, E_dim))
        # Transforming vector to LSTM input shape
        self.lin_transform_up = nn.Sequential(
                            nn.Linear(E_dim, args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(args.hidden_dim, args.latent_dim))
        
        self.num_rnn_layer = args.num_rnn_layer
        self.data_size = data_size
        self.device = device

    def nn_construction(self, E):
        offset = 0
        for n, p in self.forecast_moodel.named_parameters():
            numel = torch.numel(p)
            weight = E[:, offset:offset + numel]
            weight = weight.reshape(p.shape)
            self.forecast_moodel.state_dict()[n].data = weight
            offset += numel
    
    def forward(self, X, z, E=None, hidden=None):
        if hidden == None and E == None:
            init_c, init_h = [], []
            for _ in range(self.num_rnn_layer):
                init_c.append(torch.tanh(self.init_lin_h(z)))
                init_h.append(torch.tanh(self.init_lin_c(z)))
            # Initialize hidden inputs for the LSTM
            hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        
            # Initialize an input for the LSTM
            inputs = torch.tanh(self.init_input(z))
        else:
            inputs = self.lin_transform_up(E)

        out, hidden = self.rnn(inputs.unsqueeze(0), hidden)

        E = self.lin_transform_down(out.squeeze(0))
        self.nn_construction(E)
        pred = self.forecast_moodel(X)
        
        return E, hidden, pred