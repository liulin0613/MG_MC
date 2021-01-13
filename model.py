from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
from l0dense import L0Dense


class MG_MC(nn.Module):
    def __init__(self, u_embedding, i_embedding, embed_dim, N=30000, droprate=0.5,
                 beta_ema=0.999, device="cpu"):
        super(MG_MC, self).__init__()

        self.u_embed = u_embedding
        self.i_embed = i_embedding
        self.embed_dim = embed_dim
        self.N = N
        self.droprate = droprate
        self.beta_ema = beta_ema
        self.device = device

        self.u_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.u_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.i_layer1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.i_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.ui_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.ui_layer2 = nn.Linear(self.embed_dim, 1)

        self.u_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.i_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.ui_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)

        if beta_ema > 0.:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            self.avg_param = [a.to(self.device) for a in self.avg_param]
            self.steps_ema = 0.

        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_i):
        u_embed = self.u_embed(nodes_u, nodes_i)
        i_embed = self.i_embed(nodes_u, nodes_i)

        x_u = F.relu(self.u_bn(self.u_layer1(u_embed)), inplace=True)
        x_u = F.dropout(x_u, training=self.training, p=self.droprate)
        x_u = self.u_layer2(x_u)

        x_i = F.relu(self.i_bn(self.i_layer1(i_embed)), inplace=True)
        x_i = F.dropout(x_i, training=self.training, p=self.droprate)
        x_i = self.i_layer2(x_i)

        x_ui = torch.cat((x_u, x_i), dim=1)
        x = F.relu(self.ui_bn(self.ui_layer1(x_ui)), inplace=True)
        x = F.dropout(x, training=self.training, p=self.droprate)

        scores = self.ui_layer2(x)
        return scores.squeeze()

    def regularization(self):
        regularization = 0
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        return regularization

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def loss(self, nodes_u, nodes_i, ratings):
        scores = self.forward(nodes_u, nodes_i)
        loss = self.criterion(scores, ratings)

        total_loss = loss + self.regularization()
        return total_loss
