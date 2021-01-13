import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, embedding_dim, aggregator, neighs_feature, is_user_part=True, device="cpu"):
        super(encoder, self).__init__()

        self.embed_dim = embedding_dim
        self.aggregator = aggregator
        self.is_user = is_user_part
        self.neighs_feature = neighs_feature
        self.device = device

        # self.layer = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.att1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 3)

    def forward(self, nodes_u, nodes_i):
        if self.is_user:
            neighs_feature = self.neighs_feature(nodes_u)
            nodes_fea, embed_matrix = self.aggregator(nodes_u, neighs_feature)
            embed_matrix = embed_matrix[nodes_u.cpu().numpy()]
            combined = torch.cat((nodes_fea, neighs_feature, embed_matrix), dim=1)
        else:
            neighs_feature = self.neighs_feature(nodes_i)
            nodes_fea, embed_matrix = self.aggregator(nodes_i, neighs_feature)
            embed_matrix = embed_matrix[nodes_i.cpu().numpy()]
            combined = torch.cat((nodes_fea, neighs_feature,embed_matrix), dim=1)

        x = F.relu(self.att1(combined).to(self.device), inplace=True)  # 公式 8
        x = F.dropout(x, training=self.training)
        x = self.att2(x).to(self.device)

        att_w = F.softmax(x, dim=1)
        att_w1, att_w2, att_w3 = att_w.chunk(3, dim=1)
        att_w1.repeat(self.embed_dim, 1)
        att_w2.repeat(self.embed_dim, 1)
        att_w3.repeat(self.embed_dim, 1)

        cmp_embed_matrix = torch.mul(nodes_fea, att_w1) + torch.mul(neighs_feature, att_w2) + torch.mul(
            embed_matrix, att_w3)

        return cmp_embed_matrix
