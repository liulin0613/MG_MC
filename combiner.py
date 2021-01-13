import torch
import torch.nn as nn
import torch.nn.functional as F


# 神经网络注意力机制
class node_net_combiner(nn.Module):
    def __init__(self, embedding_dim, droprate, device="cpu"):
        super(node_net_combiner, self).__init__()

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = device

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax()

    def forward(self, feature1, feature2, n_neighs):
        feature2_reps = feature2.repeat(n_neighs, 1)
        x = torch.cat((feature1, feature2_reps), 1)  # 公式 4
        x = F.relu(self.att1(x).to(self.device), inplace=True)
        x = F.dropout(x, training=self.training, p=self.droprate)
        x = self.att2(x).to(self.device)

        att = F.softmax(x, dim=0)
        return att


# 交叉注意力机制
class node_cross_combiner(nn.Module):
    def __init__(self, embedding_dim, droprate=0.5, device="cpu"):
        super(node_cross_combiner, self).__init__()
        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = device

        self.att1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)

    def forward(self, feature1, feature2, n_neighs):
        feature2_reps = feature2.repeat(n_neighs, 1)
        x = torch.mul(torch.tanh(self.att1(feature1).to(self.device)),
                      torch.tanh(self.att2(feature2_reps).to(self.device)))
        x = F.dropout(x, training=self.training, p=self.droprate)
        x = self.att3(x).to(self.device)
        att = F.softmax(x, dim=0)

        return att


# 组件水平的注意力机制
class cmp_combiner(nn.Module):
    def __init__(self, embedding1, embedding_dim, droprate=0.5, device="cpu"):
        super(cmp_combiner, self).__init__()

        self.embedding1 = embedding1

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = device

        self.layer1 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, nodes_u, nodes_i):
        embedding1 = self.embedding1(nodes_u, nodes_i)
        x = F.relu(self.layer1(embedding1).to(self.device), inplace=True)
        x = F.dropout(x, training=self.training, p=self.droprate)

        # 公式 10
        final_embed_matrix = x
        return final_embed_matrix


# 图级别的组合
class graph_combiner(nn.Module):
    def __init__(self, embedding1, embedding2, embedding_dim, droprate=0.5, device="cpu"):
        super(graph_combiner, self).__init__()
        self.embedding1 = embedding1
        self.embedding2 = embedding2

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = device

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, nodes_u, nodes_i):
        embedding1 = self.embedding1(nodes_u, nodes_i)
        embedding2 = self.embedding2(nodes_u, nodes_i)

        x = torch.cat((embedding1, embedding2), dim=1)
        x = F.relu(self.att1(x).to(self.device), inplace=True)  # 公式 8
        x = F.dropout(x, training=self.training)
        x = self.att2(x).to(self.device)

        att_w = F.softmax(x, dim=1)
        att_w1, att_w2 = att_w.chunk(2, dim=1)
        att_w1.repeat(self.embed_dim, 1)
        att_w2.repeat(self.embed_dim, 1)
        final_embed_matrix = torch.mul(embedding1, att_w1) + torch.mul(embedding2, att_w2)
        return final_embed_matrix
