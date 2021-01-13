import torch
from torch import nn

from aggregator import aggregator
from encoder import encoder
from combiner import cmp_combiner
from combiner import node_cross_combiner as attention
from l0dense import L0Dense


def multiGraph(u_feature, i_feature, u_neighs, i_neighs, u_adj, i_adj, embed_dim, droprate=0.5, weight_decay=0.0005,
               device="cpu"):

    u_embed = multi_graph(u_feature.to(device), u_neighs, embed_dim, device=device, weight_decay=weight_decay)
    i_embed = multi_graph(i_feature.to(device), i_neighs, embed_dim, device=device, weight_decay=weight_decay)

    # user part
    u_agg_embed_cmp1 = aggregator(u_feature.to(device), i_feature.to(device), u_adj, embed_dim, droprate=droprate,
                                  device=device, weight_decay=weight_decay)
    u_embed_cmp1 = encoder(embed_dim, u_agg_embed_cmp1, u_embed, device=device)

    u_embed = cmp_combiner(u_embed_cmp1, embed_dim, droprate, device=device)

    # item part
    i_agg_embed_cmp1 = aggregator(u_feature.to(device), i_feature.to(device), i_adj, embed_dim, droprate=droprate,
                                  is_user_part=False, device=device, weight_decay=weight_decay)
    i_embed_cmp1 = encoder(embed_dim, i_agg_embed_cmp1, i_embed, is_user_part=False, device=device)

    i_embed = cmp_combiner(i_embed_cmp1, embed_dim, droprate, device=device)

    return u_embed, i_embed


class multi_graph(nn.Module):
    def __init__(self, feature, neighs, embed_dim, device="cpu", droprate=0.5, weight_decay=0.0005):
        super(multi_graph, self).__init__()

        self.feature = feature
        self.neighs = neighs
        self.embed_dim = embed_dim
        self.device = device
        self.droprate = droprate
        self.weight_decay = weight_decay

        self.layer = L0Dense(self.feature.embedding_dim, self.embed_dim,
                             weight_decay=self.weight_decay, droprate=self.droprate, device=self.device)

        self.att = attention(self.embed_dim, self.droprate, device=self.device)

    def forward(self, nodes):
        embed_matrix = torch.empty(self.feature.num_embeddings, self.embed_dim, dtype=torch.float).to(self.device)
        length = list(nodes.size())
        for i in range(length[0]):
            index = nodes[[i]].cpu().numpy()
            if self.training:
                neighs = self.neighs[index[0]]
            else:
                if index[0] in self.neighs.keys():
                    neighs = self.neighs[index[0]]
                else:
                    node_feature = self.layer(self.feature.weight[torch.LongTensor(index)]).to(self.device)
                    embed_matrix[index] = node_feature
                    continue

            n = len(neighs)

            neighs_feature = self.layer(self.feature.weight[torch.LongTensor(neighs)]).to(self.device)
            node_feature = self.layer(self.feature.weight[torch.LongTensor(index)]).to(self.device)
            if n == 0:
                embed_matrix[index] = node_feature
                continue

            # 注意力机制
            att_w = self.att(neighs_feature, node_feature, n)  # 公式3
            embedding = torch.mm(neighs_feature.t(), att_w)  # 公式 5
            embed_matrix[index] = embedding.t()

        return embed_matrix[nodes.cpu().numpy()]
