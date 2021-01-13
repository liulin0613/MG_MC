import torch
import heapq
import random
import torch.nn as nn
from combiner import node_cross_combiner as attention
from l0dense import L0Dense


class aggregator(nn.Module):
    """
    aggregator: for aggregating feature of neighbors
    """

    def __init__(self, u_feature, i_feature, adj, embed_dim, droprate=0.5, is_user_part=True, weight_decay=0.0005,
                 device="cpu"):
        super(aggregator, self).__init__()

        self.ufeature = u_feature
        self.ifeature = i_feature
        self.adj = adj
        self.embed_dim = embed_dim
        self.droprate = droprate
        self.is_user = is_user_part
        self.device = device
        self.weight_decay = weight_decay

        self.u_layer = L0Dense(self.ufeature.embedding_dim, self.embed_dim,
                               weight_decay=self.weight_decay, droprate=self.droprate, device=self.device)
        self.i_layer = L0Dense(self.ifeature.embedding_dim, self.embed_dim,
                               weight_decay=self.weight_decay, droprate=self.droprate, device=self.device)

        self.att = attention(self.embed_dim, self.droprate, device=self.device)

    def forward(self, nodes, n_feature):
        if self.is_user:
            embed_matrix = torch.empty(self.ufeature.num_embeddings, self.embed_dim,
                                       dtype=torch.float).to(self.device)
            nodes_fea = self.u_layer(self.ufeature.weight[nodes]).to(self.device)

            # average degree
            threshold = 24
        else:
            embed_matrix = torch.empty(self.ifeature.num_embeddings, self.embed_dim,
                                       dtype=torch.float).to(self.device)
            nodes_fea = self.i_layer(self.ifeature.weight[nodes]).to(self.device)
            threshold = 12

        length = list(nodes.size())
        for i in range(length[0]):
            index = nodes[[i]].cpu().numpy()

            if self.training:
                interactions = self.adj[index[0]]
            else:
                if index[0] in self.adj.keys():
                    interactions = self.adj[index[0]]
                else:
                    if self.is_user:
                        node_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(index)]).to(self.device)
                    else:
                        node_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(index)]).to(self.device)

                    embed_matrix[index] = node_feature
                    continue

            n = len(interactions)
            if n < threshold:
                interactions = [item[0] for item in interactions]
                times = n
            else:
                interactions = a_res(interactions, threshold)
                times = threshold

            if self.is_user:
                # item part
                neighs_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(index)]).to(self.device)
            else:
                # item part
                neighs_feature = self.u_layer(self.ufeature.weight[torch.LongTensor(interactions)]).to(self.device)
                node_feature = self.i_layer(self.ifeature.weight[torch.LongTensor(index)]).to(self.device)

            # 注意力机制
            att_w = self.att(neighs_feature, torch.mean(node_feature + n_feature[i].view(1, -1), dim=0), times)  # 公式3
            embedding = torch.mm(neighs_feature.t(), att_w)  # 公式 5
            embed_matrix[index] = embedding.t()

        return nodes_fea, embed_matrix


def a_res(samples, m):
    """
    samples: [(entity, weight), ...]
    m: number of selected entities
    returns: [(entity, weight), ...]
    """
    heap = []
    for sample in samples:
        w = sample[1]
        u = random.uniform(0, 1)
        k = u ** (1 / w)

        if len(heap) < m:
            heapq.heappush(heap, (k, sample))
        elif k > heap[0][0]:
            heapq.heappush(heap, (k, sample))

            if len(heap) > m:
                heapq.heappop(heap)
    return [item[1][0] for item in heap]
