import torch
import torch.nn as nn
import torch.nn.functional as F
from graphgym.config import cfg
from graphgym.register import register_network
from graphgym.models.layer import MLP
import networkx as nx


class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''
    def __init__(self, dim_in, dim_out):
        super(GNNNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)

    def _apply_index(self, batch):
        if batch.node_label_index.shape[0] == batch.node_label.shape[0]:
            return batch.node_feature[batch.node_label_index], batch.node_label
        else:
            return batch.node_feature[batch.node_label_index], \
                   batch.node_label[batch.node_label_index]

    def forward(self, batch):
        batch.node_feature = self.layer_post_mp(batch.node_feature)
        pred, label = self._apply_index(batch)
        return pred, label


def max_degree(graph, k):
    return map(lambda x: x[0], sorted(graph.degree, key=lambda x: x[1], reverse=True)[:k])


def max_reaching_centrality(graph, k):
    q = []
    for v in graph:
        centrality = nx.local_reaching_centrality(graph, v, nx.shortest_path(graph, v))
        if len(q) < k:
            q.append((v,centrality))
        else:
            if centrality > q[0][1]:
                q.pop(0)
                q.append((v,centrality))
            else:
                continue
        q = sorted(q, key=lambda x:x[1])
    return list(map(lambda x:x[0], q))

@register_network('localWL')
class LocalWLGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(dim_in, cfg.gnn.dim_inner))
        hops = cfg['localWL']['hops']
        for _ in range(hops):
            self.lins.append(nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner))
        self.eps = nn.Parameter(torch.ones(1, device=cfg['device']))
        self.post_mp = GNNNodeHead(dim_in=cfg.gnn.dim_inner*(hops+1), dim_out=dim_out)

    def forward(self, batch, loader):
        x, edge_index, x_batch, = \
            batch.node_feature, batch.edge_index, batch.batch
        agg_scatter = loader.dataset[0].agg_scatter
        agg_node_index = loader.dataset[0].agg_node_index

        x = self.lins[0](x)
        out = (1+self.eps)*x
        h = x
        for hop in range(1, len(agg_node_index) + 1):
            h = h[agg_scatter[hop-1]]
            h = torch.zeros(x.shape[0], h.shape[1], device=x.device).scatter_reduce(
                    0, agg_node_index[hop-1].view(-1, 1).broadcast_to(h.shape), h, reduce=cfg.localWL.pool,
                    include_self=False)
            out = torch.cat([out, h], dim=1)
        out = F.dropout(out, p=cfg.localWL.dropout, training=self.training)

        batch.node_feature = out
        batch = self.post_mp(batch)

        return batch

