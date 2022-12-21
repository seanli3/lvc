import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.layer import MLP
import networkx as nx
from torch_geometric.graphgym.register import pooling_dict
from copy import deepcopy

class GNNGraphHead(nn.Module):
    '''Head of GNN, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out):
        super(GNNGraphHead, self).__init__()
        # todo: PostMP before or after global pooling
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)
        self.pooling_fun = pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.graph_label

    def forward(self, batch):
        if cfg.dataset.transform == 'ego':
            graph_emb = self.pooling_fun(batch.x, batch.batch,
                                         batch.node_id_index)
        else:
            graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

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
            return batch.x[batch.node_label_index], batch.node_label
        else:
            return batch.x[batch.node_label_index], \
                   batch.node_label[batch.node_label_index]

    def forward(self, batch):
        batch.x = self.layer_post_mp(batch.x)
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

@register_network('localWL_graph')
class LocalWLGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(dim_in, cfg.gnn.dim_inner))
        hops = cfg['localWL']['hops']
        for _ in range(hops):
            self.lins.append(nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner))
        self.eps = nn.Parameter(torch.ones(1, device=cfg['device']))
        self.post_mp = GNNGraphHead(dim_in=cfg.gnn.dim_inner*(hops+1), dim_out=dim_out)

    def forward(self, b, loader):
        batch = deepcopy(b)
        x, edge_index, x_batch, = \
            batch.x, batch.edge_index, batch.batch
        agg_scatter = batch.agg_scatter
        agg_node_index = batch.agg_node_index

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

        batch.x = out
        batch = self.post_mp(batch)

        return batch

