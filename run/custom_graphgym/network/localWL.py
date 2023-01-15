import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
import networkx as nx
from torch_geometric.graphgym.register import pooling_dict
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP

from copy import deepcopy


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
        # self.lins = nn.ModuleList()
        # self.lins.append(nn.Linear(dim_in, cfg.gnn.dim_inner))
        # self.lin =  nn.Linear(dim_in, cfg.gnn.dim_inner)

        hops = cfg['localWL']['hops']
        max_path_length = hops if cfg['localWL']['walk'] == 'bfs' else cfg['localWL']['maxPathLen']
        # for _ in range(hops):
        #     self.lins.append(nn.Linear(dim_in, cfg.gnn.dim_inner))
        self.encoder = None
        if cfg.dataset.task == 'graph' and cfg.dataset.format == 'OGB':
            self.encoder = FeatureEncoder(dim_in)
            dim_in = self.encoder.dim_in
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        GNNHead = register.head_dict[cfg.gnn.head]

        if cfg.localWL.hop_pool == 'cat':
            import math
            self.post_mp = GNNHead(dim_in=(dim_in*max_path_length)*cfg.gnn.layers_mp + dim_in, dim_out=dim_out)
        else:
            self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)


        self.max_path_length = max_path_length
        self.eps = nn.Parameter(torch.ones(1) * 0.1)
        # self.layers = nn.ParameterList()
        self.layers = nn.ModuleList()
        for l in range(cfg.gnn.layers_mp):
            # self.layers.append(nn.ParameterDict({
            #     'beta1': nn.ParameterList([nn.Parameter(torch.ones(1)*0.) for _ in range(max_path_length)]),
            #     'beta2': nn.ParameterList([nn.Parameter(torch.ones(1) * 0.) for _ in range(max_path_length)]),
            #     'beta3': nn.ParameterList([nn.Parameter(torch.ones(1) * 0.) for _ in range(max_path_length)]),
            # }))
            self.layers.append(nn.ModuleList([
                 MLP(
                     new_layer_config(
                         dim_in*(1+max_path_length*l) if cfg.localWL.hop_pool == 'cat' else dim_in,
                         dim_in,
                         num_layers=2,
                         has_act=False,
                         has_bias=False,
                         cfg=cfg
                     )
                 )
                for _ in range(max_path_length)
            ]))

    def forward(self, batch):
        if self.encoder is not None:
            batch= self.encoder(batch)
        batch = self.pre_mp(batch)

        x, edge_index, x_batch, = \
            batch.x, batch.edge_index, batch.batch
        for layer in self.layers:
            out = (1 + self.eps) * x
            for hop in range(0, self.max_path_length):
                # h_v = (1+layer['beta1'][hop])*x[batch['agg_scatter_index_'+str(hop)]]
                h_v = x[batch['agg_scatter_index_'+str(hop)]]
                h = x.scatter_reduce(
                        0, batch['agg_node_index_'+ str(hop)].view(-1, 1).broadcast_to(h_v.shape), h_v, reduce=cfg.localWL.pool,
                        include_self=True)
                h = layer[hop](h)
                # h = h + (1+layer['beta2'][hop]*x)
                # h = (1+layer['beta3'][hop])*h
                if cfg.localWL.hop_pool == 'cat':
                    out = torch.cat([out, h], dim=1)
                elif cfg.localWL.hop_pool == 'sum':
                    out = out + h
                # out =  h
                out = F.dropout(out, p=cfg.localWL.dropout, training=self.training)
            x = out

        batch.x = out
        batch = self.post_mp(batch)

        return batch

