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
        self.hops = cfg['localWL']['hops']
        self.walk = cfg['localWL']['walk']

        hops = cfg['localWL']['hops']
        max_path_length = hops
        # for _ in range(hops):
        #     self.lins.append(nn.Linear(dim_in, cfg.gnn.dim_inner))
        self.encoder = None
        if cfg.dataset.task == 'graph' and cfg.dataset.format == 'OGB':
            self.encoder = FeatureEncoder(dim_in)
            dim_in = self.encoder.dim_in
        self.pre_mp = None
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        GNNHead = register.head_dict[cfg.gnn.head]

        dim_inner = cfg.gnn.dim_inner
        self.has_base_hop = self.walk == 'dfs' and hops > 1

        if cfg.localWL.hop_pool == 'cat':
            if self.has_base_hop:
                post_mp_dim_in = (dim_inner*(max_path_length+1))*cfg.gnn.layers_mp + dim_inner
            else:
                post_mp_dim_in = (dim_inner*max_path_length)*cfg.gnn.layers_mp + dim_inner
            self.post_mp = GNNHead(dim_in=post_mp_dim_in, dim_out=dim_out)
        else:
            self.post_mp = GNNHead(dim_in=dim_inner, dim_out=dim_out)

        self.max_path_length = max_path_length
        self.eps = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.1) for _ in range(cfg.gnn.layers_mp)])
        # self.layers = nn.ParameterList()

        if self.has_base_hop:
            self.base_hop_layers = nn.ModuleList()
            for l in range(cfg.gnn.layers_mp):
                in_dim = dim_in if l == 0 else dim_inner
                self.base_hop_layers.append(MLP(
                     new_layer_config(
                         in_dim*(1+(max_path_length+1)*l) if cfg.localWL.hop_pool == 'cat' else in_dim,
                         dim_inner,
                         num_layers=cfg.localWL.mlp_layer,
                         has_act=False,
                         has_bias=False,
                         cfg=cfg
                     )
                 ))

        self.layers = nn.ModuleList()
        for l in range(cfg.gnn.layers_mp):
            # self.layers.append(nn.ParameterDict({
            #     'beta1': nn.ParameterList([nn.Parameter(torch.ones(1)*0.) for _ in range(max_path_length)]),
            #     'beta2': nn.ParameterList([nn.Parameter(torch.ones(1) * 0.) for _ in range(max_path_length)]),
            #     'beta3': nn.ParameterList([nn.Parameter(torch.ones(1) * 0.) for _ in range(max_path_length)]),
            # }))
            in_dim = dim_in if l == 0 else dim_inner
            if cfg.localWL.hop_pool == 'cat':
                if self.has_base_hop:
                    layer_in_dim = in_dim * (1 + (max_path_length + 1) * l)
                else:
                    layer_in_dim = in_dim * (1 + max_path_length * l)
            else:
                layer_in_dim = in_dim
            self.layers.append(nn.ModuleList([
                 MLP(
                     new_layer_config(
                         layer_in_dim,
                         dim_inner,
                         num_layers=cfg.localWL.mlp_layer,
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
        if self.pre_mp is not None:
            batch = self.pre_mp(batch)

        x, edge_index, x_batch, = \
            batch.x, batch.edge_index, batch.batch

        prefix = "" if cfg.dataset.transductive or cfg.dataset.task == 'graph' or not self.training else "train_"

        for l in range(len(self.layers)):
            out = (1 + self.eps[l]) * x
            if self.has_base_hop:
                h_v = x[batch[prefix + 'agg_scatter_base_index']]
                h = x.scatter_reduce(
                    0, batch[prefix + 'agg_node_base_index'].view(-1, 1).broadcast_to(h_v.shape), h_v,
                    reduce=cfg.localWL.pool,
                    include_self=True)
                h = self.base_hop_layers[l](h)
                h = F.dropout(h, p=cfg.localWL.dropout, training=self.training)
                # h = h + (1+layer['beta2'][hop]*x)
                # h = (1+layer['beta3'][hop])*h
                if cfg.localWL.hop_pool == 'cat':
                    out = torch.cat([out, h], dim=1)
                elif cfg.localWL.hop_pool == 'sum':
                    out = out + h

            for hop in range(0, self.max_path_length):
                # h_v = (1+layer['beta1'][hop])*x[batch['agg_scatter_index_'+str(hop)]]
                h_v = x[batch[prefix + 'agg_scatter_index_'+str(hop)]]
                h = x.scatter_reduce(
                        0, batch[prefix + 'agg_node_index_'+ str(hop)].view(-1, 1).broadcast_to(h_v.shape), h_v, reduce=cfg.localWL.pool,
                        include_self=True)
                h = self.layers[l][hop](h)
                h = F.dropout(h, p=cfg.localWL.dropout, training=self.training)
                # h = h + (1+layer['beta2'][hop]*x)
                # h = (1+layer['beta3'][hop])*h
                if cfg.localWL.hop_pool == 'cat':
                    out = torch.cat([out, h], dim=1)
                elif cfg.localWL.hop_pool == 'sum':
                    out = out + h
                # out =  h
            out = F.dropout(out, p=cfg.gnn.dropout, training=self.training)
            x = out

        batch.x = out
        batch = self.post_mp(batch)

        return batch

