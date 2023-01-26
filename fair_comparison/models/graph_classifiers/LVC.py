#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import torch
import torch.nn.functional as F
from torch_geometric.graphgym.models.layer import MLP, LayerConfig
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import new_layer_config
from copy import deepcopy



class LVC(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(LVC, self).__init__()
        self.cfg = config
        cfg = config
        dim_in = dim_features
        dim_out = dim_target
        hops = cfg['hops']
        max_path_length = hops

        self.encoder = None
        self.pre_mp = None
        if cfg.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.dim_inner,
                                   cfg.layers_pre_mp)
            dim_in = cfg.dim_inner
        GNNHead = register.head_dict[cfg.head]

        dim_inner = cfg.dim_inner

        if cfg.hop_pool == 'cat':
            import math
            self.post_mp = GNNHead(dim_in=(dim_inner * max_path_length) * cfg.layers_mp + dim_inner,
                                   dim_out=dim_out)
        else:
            self.post_mp = GNNHead(dim_in=dim_inner, dim_out=dim_out)

        self.max_path_length = max_path_length
        self.eps = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.1) for _ in range(cfg.layers_mp)])
        # self.layers = nn.ParameterList()
        self.layers = nn.ModuleList()
        for l in range(cfg.layers_mp):
            # self.layers.append(nn.ParameterDict({
            #     'beta1': nn.ParameterList([nn.Parameter(torch.ones(1)*0.) for _ in range(max_path_length)]),
            #     'beta2': nn.ParameterList([nn.Parameter(torch.ones(1) * 0.) for _ in range(max_path_length)]),
            #     'beta3': nn.ParameterList([nn.Parameter(torch.ones(1) * 0.) for _ in range(max_path_length)]),
            # }))
            in_dim = dim_in if l == 0 else dim_inner
            self.layers.append(nn.ModuleList([
                MLP(
                    LayerConfig(
                        dim_in=in_dim * (1 + max_path_length * l) if cfg.hop_pool == 'cat' else in_dim,
                        dim_out=dim_inner,
                        num_layers=cfg.mlp_layer,
                        has_act=False,
                        has_bias=False,
                    )
                )
                for _ in range(max_path_length)
            ]))

    def forward(self, data):
        cfg = self.cfg
        batch = deepcopy(data)
        if self.encoder is not None:
            batch = self.encoder(batch)
        if self.pre_mp is not None:
            batch = self.pre_mp(batch)

        x, edge_index, x_batch, = \
            batch.x, batch.edge_index, batch.batch
        for l in range(len(self.layers)):
            out = (1 + self.eps[l]) * x
            for hop in range(0, self.max_path_length):
                # h_v = (1+layer['beta1'][hop])*x[batch['agg_scatter_index_'+str(hop)]]
                h_v = x[batch['agg_scatter_index_' + str(hop)]]
                h = x.scatter_reduce(
                    0, batch['agg_node_index_' + str(hop)].view(-1, 1).broadcast_to(h_v.shape), h_v,
                    reduce=cfg.pool,
                    include_self=True)
                h = self.layers[l][hop](h)
                h = F.dropout(h, p=cfg.ldropout, training=self.training)
                # h = h + (1+layer['beta2'][hop]*x)
                # h = (1+layer['beta3'][hop])*h
                if cfg.hop_pool == 'cat':
                    out = torch.cat([out, h], dim=1)
                elif cfg.hop_pool == 'sum':
                    out = out + h
                # out =  h
            out = F.dropout(out, p=cfg.dropout, training=self.training)
            x = out

        batch.x = out
        batch = self.post_mp(batch)

        return batch