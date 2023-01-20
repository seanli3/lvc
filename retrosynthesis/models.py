import torch
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torchdrug import core, layers
from torch import nn
from torch.nn import functional as F
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.config import cfg, set_cfg
from torchdrug.layers import MultiLayerPerceptron

set_cfg(cfg)


class LocalWLGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers_pre_mp, layers_mp, hops, dropout,
                 concat_hidden=False, readout="sum"):
        super().__init__()
        self.max_path_length = hops
        self.concat_hidden = concat_hidden
        self.layers_mp = layers_mp
        self.input_dim = input_dim
        self.pre_mp = GNNPreMP(input_dim, hidden_dim,
                               layers_pre_mp)
        # self.output_dim = hidden_dim*layers_mp if concat_hidden else hidden_dim
        self.output_dim = (hidden_dim * self.max_path_length) * self.layers_mp + hidden_dim if concat_hidden else hidden_dim
        self.dropout = dropout

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

        self.eps = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.1) for _ in range(layers_mp)])
        self.layers = nn.ModuleList()
        for l in range(layers_mp):
            self.layers.append(nn.ModuleList([
                 MultiLayerPerceptron(
                     hidden_dim, [hidden_dim,hidden_dim]
                 )
                for _ in range(self.max_path_length)
            ]))

    def forward(self, graph, input, all_loss=None, metric=None):
        x = self.pre_mp(input)
        # x = input

        for l in range(self.layers_mp):
            out = (1 + self.eps[l]) * x
            for hop in range(0, self.max_path_length):
                scatter_index_mask = graph.data_dict['agg_scatter_index_'+str(hop)] != -1
                node_index_mask = graph.data_dict['agg_node_index_'+str(hop)] != -1
                scatter_index = graph.data_dict['agg_scatter_index_'+str(hop)][scatter_index_mask & node_index_mask]
                node_index = graph.data_dict['agg_node_index_'+str(hop)][scatter_index_mask & node_index_mask]
                h_v = x[scatter_index]
                h = x.scatter_reduce(
                        0, node_index.view(-1, 1).broadcast_to(h_v.shape), h_v, reduce='sum',
                        include_self=True)
                h = self.layers[l][hop](h)
                if self.concat_hidden:
                    out = torch.cat([out, h], dim=1)
                else:
                    out = out + h
                # out =  h
                out = F.dropout(out, p=self.dropout, training=self.training)
            x = out

        graph_feature = self.readout(graph, x)

        return {
            "graph_feature": graph_feature,
            "node_feature": x
        }