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
from run.custom_graphgym.loader.util import add_walk_info
import torch
import networkx as nx


class DatasetGetter:

    def __init__(self, outer_k=None, inner_k=None):
        self.outer_k = outer_k
        self.inner_k = inner_k

    def set_inner_k(self, k):
        self.inner_k = k

    def get_train_val(self, dataset, batch_size, hops, walk, shuffle=True):
        train_loader, val_loader = dataset.get_model_selection_fold(self.outer_k, self.inner_k, batch_size, shuffle)
        return add_aggregation_info(train_loader, hops, walk), add_aggregation_info(val_loader, hops, walk)

    def get_test(self, dataset, batch_size, hops, walk, shuffle=True):
        test_loader = dataset.get_test_fold(self.outer_k, batch_size, shuffle)
        return add_aggregation_info(test_loader, hops, walk)


def add_aggregation_info(loader, hops, walk):
    data_list = [add_hop_info_pyg(d, hops, walk) for d in loader]
    max_path_len = 0
    for d in data_list:
        max_len = max(map(lambda key: int(key[-1]), filter(lambda key: 'agg_node_index' in key, d.keys)))
        if max_path_len < max_len:
            max_path_len = max_len
    for d in data_list:
        for i in range(max_path_len, -1, -1):
            if 'agg_node_index_' + str(i) not in d or 'agg_scatter_index_' + str(i) not in d:
                d['agg_node_index_' + str(i)] = torch.LongTensor([])
                d['agg_scatter_index_' + str(i)] = torch.LongTensor([])
    return data_list


def add_hop_info_pyg(batch, hops, walk):
    nx_g = nx.Graph(batch.edge_index.T.tolist())
    x = batch.x
    agg_scatter, agg_node_index = add_walk_info(hops, nx_g, None, walk, x)
    for i in range(len(agg_scatter)):
        batch['agg_scatter_index_' + str(i)] = agg_scatter[i]
    for i in range(len(agg_node_index)):
        batch['agg_node_index_' + str(i)] = agg_node_index[i]
    return batch

