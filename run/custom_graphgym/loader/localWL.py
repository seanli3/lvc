from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_networkx
from torch_geometric.graphgym.register import register_loader
import networkx as nx
from ogb.graphproppred import PygGraphPropPredDataset
import torch
import pickle
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset, LINKXDataset, WebKB, WikipediaNetwork, Actor, ZINC)
from .util import add_hop_info_pyg
from torch_geometric.graphgym.loader import load_pyg, load_ogb

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

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return

def load_dataset_example(format, name, dataset_dir):
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
            dataset_raw = LINKXDataset(dataset_dir, name=name)
        elif name in ["Cornell", 'Texas', 'Wisconsin']:
            dataset_raw = WebKB(dataset_dir, name=name)
        elif name in ['Chameleon', 'Squirrel']:
            dataset_raw = WikipediaNetwork(dataset_dir, name=name, geom_gcn_preprocess=True)
        elif name in ["Actor"]:
            dataset_raw = Actor(dataset_dir)
        elif name in ['ZINC']:
            train = ZINC(dataset_dir, subset=True, split="train")
            val = ZINC(dataset_dir, subset=True, split="val")
            test = ZINC(dataset_dir, subset=True, split="test")
            dataset_raw = InMemoryDataset()
            train_data = [data for data in train]
            val_data = [data for data in val]
            test_data = [data for data in test]
            dataset_raw.data, dataset_raw.slices = dataset_raw.collate(train_data + val_data + test_data)
            dataset_raw.data.x = torch.nn.functional.one_hot(dataset_raw.data.x.view(-1)).float()
            dataset_raw.data.train_graph_index = torch.arange(len(train))
            dataset_raw.data.val_graph_index = torch.arange(len(train), len(train)+len(val))
            dataset_raw.data.test_graph_index = torch.arange(len(train)+len(val), len(train)+len(val)+len(test))
        else:
            dataset_raw = load_pyg(name, dataset_dir)
    elif format == 'OGB':
        dataset_raw = load_ogb(name.replace('_', '-'), dataset_dir)
    elif format == 'nx':
        try:
            with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
                graphs = pickle.load(file)
        except Exception:
            graphs = nx.read_gpickle('{}/{}.gpickle'.format(dataset_dir, name))
            if not isinstance(graphs, list):
                graphs = [graphs]
        return graphs
    else:
        raise ValueError('Unknown data format: {}'.format(format))

    splits = extract_splits(dataset_raw)
    add_aggregation_info(dataset_raw)
    # Add splits back in after aggregation. Note: this has to be done after aggregation
    add_splits_if_not_available(dataset_raw, splits)

    return dataset_raw


def add_splits_if_not_available(dataset_raw, splits):
    if cfg.dataset.task == 'graph':
        if len(splits) == 0:
            skf = StratifiedKFold(n_splits=10)
            for i, (train_index, val_index) in enumerate(skf.split(dataset_raw.data.y, dataset_raw.data.y)):
                if i == cfg.seed - 1:
                    dataset_raw.data.train_graph_index = torch.LongTensor(train_index)
                    dataset_raw.data.val_graph_index = torch.LongTensor(val_index)
                    dataset_raw.data.test_graph_index = torch.LongTensor(val_index)
        else:
            dataset_raw.data.train_graph_index = splits[0]
            dataset_raw.data.val_graph_index = splits[1]
            dataset_raw.data.test_graph_index = splits[2]


def extract_splits(dataset_raw):
    splits = []
    if cfg.dataset.task == 'graph':
        split_nanes = ['train_graph_index', 'val_graph_index', 'test_graph_index']
        for split_name in split_nanes:
            if split_name in dataset_raw.data:
                splits.append(dataset_raw.data[split_name])
                delattr(dataset_raw.data, split_name)
    return splits


def add_aggregation_info(dataset_raw):
    data_list = [add_hop_info_pyg(d) for d in dataset_raw]
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
    dataset_raw._indices = None
    dataset_raw._data_list = None
    dataset_raw.data, dataset_raw.slices = dataset_raw.collate(data_list)


register_loader('localWL', load_dataset_example)
