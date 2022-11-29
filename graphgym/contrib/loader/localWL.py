from deepsnap.dataset import GraphDataset
from graphgym.config import cfg
from torch_geometric.utils import to_networkx
from graphgym.register import register_loader
import networkx as nx
from ogb.graphproppred import PygGraphPropPredDataset
from graphgym.cmd_args import parse_args
import torch
import pickle
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset, LINKXDataset, WebKB, WikipediaNetwork, Actor)

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
    args = parse_args()

    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset_raw = Planetoid(dataset_dir, name)
        elif name[:3] == 'TU_':
            # TU_IMDB doesn't have node features
            if name[3:] == 'IMDB':
                name = 'IMDB-MULTI'
                dataset_raw = TUDataset(dataset_dir, name, transform=T.Constant())
            else:
                dataset_raw = TUDataset(dataset_dir, name[3:])
                if dataset_raw.data.num_node_features  == 0:
                    max_degree = 0
                    degs = []
                    for data in dataset_raw:
                        degs += [degree(data.edge_index[0], dtype=torch.long)]
                        max_degree = max(max_degree, degs[-1].max().item())

                    if max_degree < 1000:
                        dataset_raw.transform = T.OneHotDegree(max_degree)
                    else:
                        deg = torch.cat(degs, dim=0).to(torch.float)
                        mean, std = deg.mean().item(), deg.std().item()
                        dataset_raw.transform = NormalizedDegree(mean, std)
            # TU_dataset only has graph-level label
            # The goal is to have synthetic tasks
            # that select smallest 100 graphs that have more than 200 edges
            if cfg.dataset.tu_simple and cfg.dataset.task != 'graph':
                size = []
                for data in dataset_raw:
                    edge_num = data.edge_index.shape[1]
                    edge_num = 9999 if edge_num < 200 else edge_num
                    size.append(edge_num)
                size = torch.tensor(size)
                order = torch.argsort(size)[:100]
                dataset_raw = dataset_raw[order]
        elif name == 'Karate':
            dataset_raw = KarateClub()
        elif 'Coauthor' in name:
            if 'CS' in name:
                dataset_raw = Coauthor(dataset_dir, name='CS')
            else:
                dataset_raw = Coauthor(dataset_dir, name='Physics')
        elif 'Amazon' in name:
            if 'Computers' in name:
                dataset_raw = Amazon(dataset_dir, name='Computers')
            else:
                dataset_raw = Amazon(dataset_dir, name='Photo')
        elif name == 'MNIST':
            dataset_raw = MNISTSuperpixels(dataset_dir)
        elif name == 'PPI':
            dataset_raw = PPI(dataset_dir)
        elif name == 'QM7b':
            dataset_raw = QM7b(dataset_dir)
            graphs = GraphDataset.pyg_to_graphs(dataset_raw)
            return graphs
        elif name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
            dataset_raw = LINKXDataset(dataset_dir, name=name)
        elif name in ["Cornell", 'Texas', 'Wisconsin']:
            dataset_raw = WebKB(dataset_dir, name=name)
        elif name in ['Chameleon', 'Squirrel']:
            dataset_raw = WikipediaNetwork(dataset_dir, name=name, geom_gcn_preprocess=True)
        elif name in ["Actor"]:
            dataset_raw = Actor(dataset_dir)
        else:
            raise ValueError('{} not support'.format(name))
    elif format == 'nx':
        try:
            with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
                graphs = pickle.load(file)
        except Exception:
            graphs = nx.read_gpickle('{}/{}.gpickle'.format(dataset_dir, name))
            if not isinstance(graphs, list):
                graphs = [graphs]
        return graphs

        # Load from OGB formatted data
    elif cfg.dataset.format == 'OGB':
        if cfg.dataset.name == 'ogbg-molhiv':
            dataset_raw = PygGraphPropPredDataset(name=cfg.dataset.name)
            graphs = GraphDataset.pyg_to_graphs(dataset_raw)
        # Note this is only used for custom splits from OGB
        split_idx = dataset_raw.get_idx_split()
        return graphs, split_idx

    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    return graphs

register_loader('localWL', load_dataset_example)
