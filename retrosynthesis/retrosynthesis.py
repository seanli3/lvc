import argparse

import torchdrug as td
from torchdrug import data
import matplotlib
import torch
from torchdrug import core, tasks
from torchdrug import datasets

from custom import SynthonCompletion
from models import LocalWLGNN

from run.custom_graphgym.loader.util import add_hop_info_drug

from torchdrug import data, datasets, utils
from torchdrug.data import Graph


def parse_args():
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='Retrosynthesis')
    parser.add_argument('--hops', dest='hops', type=int, required=True,
                        help='Hops')
    parser.add_argument('--walk', dest='walk', type=str, required=True,
                        help='Walk')
    return parser.parse_args()

args = parse_args()

HOPS = args.hops
WALK = args.walk
synthon_model_path = "g2gs_synthon_model_hops_{}_walk_{}.pth".format(str(HOPS), WALK)
reaction_model_path = "g2gs_reaction_model_hops_{}_walk_{}.pth".format(str(HOPS), WALK)
# HOPS = 3
# WALK = 'DFS'

for h in range(HOPS):
    Graph._meta_types.add('agg_node_index_'+str(h))
    Graph._meta_types.add('agg_scatter_index_'+str(h))


def add_aggregation_info(dataset_raw, hops, walk):
    data_list = [add_hop_info_drug(d, hops, walk) for d in dataset_raw]
    max_path_len = 0
    for d in data_list:
        for g in d['graph']:
            max_len = max(map(lambda key: int(key[-1]), filter(lambda key: 'agg_node_index' in key, dir(g))))
            if max_path_len < max_len:
                max_path_len = max_len
    for d in data_list:
        for g in d['graph']:
            for i in range(max_path_len, -1, -1):
                if 'agg_node_index_' + str(i) not in dir(g) or 'agg_scatter_index_' + str(i) not in dir(g):
                    g.meta_dict['agg_scatter_index_' + str(i)] = {'node reference'}
                    setattr(g, 'agg_scatter_index_' + str(i), torch.LongTensor([]))
                    g.meta_dict['agg_node_index_' + str(i)] = {'node reference'}
                    setattr(g, 'agg_node_index_' + str(i), torch.LongTensor([]))


reaction_dataset = datasets.USPTO50k("~/molecule-datasets/",
                                     atom_feature="center_identification",
                                     kekulize=True)
synthon_dataset = datasets.USPTO50k("~/molecule-datasets/", as_synthon=True,
                                    atom_feature="synthon_completion",
                                    kekulize=True)

from torchdrug.utils import plot
from torch.utils import data as torch_data


torch.manual_seed(1)
reaction_train, reaction_valid, reaction_test = reaction_dataset.split()
add_aggregation_info(reaction_test, HOPS, WALK)
add_aggregation_info(reaction_valid, HOPS, WALK)
add_aggregation_info(reaction_train, HOPS, WALK)

torch.manual_seed(1)
synthon_train, synthon_valid, synthon_test = synthon_dataset.split()
add_aggregation_info(synthon_test, HOPS, WALK)
add_aggregation_info(synthon_valid, HOPS, WALK)
add_aggregation_info(synthon_train, HOPS, WALK)


reaction_model = LocalWLGNN(input_dim=reaction_dataset.node_feature_dim, layers_pre_mp=1,
                            hidden_dim=256, layers_mp=5, hops=HOPS, dropout=0.2,
                            concat_hidden=False, readout='sum')
reaction_task = tasks.CenterIdentification(reaction_model,
                                           feature=("graph", "atom", "bond", "reaction"))

reaction_optimizer = torch.optim.Adam(reaction_task.parameters(), lr=1e-3)
reaction_solver = core.Engine(reaction_task, reaction_train, reaction_valid,
                              reaction_test, reaction_optimizer,
                              batch_size=128)

reaction_solver.train(num_epoch=10)
reaction_solver.evaluate("valid")
reaction_solver.save(reaction_model_path)

synthon_model = LocalWLGNN(input_dim=synthon_dataset.node_feature_dim, layers_pre_mp=1,
                            hidden_dim=256, layers_mp=5, hops=HOPS, dropout=0.2,
                            concat_hidden=False, readout='sum')
synthon_task = SynthonCompletion(synthon_model, feature=("graph", "atom", "reaction",))

synthon_optimizer = torch.optim.Adam(synthon_task.parameters(), lr=1e-3)
synthon_solver = core.Engine(synthon_task, synthon_train, synthon_valid,
                             synthon_test, synthon_optimizer,
                             batch_size=128)
synthon_solver.train(num_epoch=10)
synthon_solver.evaluate("valid")
synthon_solver.save(synthon_model_path)


reaction_task.preprocess(reaction_train, None, None)
synthon_task.preprocess(synthon_train, None, None)
task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=2,
                            num_synthon_beam=10, max_prediction=10)

lengths = [len(reaction_valid) // 10,
           len(reaction_valid) - len(reaction_valid) // 10]
reaction_valid_small = torch_data.random_split(reaction_valid, lengths)[0]

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, reaction_train, reaction_valid_small, reaction_test,
                     optimizer, batch_size=128)

solver.load(reaction_model_path, load_optimizer=False)
solver.load(synthon_model_path, load_optimizer=False)
task.cpu()
solver.evaluate("valid")