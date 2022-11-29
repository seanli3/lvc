import logging
import os

import torch
from torch_geometric import seed_everything
import networkx as nx

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device

def build_message_passing_node_index(agg_node_scatter, tree, level, v, parents):
    for parent in parents:
        if parent in tree:
            for child in tree[parent]:
                agg_node_scatter[level][child].append(parent)
            build_message_passing_node_index(agg_node_scatter, tree, level + 1, v, tree[parent])


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        setup_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        datasets = create_dataset()
        loaders = create_loader(datasets)

        def add_hop_info(batch):
            nx_g = nx.Graph(batch.edge_index.T.tolist())
            hops = cfg['localWL']['hops']

            agg_node_index = [[[] for _ in range(batch.node_feature.shape[0])] for _ in range(hops)]

            # nodes = max_reaching_centrality(nx_g, 10)
            nodes = nx_g
            print('vertex cover: {}, original nodes: {}'.format(len(nodes), len(nx_g)))
            for v in nodes:
                build_message_passing_node_index(agg_node_index, dict(nx.bfs_successors(nx_g, v, hops)), 0, v, [v])

            agg_scatter = [
                torch.tensor([j for j in range(batch.node_feature.shape[0]) for _ in level_node_index[j]], device=cfg.device)
                for
                level_node_index in agg_node_index]

            agg_node_index = [torch.tensor([i for j in agg_node_index_k for i in j], device=cfg.device) for
                              agg_node_index_k
                              in agg_node_index]

            batch.agg_scatter = agg_scatter
            batch.agg_node_index = agg_node_index
            return batch

        loaders = list(map(lambda loader: list(map(add_hop_info, loader)), loaders))

        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
