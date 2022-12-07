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
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity

def sort_neighbours(G, v, neighbours, node_sim, shortest_distance, visited):
    ret = neighbours
    if cfg['localWL']['sortBy'] == 'degree':
        degrees = list(G.degree(neighbours))
        degrees.sort(key=lambda i: i[1], reverse=cfg['localWL']['reverse'])
        ret = map(lambda d: d[0], degrees)
    elif cfg['localWL']['sortBy'] == 'sim':
        sim = torch.tensor(node_sim[v][neighbours])
        sorted_indices = torch.argsort(sim, descending=cfg['localWL']['reverse'])
        ret = (neighbours[i] for i in sorted_indices)
    if shortest_distance is not None and cfg['localWL']['walk'] == 'dfs':
        # Ensure DFS follows DIT rule
        neighbours_less_distance = set(filter(lambda n: shortest_distance[n] <= shortest_distance[v], neighbours))
        ret = sorted(ret, key=lambda n: 0 if n in neighbours_less_distance else float('inf'))
    # Filter out visited nodes before beam
    ret = filter(lambda n: n not in visited, ret)
    # Beam search
    if cfg['localWL']['beamSize'] is not None:
        ret = (i for i in list(ret)[:cfg['localWL']['beamSize']])
    return ret

def dfs_edges(G, source=None, depth_limit=None, node_sim=None, shortest_distance=None):
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]
    visited = set()
    if depth_limit is None:
        depth_limit = len(G)
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        sorted_neighbours = sort_neighbours(G, start, list(G[start]), node_sim, shortest_distance, visited)
        stack = [(start, depth_limit, sorted_neighbours)]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    if depth_now > 1:
                        # sort node by degree, travese from the smallest degree
                        sorted_neighbours = sort_neighbours(G, child, list(G[child]), node_sim, shortest_distance, visited)
                        stack.append((child, depth_now - 1, sorted_neighbours ))
            except StopIteration:
                stack.pop()


def dfs_successors(G, source=None, depth_limit=None, node_sim=None):
    d = defaultdict(list)
    shortest_distance=nx.shortest_path_length(G, source)
    for s, t in dfs_edges(G, source=source, depth_limit=depth_limit, node_sim=node_sim, shortest_distance=shortest_distance):
        d[s].append(t)
    return dict(d)

def generic_bfs_edges(G, source, neighbors=None, depth_limit=None, node_sim=None):
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)
    sorted_neighbours = sort_neighbours(G, source, list(neighbors(source)), node_sim, None, visited)
    queue = deque([(source, depth_limit, sorted_neighbours)])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    sorted_neighbours = sort_neighbours(G, child, list(neighbors(child)), node_sim, None, visited)
                    queue.append((child, depth_now - 1, sorted_neighbours))
        except StopIteration:
            queue.popleft()

def bfs_edges(G, source, reverse=False, depth_limit=None, node_sim=None):
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    yield from generic_bfs_edges(G, source, successors, depth_limit, node_sim=node_sim)

def bfs_successors(G, source, depth_limit=None, node_sim=None):
    parent = source
    children = []
    for p, c in bfs_edges(G, source, depth_limit=depth_limit, node_sim=node_sim):
        if p == parent:
            children.append(c)
            continue
        yield (parent, children)
        children = [c]
        parent = p
    yield (parent, children)


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
            node_sim = None if cfg['localWL']['sortBy'] != 'sim' else cosine_similarity(batch.node_feature,
                                                                                          batch.node_feature)

            agg_node_index = [[[] for _ in range(batch.node_feature.shape[0])] for _ in range(hops)]

            # nodes = max_reaching_centrality(nx_g, 10)
            nodes = nx_g
            print('vertex cover: {}, original nodes: {}'.format(len(nodes), len(nx_g)))
            for v in nodes:
                # FIXME: bfx_successors only visit nodes once, In a directed graph with edges [(0, 1), (1, 2), (2, 1)], the edge (2, 1) would not be visited
                walk_tree = dict(bfs_successors(nx_g, v, hops, node_sim)) if cfg['localWL']['walk'] == 'bfs' else dict(dfs_successors(nx_g, v, hops, node_sim))
                build_message_passing_node_index(agg_node_index, walk_tree, 0, v, [v])

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
