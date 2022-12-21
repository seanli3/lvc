from torch_geometric.graphgym.config import cfg
import networkx as nx
import torch
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

def dfs_edges(G, source=None, dist_limit=None, depth_limit=None, node_sim=None, shortest_distance=None):
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
                if child not in visited and shortest_distance[child] <= dist_limit:
                    yield parent, child
                    visited.add(child)
                    if depth_now > 1:
                        # sort node by degree, travese from the smallest degree
                        sorted_neighbours = sort_neighbours(G, child, list(G[child]), node_sim, shortest_distance, visited)
                        stack.append((child, depth_now - 1, sorted_neighbours ))
            except StopIteration:
                stack.pop()


def dfs_successors(G, source=None, dist_limit=None, depth_limit=None, node_sim=None):
    # # Max path length is the number of nodes
    # sys.setrecursionlimit(G.number_of_nodes() + depth_limit)
    d = defaultdict(list)
    shortest_distance=nx.shortest_path_length(G, source)
    for s, t in dfs_edges(G, dist_limit=dist_limit, depth_limit=depth_limit, source=source, node_sim=node_sim, shortest_distance=shortest_distance):
        d[s].append(t)
    return dict(d)

def bfs_successors(G, source, depth_limit=None, node_sim=None, sort_neighbours=None):
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)
    if sort_neighbours:
        sorted_neighbours = sort_neighbours(G, source, list(G.neighbors(source)), node_sim, None, visited)
    else:
        sorted_neighbours = G.neighbors(source)
    queue = deque([(source, 1, sorted_neighbours)])
    depth_now = 0
    depth_now_visited = {source}
    children = []
    while queue:
        depth_next = queue[0][1]
        if depth_now < depth_next:
            visited.update(depth_now_visited)
            depth_now_visited = set()
        parent, depth_now, neighbours = queue[0]
        try:
            child = next(neighbours)
            if child not in visited:
                children.append(child)
                # avoid adding the same child twice (when there is a loop)
                if depth_now < depth_limit and child not in depth_now_visited:
                    if sort_neighbours:
                        sorted_neighbours = sort_neighbours(G, child, list(G.neighbors(child)), node_sim, None, visited)
                    else:
                        sorted_neighbours = G.neighbors(child)
                    queue.append((child, depth_now + 1, sorted_neighbours))
                depth_now_visited.add(child)
        except StopIteration:
            queue.popleft()
            if children:
                yield (parent, children)
            children = []

def build_message_passing_node_index(agg_node_scatter, tree, level, v, parents):
    for parent in parents:
        if parent in tree:
            for child in tree[parent]:
                while level >= len(agg_node_scatter):
                    agg_node_scatter.append([[] for _ in range(len(agg_node_scatter[0]))])
                agg_node_scatter[level][child].append(parent)
            build_message_passing_node_index(agg_node_scatter, tree, level + 1, v, tree[parent])

def add_hop_info(batch):
    nx_g = nx.Graph(batch.edge_index.T.tolist())
    hops = cfg['localWL']['hops']
    node_sim = None if cfg['localWL']['sortBy'] != 'sim' else cosine_similarity(batch.x,
                                                                                batch.x)

    agg_node_index = [[[] for _ in range(batch.x.shape[0])]]

    # nodes = max_reaching_centrality(nx_g, 10)
    nodes = nx_g
    # print('vertex cover: {}, original nodes: {}'.format(len(nodes), len(nx_g)))
    for v in nodes:
        # FIXME: bfx_successors only visit nodes once, In a directed graph with edges [(0, 1), (1, 2), (2, 1)], the edge (2, 1) would not be visited
        walk_tree = dict(bfs_successors(nx_g, v, depth_limit=hops, node_sim=node_sim)) \
            if cfg['localWL']['walk'] == 'bfs' else \
            dict(dfs_successors(nx_g, v, dist_limit=hops,
                                depth_limit=cfg['localWL']['maxPathLen'],node_sim=node_sim))
        build_message_passing_node_index(agg_node_index, walk_tree, 0, v, [v])

    agg_scatter = [
        torch.tensor([j for j in range(batch.x.shape[0]) for _ in level_node_index[j]], device=batch.x.device)
        for
        level_node_index in agg_node_index]

    agg_node_index = [torch.tensor([i for j in agg_node_index_k for i in j], device=batch.x.device) for
                      agg_node_index_k
                      in agg_node_index]

    for i in range(len(agg_scatter)):
        batch['agg_scatter_index_'+str(i)] = agg_scatter[i]
    for i in range(len(agg_node_index)):
        batch['agg_node_index_'+str(i)] = agg_node_index[i]
    return batch


