from copy import copy

from torch_geometric.graphgym.config import cfg
import networkx as nx
import torch
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity

def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs.update(set((nbr for n in nbrs for nbr in G[n])))
    return nbrs

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

def dfs_successors(G, source=None, dist_limit=None, depth_limit=None, node_sim=None, sort_neighbours=None):
    visited = {source}
    visit_time = {source: 0}
    if depth_limit is None:
        depth_limit = float('inf')
    if dist_limit is None:
        kneighbors = G.nodes
    else:
        kneighbors = knbrs(G, source, dist_limit)
    kneighbors = set(kneighbors)

    if sort_neighbours:
        sorted_neighbours = sort_neighbours(G, source, list(G.neighbors(source)), node_sim, None, visited)
    else:
        sorted_neighbours = G.neighbors(source)
    stack = [(source, depth_limit, sorted_neighbours)]
    back_edges = set()
    tree_edges = set()
    ret = defaultdict(set)
    while stack:
        parent, depth_now, neighbours = stack[-1]
        visited.add(parent)
        try:
            child = next(neighbours)
            if child in kneighbors:
                if child in visited:
                    if (child, parent) not in tree_edges and (child, parent) not in back_edges:
                        back_edges.add((parent, child))
                else:
                    visit_time[child] = max(visit_time.values()) + 1
                    ret[child].add(parent)
                    tree_edges.add((parent, child))
                    if depth_now > 1:
                        if sort_neighbours:
                            sorted_neighbours = sort_neighbours(G, child, list(G[child]), node_sim, None, visited)
                        else:
                            sorted_neighbours = G.neighbors(child)
                        stack.append((child, depth_now - 1, sorted_neighbours))
        except StopIteration:
            stack.pop()

    b_u = sigma_df(kneighbors, back_edges, visit_time)
    for k in ret:
        if k in b_u:
            ret[k].update(b_u[k])
    return ret


def find_crossover_bedges(back_edges, visit_time):
    crossover_bedges = defaultdict(set)
    for e1 in back_edges:
        for e2 in back_edges:
            if visit_time[e1[0]] >= visit_time[e2[0]] > visit_time[e1[1]] >= visit_time[e2[1]]:
                crossover_bedges[e1].add(e2)
                crossover_bedges[e2].add(e1)
            elif visit_time[e1[1]] <= visit_time[e2[1]] < visit_time[e1[0]] <= visit_time[e2[0]]:
                crossover_bedges[e1].add(e2)
                crossover_bedges[e2].add(e1)
    return crossover_bedges


def find_cover_bedges(v, bedges, visit_time):
    for e in bedges:
        if visit_time[e[1]] <= visit_time[v] <= visit_time[e[0]]:
            yield e


def find_covered_vertices(be, visit_time, inv_visit_time):
    low_time = visit_time[be[1]]
    up_time = visit_time[be[0]]
    for t in range(low_time, up_time+1):
        if t in inv_visit_time:
            yield inv_visit_time[t]


def sigma_df(vertices, bedges, visit_time):
    sigma = defaultdict(set)
    inv_visit_time = {v: k for k, v in visit_time.items()}
    crossover_bedges = find_crossover_bedges(bedges, visit_time)
    for v in vertices:
        es = find_cover_bedges(v, bedges, visit_time)
        for e in es:
            cross_edges = crossover_bedges[e]
            for ce in cross_edges:
                covered_vertices = find_covered_vertices(ce, visit_time, inv_visit_time)
                sigma[v].update(covered_vertices)
    return sigma







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

def build_bfs_message_passing_node_index(agg_node_scatter, tree, depth_limit, level, v, parents):
    for parent in parents:
        if parent in tree:
            for child in tree[parent]:
                while level >= len(agg_node_scatter):
                    agg_node_scatter.append([[] for _ in range(len(agg_node_scatter[0]))])
                agg_node_scatter[level][child].append(parent)
                tmp = [child]
                if level < depth_limit+1:
                    for l in range(level+1, depth_limit):
                        while l >= len(agg_node_scatter):
                            agg_node_scatter.append([[] for _ in range(len(agg_node_scatter[0]))])
                        chidren = []
                        for t in tmp :
                            if t in tree:
                                for c in tree[t]:
                                    chidren.append(c)
                                    agg_node_scatter[l][c].append((parent))
                        tmp = chidren
            build_bfs_message_passing_node_index(agg_node_scatter, tree, depth_limit, level + 1, v, tree[parent])


def build_dfs_message_passing_node_index(G, agg_node_scatter, dist_limit, v):
    vertice_dist = nx.single_source_shortest_path_length(G, v, dist_limit)
    sigma = dfs_successors(G, v, dist_limit=dist_limit, node_sim=None)
    for n, dist in vertice_dist.items():
        if n == 5 and dist == 1:
            print()
        if dist == 0:
            continue
        if dist > len(agg_node_scatter):
            agg_node_scatter.append([set() for _ in range(len(agg_node_scatter[0]))])
        agg_node_scatter[dist-1][n].update(sigma[n])
        children = set()
        tmp = sigma[n]
        for d in range(dist-1, 0, -1):
            for t in tmp:
                agg_node_scatter[dist-1][n].update(sigma[t])
                children.update(sigma[t])
            tmp = copy(children)


def build_reverse_walk(walk):
    new_dic = defaultdict(list)
    for k, v in walk.items():
        for x in v:
            new_dic[x].append(k)
    return new_dic


def add_hop_info(batch):
    nx_g = nx.Graph(batch.edge_index.T.tolist())
    hops = cfg['localWL']['hops']
    node_sim = None if cfg['localWL']['sortBy'] != 'sim' else cosine_similarity(batch.x,
                                                                                batch.x)

    if cfg['localWL']['walk'] == 'bfs':
        agg_node_index = [[[] for _ in range(batch.x.shape[0])]]
    else:
        agg_node_index = [[set() for _ in range(batch.x.shape[0])]]

    # nodes = max_reaching_centrality(nx_g, 10)
    nodes = nx_g
    # print('vertex cover: {}, original nodes: {}'.format(len(nodes), len(nx_g)))
    for v in nodes:
        if cfg['localWL']['walk'] == 'bfs':
            walk_tree = dict(bfs_successors(nx_g, v, depth_limit=hops, node_sim=node_sim))
            build_bfs_message_passing_node_index(agg_node_index, walk_tree, hops, 0, v, [v])
        else:
            build_dfs_message_passing_node_index(nx_g, agg_node_index, hops, v)

    agg_scatter = [
        torch.tensor([j for j in range(batch.x.shape[0]) for _ in level_node_index[j]], device=batch.x.device)
        for
        level_node_index in agg_node_index]

    agg_node_index = [torch.tensor([i for j in list(agg_node_index_k) for i in j], device=batch.x.device) for
                      agg_node_index_k
                      in agg_node_index]

    for i in range(len(agg_scatter)):
        batch['agg_scatter_index_'+str(i)] = agg_scatter[i]
    for i in range(len(agg_node_index)):
        batch['agg_node_index_'+str(i)] = agg_node_index[i]
    return batch


