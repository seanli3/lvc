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
    child_parent = {}
    while stack:
        parent, depth_now, neighbours = stack[-1]
        visited.add(parent)
        try:
            child = next(neighbours)
            if child in kneighbors:
                if child in visited:
                    if (child, parent) not in tree_edges and (child, parent) not in back_edges and child != parent:
                        back_edges.add((parent, child))
                else:
                    visit_time[child] = max(visit_time.values()) + 1
                    child_parent[child] = parent
                    tree_edges.add((parent, child))
                    if depth_now > 1:
                        if sort_neighbours:
                            sorted_neighbours = sort_neighbours(G, child, list(G[child]), node_sim, None, visited)
                        else:
                            sorted_neighbours = G.neighbors(child)
                        stack.append((child, depth_now - 1, sorted_neighbours))
        except StopIteration:
            stack.pop()

    b_u = sigma_df(kneighbors, back_edges, visit_time, child_parent)
    ret = defaultdict(set)
    for k in child_parent:
        ret[k].add(child_parent[k])
        if k in b_u:
            ret[k].update(b_u[k])
    return ret


def find_crossover_bedges(back_edges, visit_time):
    crossover_bedges = defaultdict(set)
    for e1 in back_edges:
        for e2 in back_edges:
            if visit_time[e1[0]] > visit_time[e2[0]] > visit_time[e1[1]] > visit_time[e2[1]]:
                crossover_bedges[e1].add(e2)
                crossover_bedges[e2].add(e1)
            elif visit_time[e1[1]] < visit_time[e2[1]] < visit_time[e1[0]] < visit_time[e2[0]]:
                crossover_bedges[e1].add(e2)
                crossover_bedges[e2].add(e1)
    return crossover_bedges


def find_cover_bedges(v, bedges, child_parent):
    for e in bedges:
        path = bedge_path(e, child_parent)
        if v in path:
            yield e


def bedge_path(e, child_parent):
    path = set()
    path.add(e[0])
    v = child_parent[e[0]]
    path.add(v)
    while v != e[1]:
        v = child_parent[v]
        path.add(v)
    return path


def find_covered_vertices(be, child_parent):
    return bedge_path(be, child_parent)


def sigma_df(vertices, bedges, visit_time, child_parent):
    sigma = defaultdict(set)
    crossover_bedges = find_crossover_bedges(bedges, visit_time)
    for v in vertices:
        es = find_cover_bedges(v, bedges, child_parent)
        for e in es:
            sigma[v].update(find_covered_vertices(e, child_parent))
            cross_edges = crossover_bedges[e]
            for ce in cross_edges:
                covered_vertices = find_covered_vertices(ce, child_parent)
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


def build_dfs_message_passing_node_index(G, agg_node_scatter, dist_limit, v, depth_limit=None):
    vertice_dist = nx.single_source_shortest_path_length(G, v, dist_limit)
    sigma = dfs_successors(G, v, dist_limit=dist_limit, depth_limit=depth_limit, node_sim=None)
    for n, dist in vertice_dist.items():
        if dist == 0:
            continue
        if dist > len(agg_node_scatter):
            agg_node_scatter.append([[] for _ in range(len(agg_node_scatter[0]))])
        agg_node_scatter[dist-1][n] += list(sigma[n])
        children = set()
        tmp = sigma[n]
        for d in range(dist-1, 0, -1):
            for t in tmp:
                agg_node_scatter[dist-1][n] += list(sigma[t])
                children.update(sigma[t])
            tmp = copy(children)


def build_reverse_walk(walk):
    new_dic = defaultdict(list)
    for k, v in walk.items():
        for x in v:
            new_dic[x].append(k)
    return new_dic


def add_hop_info_pyg(batch):
    nx_g = nx.Graph(batch.edge_index.T.tolist())
    x = batch.x
    hops = cfg['localWL']['hops']
    walk = cfg['localWL']['walk']
    sort_by = cfg['localWL']['sortBy']
    depth_limit = cfg['localWL']['maxPathLen']
    if walk == 'dfs' and hops > 1:
        agg_scatter, agg_node_index = add_walk_info(1, nx_g, sort_by, walk, x)
        batch['agg_scatter_base_index'] = agg_scatter[0]
        batch['agg_node_base_index'] = agg_node_index[0]
    agg_scatter, agg_node_index = add_walk_info(hops, nx_g, sort_by, walk, x, depth_limit=depth_limit)
    for i in range(len(agg_scatter)):
        batch['agg_scatter_index_' + str(i)] = agg_scatter[i]
    for i in range(len(agg_node_index)):
        batch['agg_node_index_' + str(i)] = agg_node_index[i]
    return batch


def add_hop_info_drug(pair, hops, walk, sort_by=None):
    for g in pair['graph']:
        nx_g = nx.Graph(g.edge_list.t()[:2].t().tolist())
        x = g.node_feature
        agg_scatter, agg_node_index = add_walk_info(hops, nx_g, sort_by, walk, x)
        for i in range(len(agg_scatter)):
            g.meta_dict['agg_scatter_index_' + str(i)] = {'node reference'}
            setattr(g, 'agg_scatter_index_' + str(i), agg_scatter[i])
        for i in range(len(agg_node_index)):
            g.meta_dict['agg_node_index_' + str(i)] = {'node reference'}
            setattr(g, 'agg_node_index_' + str(i), agg_node_index[i])
    return pair


def add_walk_info(hops, nx_g, sort_by, walk, x, depth_limit=None):
    node_sim = None if sort_by != 'sim' else cosine_similarity(x, x)
    if walk == 'bfs':
        agg_node_index = [[[] for _ in range(x.shape[0])]]
    else:
        agg_node_index = [[[] for _ in range(x.shape[0])]]
    # nodes = max_reaching_centrality(nx_g, 10)
    nodes = nx_g
    # print('vertex cover: {}, original nodes: {}'.format(len(nodes), len(nx_g)))
    for v in nodes:
        if walk == 'bfs':
            walk_tree = dict(bfs_successors(nx_g, v, depth_limit=hops, node_sim=node_sim))
            build_bfs_message_passing_node_index(agg_node_index, walk_tree, hops, 0, v, [v])
        else:
            build_dfs_message_passing_node_index(nx_g, agg_node_index, hops, v, depth_limit=depth_limit)
    agg_scatter = [
        torch.LongTensor([j for j in range(x.shape[0]) for _ in level_node_index[j]], device=x.device)
        for
        level_node_index in agg_node_index]
    agg_node_index = [torch.LongTensor([i for j in list(agg_node_index_k) for i in j], device=x.device) for
                      agg_node_index_k
                      in agg_node_index]
    return agg_scatter, agg_node_index


