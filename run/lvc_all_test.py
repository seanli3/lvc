from custom_graphgym.loader.util import bfs_successors, build_bfs_message_passing_node_index,\
    build_reverse_walk, dfs_successors, build_dfs_message_passing_node_index
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

nx_g = nx.Graph()
nx_g.add_edges_from([(0,1),(1,2),(2,5),(5,0),(2,3),(0,4)])
agg_node_index = [[[] for _ in range(len(nx_g))]]

# for v in nx_g:
#     walk_tree = dict(bfs_successors(nx_g, v, depth_limit=5, node_sim=None))
#     inverted_walk = build_reverse_walk(walk_tree)
#     print(v, walk_tree, inverted_walk)
#     build_bfs_message_passing_node_index(agg_node_index, walk_tree, 5, 0, v, [v])
# print(agg_node_index)

agg_node_index = [[set() for _ in range(len(nx_g))]]
for v in nx_g:
    build_dfs_message_passing_node_index(nx_g, agg_node_index, 2, v)
print(agg_node_index)

nx.draw_networkx(nx_g)
plt.show()
