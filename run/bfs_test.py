from custom_graphgym.loader.util import bfs_successors
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

g = nx.cycle_graph(4)
assert(list(bfs_successors(g, 0))[:2] == list(nx.bfs_successors(g, 0)))
assert(list(bfs_successors(g, 0))[2] == (3, [2]))

g = nx.balanced_tree(2, 3)
assert(list(bfs_successors(g, 0)) == list(nx.bfs_successors(g, 0)))
assert(list(bfs_successors(g, 3)) == list(nx.bfs_successors(g, 3)))
assert(list(bfs_successors(g, 1)) == list(nx.bfs_successors(g, 1)))
# print(list(bfs_successors(g, 0, depth_limit=2)) == list(nx.bfs_successors(g, 0, depth_limit=2)))

g = nx.barbell_graph(4, 5)
# nx.draw_networkx(g)
# plt.show()
assert(list(bfs_successors(g, 0)) == list(nx.bfs_successors(g, 0)))

g = nx.binomial_graph(8, 0.6, seed=2)
assert(list(bfs_successors(g, 0))[:2] == list(nx.bfs_successors(g, 0))[:2])
assert(list(bfs_successors(g, 0))[3] == list(nx.bfs_successors(g, 0))[2])
assert(list(bfs_successors(g, 0)) != list(nx.bfs_successors(g, 0)))
assert(list(bfs_successors(g, 0)) != list(nx.bfs_successors(g, 0)))

g = nx.wheel_graph(5)
# nx.draw_networkx(g)
# plt.show()
assert(list(bfs_successors(g, 0)) == list(nx.bfs_successors(g, 0)))
assert(list(bfs_successors(g, 4)) != list(nx.bfs_successors(g, 4)))
