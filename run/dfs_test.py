from custom_graphgym.loader.util import dfs_successors
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

g = nx.cycle_graph(4)
ret_0 = list(dfs_successors(g, 1))
ret_1 = list(dfs_successors(g, 1))
assert(len(ret_0) == len(ret_1))
for i in range(len(ret_0)):
    assert(ret_0[i] == ret_1[i])

g = nx.balanced_tree(2, 3)
# nx.draw_networkx(g)
# plt.show()
ret_0 = dict(dfs_successors(g, 0))
ret_1 = dict(dfs_successors(g, 1))
for k in ret_0:
    if k != 0 and k!=1:
        assert(ret_1[k] == ret_0[k])

g = nx.barbell_graph(4, 5)
# nx.draw_networkx(g)
# plt.show()
ret_0 = dict(dfs_successors(g, 0))
ret_1 = dict(dfs_successors(g, 12))
# print(ret_0)
# print(ret_1)
for i in range(1,4):
    assert(ret_0[i] == {0,1,2,3})
assert (ret_0[9] == {8,9,10,11,12})
for i in range(10,13):
    assert(ret_0[i] == {9,10,11,12})
for i in range(1, 3):
    assert (ret_1[i] == {0, 1, 2, 3})
assert (ret_1[3] == {0, 1, 2, 3, 4})

g = nx.binomial_graph(8, 0.6, seed=2)
# nx.draw_networkx(g)
ret_0 = dict(dfs_successors(g, 0))
ret_1 = dict(dfs_successors(g, 4))
# print(ret_0)
# print(ret_1)
# plt.show()
for i in ret_0:
    assert(ret_0[i] == {0,1,2,3,4,5,6,7})
for i in ret_1:
    assert(ret_1[i] == {0,1,2,3,4,5,6,7})
# assert(list(bfs_successors(g, 0))[:2] == list(nx.bfs_successors(g, 0))[:2])
# assert(list(bfs_successors(g, 0))[3] == list(nx.bfs_successors(g, 0))[2])
# assert(list(bfs_successors(g, 0)) != list(nx.bfs_successors(g, 0)))
# assert(list(bfs_successors(g, 0)) != list(nx.bfs_successors(g, 0)))
#
g = nx.wheel_graph(5)
nx.draw_networkx(g)
plt.show()
ret_0 = dict(dfs_successors(g, 0))
ret_1 = dict(dfs_successors(g, 4))
print(ret_0)
print(ret_1)
for i in ret_0:
    assert(ret_0[i] == {0,1,2,3,4})
for i in ret_1:
    assert(ret_1[i] == {0,1,2,3,4})
