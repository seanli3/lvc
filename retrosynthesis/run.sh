PYTHONPATH=/home/sean/graphgym python retrosynthesis.py --hops 1 --walk BFS > results/hops_1_walk_DFS.txt &
PYTHONPATH=/home/sean/graphgym python retrosynthesis.py --hops 2 --walk BFS > results/hops_2_walk_DFS.txt &
PYTHONPATH=/home/sean/graphgym python retrosynthesis.py --hops 3 --walk BFS > results/hops_3_walk_DFS.txt &
