#!/usr/bin/env bash

CONFIG=localWL
GRID=localWL
REPEAT=2
MAX_JOBS=8

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
PYTHONPATH=/home/sean/graphgym python configs_gen.py --config configs/${CONFIG}.yaml \
  --grid grids/${GRID}.txt \
  --out_dir configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS

# aggregate results for the batch
PYTHONPATH=/home/sean/graphgym python agg_batch.py --dir results/${CONFIG}_grid_${GRID}
