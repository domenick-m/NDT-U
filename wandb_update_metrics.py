import wandb
import numpy as np

from collections import defaultdict

ENTITY = 'emory-bg2'
PROJECT = 'overlap_explore'
METRIC_NAME = 'lr'

api = wandb.Api()

run = api.run(f"{ENTITY}/{PROJECT}/1q8j1sib")

groups = set()
metrics = defaultdict(list)

max_metrics = [
    'val ho_lt_co_bps', 
    'val ho_co_bps', 
    'val hi_lt_co_bps', 
    'val hi_co_bps'
]
min_metrics = [
    'train nll', 
    'train lt_nll', 
    'val nll', 
    'val lt_nll'
]
avg_metrics = [
    'val ho_lt_co_bps', 
    'val ho_co_bps', 
    'val hi_lt_co_bps', 
    'val hi_co_bps',
    'val nll', 
    'val lt_nll'
]
for metric in max_metrics:
    if metric in run.summary.keys():
        run.summary[f"{metric}_max"] = np.max(run.summary[metric])

run.summary.update()
exit()

for run in runs:
    if METRIC_NAME in run.summary:
        metrics[run.name].append(run.summary[METRIC_NAME])
        print(run.name, '\n', np.max(run.history()['train lt loss']))
        print(np.min(run.history()['train lt loss']))
        exit()

maximums = {}
minimums = {}
std_devs = {}

for group in metrics.keys():    # Aggregate each group individually
    maximums[group] = np.max(metrics[group])
    minimums[group] = np.min(metrics[group])
    std_devs[group] = np.std(metrics[group])

for run in runs:    # Update runs
    if METRIC_NAME in run.summary:
        run.summary[f"{METRIC_NAME}_max"] = maximums[run.name]
        run.summary[f"{METRIC_NAME}_min"] = minimums[run.name]
        run.summary[f"{METRIC_NAME}_std"] = std_devs[run.name]

        run.summary.update()