import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

experiments_path = "experiments"

# experiments = list(os.walk(experiments_path))


path = Path(experiments_path)
latest_trajectories = sorted(path.rglob(f'*zoo*/*/*/steps/*/trajectories.json.gz'))
latest_runs = np.unique([trajectories.parent.parent.parent for trajectories in latest_trajectories])


N = 5
run_selection = latest_runs[-N:][::-1]
print(f"Last runs:")
[print(f"  {i}: {run}") for i, run in enumerate(run_selection)]
input_string = input(f"Select run (0-{N-1})[0]: ")
run_index = int(input_string) if input_string else 0

run = run_selection[run_index]
print(f"Selected run: {run}")
steps = sorted(os.listdir(os.path.join(run, "steps")))

# create subplots
plots_per_row = 3
fig, axs = plt.subplots(len(steps)//plots_per_row+(1 if len(steps) % plots_per_row != 0 else 0), plots_per_row) #, figsize=(20, 20))

for ax, step in zip(axs.flatten(), steps):
    trajectories_path = os.path.join(run, "steps", step, "trajectories.json.gz")
    data = None
    with gzip.open(trajectories_path, "rt") as f:
        data = json.load(f)


    actions = [[step["action"] for step in episode] for episode in data]
    actions = np.array(actions)
    ax.hist(actions.flatten(), bins=100)
    ax.set_xlim(-1, 1)
    ax.set_title(f"Step {int(step)}")
fig.suptitle(f"Action distribution ({run})")
plt.tight_layout()
plt.show()