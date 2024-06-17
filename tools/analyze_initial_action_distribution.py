import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import os


run = "experiments/2024-06-17_17-02-34/8c56263_zoo_algorithm_environment/sac_acrobot-swingup-v0/0000"

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