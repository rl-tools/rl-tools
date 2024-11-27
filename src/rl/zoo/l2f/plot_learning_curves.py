import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_experiment(experiment, time, lower_percentile=0, upper_percentile=100):
    steps = []
    steps_set = False
    returns = []

    for seed in os.listdir(experiment):
        run_path = os.path.join(experiment, seed)
        if os.path.exists(os.path.join(run_path, "return.json.set")):
            with open(os.path.join(run_path, "return.json")) as f:
                results = json.load(f)
                for i, evaluation_step in enumerate(results):
                    if not steps_set:
                        steps.append(evaluation_step["step"])
                        returns.append([])
                    returns[i].append(evaluation_step["returns"])
            steps_set = True

    returns = np.array(returns)
    returns = returns.reshape((len(steps), -1))
    steps = np.array(steps)

    # mean_returns = np.mean(returns, axis=1)
    # std_returns = np.std(returns, axis=1)
    wall_time = steps/(steps.max() / time)
    upper_percentile_values = np.percentile(returns, upper_percentile, axis=1)
    lower_percentile_values = np.percentile(returns, lower_percentile, axis=1)
    iqm_mask = np.logical_and(returns > lower_percentile_values[:, None], returns < upper_percentile_values[:, None])
    iqm_data = np.where(iqm_mask, returns, np.nan)
    iqm_mean_returns = np.nanmean(iqm_data, axis=1)
    iqm_std_returns = np.nanstd(iqm_data, axis=1)
    return steps, wall_time, iqm_mean_returns, iqm_std_returns


experiments = {
    # "baseline": ("experiments/2024-11-26_14-33-33/c91e9c4_zoo_environment_algorithm/l2f_sac", 1.144),
    # "50k": ("experiments/2024-11-26_16-12-12/c91e9c4_zoo_environment_algorithm/l2f_sac", 0.844),
    # "50k_+1-layer": ("experiments/2024-11-26_16-15-58/c91e9c4_zoo_environment_algorithm/l2f_sac", 0.844),
    "aftera-all": ("experiments/2024-11-26_20-35-04/d56db9c_zoo_environment_algorithm/l2f_sac/", 0.099)
}


combinations = [
    (True, (0, 100)),
    (True, (25, 75)),
    (True, (90, 100)),
    (False, (0, 100)),
    (False, (25, 75)),
    (False, (90, 100)),
]

cols = 3
fig, axes = plt.subplots(2, cols, figsize=(12, 10))

for idx, (use_wall_time, (iqm_lower_percentile, iqm_upper_percentile)) in enumerate(combinations):
    ax = axes[idx // cols, idx % cols]
    for experiment_name, experiment in experiments.items():
        steps, wall_time, mean_returns, std_returns = load_experiment(*experiment, lower_percentile=iqm_lower_percentile, upper_percentile=iqm_upper_percentile)
        x = wall_time if use_wall_time else steps
        ax.plot(x, mean_returns, label=experiment_name)
        ax.fill_between(x, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
    
    xlabel = "Time [s]" if use_wall_time else "Step"
    ylabel = "Returns"
    title = f"Learning Curve ({'Wall Time' if use_wall_time else 'Steps'}, {f'IQM({iqm_lower_percentile}%:{iqm_upper_percentile}%)'})"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
