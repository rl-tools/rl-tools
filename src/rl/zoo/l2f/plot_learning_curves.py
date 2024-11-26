import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_experiment(experiment, time):
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

    mean_returns = np.mean(returns, axis=1)
    std_returns = np.std(returns, axis=1)
    wall_time = steps/(steps.max() / time)
    return steps, wall_time, mean_returns, std_returns


experiments = {
    "baseline": ("experiments/2024-11-26_11-51-39/f06eafb_zoo_environment_algorithm/l2f_sac", 1.38),
    "challenger": ("experiments/2024-11-26_12-00-52/f06eafb_zoo_environment_algorithm/l2f_sac", 2.45),
    "challenger2": ("experiments/2024-11-26_12-05-50/f06eafb_zoo_environment_algorithm/l2f_sac", 2.45),
}



use_wall_time = False
plt.figure()
for experiment_name, experiment in experiments.items():
    steps, wall_time, mean_returns, std_returns = load_experiment(*experiment)
    x = wall_time if use_wall_time else steps
    plt.plot(x, mean_returns, label=experiment_name)
    plt.fill_between(x, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
plt.xlabel("Time") if use_wall_time else plt.xlabel("Step")
plt.ylabel("Returns")
plt.title("Learning curve")
plt.legend()
plt.show()
