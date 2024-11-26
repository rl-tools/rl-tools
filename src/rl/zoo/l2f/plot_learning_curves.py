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
    return steps/(steps.max() / time), mean_returns, std_returns


experiments = {
    "baseline": ("experiments/2024-11-26_11-04-29/dd10193_zoo_environment_algorithm/l2f_sac", 1.38),
    "challenger": ("experiments/2024-11-26_11-55-37/f06eafb_zoo_environment_algorithm/l2f_sac", 1.38),
    "challenger2": ("experiments/2024-11-26_11-51-39/f06eafb_zoo_environment_algorithm/l2f_sac", 1.38)
}



plt.figure()
for experiment_name, experiment in experiments.items():
    steps, mean_returns, std_returns = load_experiment(*experiment)
    plt.plot(steps, mean_returns, label=experiment_name)
    plt.fill_between(steps, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
plt.xlabel("Time")
plt.ylabel("Returns")
plt.title("Learning curve")
plt.legend()
plt.show()
