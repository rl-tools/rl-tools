import json
import numpy as np
from io import StringIO
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt

results_path = "results.json"

with open(results_path, "r") as f:
    results = json.load(f)


data = []
for job in results:
    if job["status"] != "done":
        continue
    f = StringIO(job["result"]["test_stats"])
    job_data = np.genfromtxt(f, delimiter=",", names=True)  # skip header row if present
    for k, v in job["spec"].items():
        job_data = rfn.append_fields(job_data, k, np.full(len(job_data), v), usemask=False)
    data.append(job_data)

data = np.concatenate(data)


default_mask = data["dmodel"] == 16
seed_data = {seed:data[default_mask][data[default_mask]["seed"] == seed] for seed in np.unique(data[default_mask]["seed"])}
assert all(len(np.unique(v["epoch"])) == len(np.unique(seed_data[list(seed_data.keys())[0]]["epoch"])) for v in seed_data.values()), "Epochs must be unique"

vertical_axis = "return_mean"
horizontal_axis = "epoch"
# group_axis = "dmodel"
group_axis = "model"
aggregation_function = np.mean

plt.figure()
for group_axis_value in np.unique(data[group_axis]):
    group_mask = data[group_axis] == group_axis_value
    horizontal_values = np.unique(data[group_mask][horizontal_axis])
    aggregated_data = [aggregation_function(data[group_mask][data[group_mask][horizontal_axis] == v][vertical_axis]) for v in horizontal_values]
    plt.plot(horizontal_values, aggregated_data, label=group_axis_value)
plt.legend()
plt.show()





