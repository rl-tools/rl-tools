import json
import numpy as np
from io import StringIO
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt
from tqdm import tqdm

results_path = "results.json"

with open(results_path, "r") as f:
    results = json.load(f)


data = []
for job in results:
    if job["status"] != "done":
        continue
    csv_data = job["result"]["test_stats"]
    f = StringIO(csv_data)
    job_data = np.genfromtxt(f, delimiter=",", dtype=None, names=True)  # skip header row if present
    for k, v in job["spec"].items():
        job_data = rfn.append_fields(job_data, k, np.full(len(job_data), v), usemask=False)
    data.append(job_data)

data = np.concatenate(data)


filters = {
    "dmodel": ("==", 16),
    "epoch": ("<=", 200),
}
# vertical_axis = "return_mean"
vertical_axis = "episode_length_mean"
horizontal_axis = "epoch"
# group_axis = "dmodel"
group_axis = "model"
aggregation_function = np.mean

def filter_mask(col, filter):
    operation, value = filter
    if operation == "==":
        return col == value
    elif operation == "!=":
        return col != value
    elif operation == ">":
        return col > value
    elif operation == "<":
        return col < value
    elif operation == ">=":
        return col >= value
    elif operation == "<=":
        return col <= value
    else:
        raise ValueError(f"Invalid operation: {operation}")

data = data[np.all([filter_mask(data[k], v) for k, v in filters.items()], axis=0)]

plt.figure()
for group_axis_value in tqdm(np.unique(data[group_axis])):
    group_mask = data[group_axis] == group_axis_value
    horizontal_values = np.unique(data[group_mask][horizontal_axis])
    aggregated_data = [aggregation_function(data[group_mask][data[group_mask][horizontal_axis] == v][vertical_axis]) for v in horizontal_values]
    plt.plot(horizontal_values, aggregated_data, label=group_axis_value)
plt.legend()
plt.show()





