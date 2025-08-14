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
def inference_flops(architecture):
    flops = 2 * (architecture[0] + 1) * architecture[1] # 2x for FMAC, +1 for bias addition
    flops += 2 * (architecture[1] + 1) * architecture[1] * 2 * 3 + architecture[1] * 4
    flops += 2 * (architecture[1] + 1) * architecture[2]
    return flops
inference_flops_dmodel = np.vectorize(lambda dmodel: inference_flops([3 + 9 + 3 + 3 + 4, dmodel, 4]))
data = rfn.append_fields(data, "compute", inference_flops_dmodel(data["dmodel"]), usemask=False)


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

label_map = {
    "episode_length_mean": "Episode Length [steps]",
    "compute": "Compute [FLOPS]"
}

def plot(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function, scatter=False):
    data = data[np.all([filter_mask(data[k], v) for k, v in filters.items()], axis=0)]

    plt.figure()
    for group_axis_value in tqdm(np.unique(data[group_axis])) if group_axis is not None else [None]:
        group_mask = data[group_axis] == group_axis_value if group_axis is not None else np.ones(len(data), dtype=bool)
        horizontal_values = np.unique(data[group_mask][horizontal_axis])
        aggregated_data = [aggregation_function(data[group_mask][data[group_mask][horizontal_axis] == v][vertical_axis]) for v in horizontal_values]
        (plt.scatter if scatter else plt.plot)(horizontal_values, aggregated_data, label=group_axis_value)
        plt.xlabel(label_map[horizontal_axis] if horizontal_axis in label_map else horizontal_axis)
        plt.ylabel(label_map[vertical_axis] if vertical_axis in label_map else vertical_axis)
    plt.legend() if group_axis else None
    plt.show()

# filters = {
#     "dmodel": ("==", 64),
#     "epoch": ("<=", 200),
# }
# # vertical_axis = "return_mean"
# vertical_axis = "episode_length_mean"
# horizontal_axis = "epoch"
# # group_axis = "dmodel"
# group_axis = "model"
# aggregation_function = np.mean

# plot(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function)



filters = {
    # "epoch": ("<=", 200),
    "model": ("==", "\"crazyflie\"")
}
# vertical_axis = "return_mean"
vertical_axis = "episode_length_mean"
horizontal_axis = "compute"
group_axis = "dmodel"
# group_axis = "model"
# group_axis = None
aggregation_function = np.mean

plot(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function, scatter=True)





