import json
import numpy as np
from io import StringIO
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt
from tqdm import tqdm

def load(results_path):
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
            if k == "teacher_selection":
                continue
            job_data = rfn.append_fields(job_data, k, np.full(len(job_data), v), usemask=False)
        if "teacher_selection" not in job_data.dtype.names:
            job_data = rfn.append_fields(job_data, "teacher_selection", np.full(len(job_data), "all").astype("S6"), usemask=False)
        job_data["model"] = [m.decode("utf-8").strip('"').encode("utf-8") for m in job_data["model"]]
        job_data[job_data["model"] == b"race"]["model"] = b"arpl"
        data.append(job_data)

    data = np.concatenate(data)
    return data

def inference_flops(architecture):
    flops = 2 * (architecture[0] + 1) * architecture[1] # 2x for FMAC, +1 for bias addition
    flops += 2 * (architecture[1] + 1) * architecture[1] * 2 * 3 + architecture[1] * 4
    flops += 2 * (architecture[1] + 1) * architecture[2]
    return flops
inference_flops_dmodel = np.vectorize(lambda dmodel: inference_flops([3 + 9 + 3 + 3 + 4, dmodel, 4]))



results_paths = ["results0.json", "results1.json"]
datas = [load(results_path) for results_path in results_paths]
for i, data in enumerate(datas):
    if i == 0:
        continue
    datas[i] = np.rec.fromarrays([data[n] for n in datas[0].dtype.names], names=datas[0].dtype.names)

data =np.concatenate(datas)
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
    elif operation == "in":
        return np.array([c in value for c in col])
    else:
        raise ValueError(f"Invalid operation: {operation}")

label_map = {
    "episode_length_mean": "Episode Length [steps]",
    "compute": "Inference Compute [FLOPS]"
}

def process(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function):
    data = data[np.all([filter_mask(data[k], v) for k, v in filters.items()], axis=0)]

    output_axes = ["return_mean", "return_std", "episode_length_mean", "episode_length_std", "share_terminated", "compute"]
    horizontal_mask = data[horizontal_axis] == data[horizontal_axis][0]
    group_mask = data[group_axis] == data[group_axis][0] if group_axis is not None else np.ones(len(data), dtype=bool)
    combined_mask = horizontal_mask & group_mask
    variance_sets = {k:v for k, v in {k: np.unique(data[combined_mask][k]) for k in data.dtype.names}.items() if len(v) > 1 and k not in output_axes}
    print(f"Total rows for horizontal step: {horizontal_mask.sum()}")
    for k, v in variance_sets.items():
        try:
            print(f"{k}: {len(v)} values min: {v.min()} max: {v.max()} mean: {v.mean()} std: {v.std()}")
        except:
            print(f"{k}: {len(v)} values: {v}")


    for group_axis_value in tqdm(np.unique(data[group_axis])) if group_axis is not None else [None]:
        group_mask = data[group_axis] == group_axis_value if group_axis is not None else np.ones(len(data), dtype=bool)
        horizontal_values = np.unique(data[group_mask][horizontal_axis])
        aggregated_data, aggregated_count = np.array([[f(data[group_mask][data[group_mask][horizontal_axis] == v][vertical_axis]) for v in horizontal_values] for f in [aggregation_function, np.size]])
        # min_value = max(aggregated_data.min(), 0)
        # aggregated_data = (aggregated_data - min_value) / (aggregated_data.max() - min_value)
        yield horizontal_values, aggregated_data, group_axis_value

def plot(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function, scatter=False):
    plt.figure()
    for horizontal_values, aggregated_data, group_axis_value in process(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function):
        (plt.scatter if scatter else plt.plot)(horizontal_values, aggregated_data, label=group_axis_value)
    plt.xlabel(label_map[horizontal_axis] if horizontal_axis in label_map else horizontal_axis)
    plt.ylabel(label_map[vertical_axis] if vertical_axis in label_map else vertical_axis)
    plt.legend() if group_axis else None
    plt.show()

def plot_teacher_curve(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function, scatter=False):
    plt.figure()
    for horizontal_values, aggregated_data, group_axis_value in process(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function):
        line = plt.plot(horizontal_values, aggregated_data, label=group_axis_value)
        plt.scatter(horizontal_values, aggregated_data, color=line[0].get_color())
    plt.xlabel(label_map[horizontal_axis] if horizontal_axis in label_map else horizontal_axis)
    plt.ylabel(label_map[vertical_axis] if vertical_axis in label_map else vertical_axis)
    plt.legend() if group_axis else None
    plt.show()

paper_drones = [
    b'arpl',
    b'crazyflie',
    # b'flightmare',
    # b'fs',
    # b'mrs',
    b'soft',
    b'x500'
]

filters = {
    "dmodel": ("==", 16),
    "epoch": ("<=", 100),
    "model": ("in", paper_drones),
    "num_teachers": ("==", 1000)
}
# vertical_axis = "return_mean"
vertical_axis = "episode_length_mean"
horizontal_axis = "epoch"
# group_axis = "dmodel"
group_axis = "model"
aggregation_function = np.mean

plot(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function)



# filters = {
#     # "epoch": ("<=", 200),
#     "model": ("in", [b'\"crazyflie\"', b'\"x500\"', b'\"soft\"', b'\"arpl\"'])
# }
# # vertical_axis = "return_mean"
# vertical_axis = "episode_length_mean"
# horizontal_axis = "compute"
# group_axis = "dmodel"
# # group_axis = "model"
# # group_axis = None
# aggregation_function = np.mean

# plot(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function, scatter=True)


# # Num Teachers Learning Curves
# filters = {
#     # "epoch": ("<=", 200),
#     # "model": ("==", b'\"crazyflie\"')
#     "model": ("in", [b'\"crazyflie\"', b'\"x500\"', b'\"soft\"', b'\"arpl\"'])
# }
# # vertical_axis = "return_mean"
# vertical_axis = "episode_length_mean"
# horizontal_axis = "epoch"
# group_axis = "num_teachers"
# # group_axis = "model"
# # group_axis = None
# aggregation_function = np.mean


# plot(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function)


# Num Teachers Pareto Curve
filters = {
    # "epoch": ("<=", 200),
    # "model": ("==", b'\"crazyflie\"')
    "dmodel": ("==", 16),
    "model": ("in", paper_drones),
    "teacher_selection": ("==", b'random')
}
vertical_axis = "return_mean"
# vertical_axis = "episode_length_mean"
horizontal_axis = "num_teachers"
# group_axis = "num_teachers"
# group_axis = "model"
group_axis = None
aggregation_function = np.mean



print("Models:")
for model in np.unique(data["model"]):
    print(model)

plot_teacher_curve(data, filters, horizontal_axis, vertical_axis, group_axis, aggregation_function)