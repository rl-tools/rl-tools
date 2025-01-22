import os, gzip, json, numpy as np
import matplotlib.pyplot as plt
path = "experiments/2025-01-22_15-27-11/a33a79c_zoo_environment_algorithm/flag_sac/0001/steps/000000000000000/nn_analytics.json.gz"
with gzip.open(path, 'rb') as f:
    data_string = f.read()
    data = json.loads(data_string)


layer_output = np.array(data["actor"]["layers"][1]["output"])
print(f"data keys: {layer_output}")