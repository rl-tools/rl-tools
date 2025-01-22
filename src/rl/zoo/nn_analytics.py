import os, gzip, json, numpy as np
import matplotlib.pyplot as plt
path = "experiments/2025-01-22_16-30-16/786af4a_zoo_environment_algorithm/flag_sac/0010/steps/000000000000101/nn_analytics.json.gz"
with gzip.open(path, 'rb') as f:
    data_string = f.read()
    data = json.loads(data_string)


layer_output = np.array(data["actor"]["layers"][1]["output"])
plt.imshow(layer_output[:, 0], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.ylabel("Sequence")
plt.xlabel("Feature")
plt.show()