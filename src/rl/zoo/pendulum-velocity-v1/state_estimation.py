import h5py
import numpy as np
import matplotlib.pyplot as plt

def load(filename):
    with h5py.File(filename, 'r') as f:
        print(list(f["0"].keys()))
        return {k:f["0"][k][:] for k in ['actions', 'full', 'next_observations', 'observations', 'position', 'rewards', 'terminated', 'truncated', 'episode_start']}

old = load("replay_buffer_old.h5")
new = load("replay_buffer.h5")

assert (old["observations"] == new["observations"]).all()
assert (old["episode_start"] == new["episode_start"]).all()

print("end")