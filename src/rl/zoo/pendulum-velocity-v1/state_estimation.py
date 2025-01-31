import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File('replay_buffer.h5', 'r') as f:
    print(list(f["0"].keys()))
    actions, full, next_observations, observations, position, rewards, terminated, truncated, episode_start = [f["0"][k][:] for k in ['actions', 'full', 'next_observations', 'observations', 'position', 'rewards', 'terminated', 'truncated', 'episode_start']]


observations_normalized = observations / observations.std(axis=0)
