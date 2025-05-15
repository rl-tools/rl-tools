from foundation_model import QuadrotorPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R
import l2f
import os
from  copy import copy
import matplotlib.pyplot as plt

policy = QuadrotorPolicy()

position = [0, 0, 0]
orientation = [1, 0, 0, 0]
orientation_rotation_matrix = R.from_quat(np.array(orientation)[[1, 2, 3, 0]]).as_matrix().ravel().tolist()
linear_velocity = [0, 0, 0]
angular_velocity = [0, 0, 0]
previous_action = [0, 0, 0, 0]

observation = np.array([position + orientation_rotation_matrix + linear_velocity + angular_velocity + previous_action])
policy.reset()
action = policy.evaluate_step(observation)
print(action)



device = l2f.Device()
rng = l2f.Rng()
env = l2f.Environment()
params = l2f.Parameters()
state = l2f.State()
observation = l2f.Observation()
next_state = l2f.State()
observation = l2f.Observation()
l2f.initialize_environment(device, env)
l2f.initialize_rng(device, rng, 0)

N_STEPS = 50
N_TRAJECTORIES = 100
N_DRONES = 10


for drone_i in range(N_DRONES):
    hidden_trajectories = []
    l2f.sample_initial_parameters(device, env, params, rng)
    for trajectory_i in range(N_TRAJECTORIES):
        hidden_states = []
        params_json = l2f.parameters_to_json(device, env, params)
        l2f.sample_initial_state(device, env, params, state, rng)
        policy.reset()
        for step_i in range(N_STEPS):
            l2f.observe(device, env, params, state, observation, rng)
            obs = np.concatenate([np.array(observation.observation)[:18], np.array(observation.observation)[-4:]]) # concatenating position, orientation (rotation matrix), linear velocity, angular velocity, and previous action (note in newer versions of l2f the most recent action follows right after the angular velocity)
            action = policy.evaluate_step(np.array([obs]))[0]
            hidden_states.append(policy.layers[1].state[0])
            print("step: ", step_i, " position", state.position, " orientation", state.orientation, " linear_velocity", state.linear_velocity, " angular_velocity", state.angular_velocity, " rpm", state.rpm)
            l2f.step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
        hidden_trajectories.append(hidden_states)
    hidden_trajectories = np.array(hidden_trajectories)


    first_hidden_states = hidden_trajectories[:, 0]
    first_hidden_states_standardized = (first_hidden_states - np.mean(first_hidden_states, axis=0)) / np.std(first_hidden_states, axis=0)
    U, S, Vt = np.linalg.svd(first_hidden_states_standardized, full_matrices=False)
    var_ratio = S**2 / (S**2).sum()
    cum = np.cumsum(var_ratio)
import matplotlib.pyplot as plt
plt.plot(cum)
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.show()
# plt.savefig(f"src/foundation_policy/analysis/cumulative_explained_variance.png", dpi=600)


# for hidden_states in hidden_trajectories:
#     fig, ax = plt.subplots(figsize=(12, 4))
#     hidden_states_standardized = (hidden_states - np.mean(hidden_states, axis=0)) / np.std(hidden_states, axis=0)
#     im = ax.imshow(hidden_states_standardized.T,
#                 aspect='auto',
#                 origin='lower',
#                 cmap='inferno',                 # or 'inferno' / 'rocket'
#                 interpolation='none')
#     fig.colorbar(im, ax=ax, label='activation')
#     plt.show()