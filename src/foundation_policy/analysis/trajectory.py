from foundation_model import QuadrotorPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R
import l2f
import os
from  copy import copy

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


hidden_states = []
def loop(N_STEPS = 500):
    l2f.sample_initial_parameters(device, env, params, rng)
    l2f.initial_state(device, env, params, state)
    policy.reset()
    for step_i in range(N_STEPS):
        l2f.observe(device, env, params, state, observation, rng)
        obs = np.concatenate([np.array(observation.observation)[:18], np.array(observation.observation)[-4:]]) # concatenating position, orientation (rotation matrix), linear velocity, angular velocity, and previous action (note in newer versions of l2f the most recent action follows right after the angular velocity)
        action = policy.evaluate_step(np.array([obs]))[0]
        hidden_states.append(policy.layers[1].state[0])
        print("step: ", step_i, " position", state.position, " orientation", state.orientation, " linear_velocity", state.linear_velocity, " angular_velocity", state.angular_velocity, " rpm", state.rpm)
        l2f.step(device, env, params, state, action, next_state, rng)
        state.assign(next_state)
        yield copy(state)

for _ in loop():
    pass

hidden_states = np.array(hidden_states)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(hidden_states[:50].T,           # shape: (T, units) or (units, T)
               aspect='auto',                # keeps the natural x/y ratio
               origin='lower',               # time starts at the left-bottom
               cmap='magma',                 # or 'inferno' / 'rocket'
               interpolation='none')         # no blurring
fig.colorbar(im, ax=ax, label='activation')
plt.show()