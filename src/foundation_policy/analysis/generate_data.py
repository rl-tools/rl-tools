from foundation_model import QuadrotorPolicy
import numpy as np
import l2f
import os
from  copy import copy
import json
from tqdm import tqdm

N_STEPS = 500
N_TRAJECTORIES = 100
N_DRONES = 1000
trajectories_path = os.path.join(os.path.dirname(__file__), "trajectories")


def shrink_state(state):
    return np.concatenate((state.position, state.orientation, state.linear_velocity, state.angular_velocity, state.rpm), axis=None)


if __name__ == "__main__":
    policy = QuadrotorPolicy()
    policy.reset()

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

    experiment = "2025-04-16_20-10-58"

    for drone_i in tqdm(range(N_DRONES)):
        with open(f"src/foundation_policy/dynamics_parameters_{experiment}/{drone_i}.json", "r") as parameters_file:
            parameters_json_input = json.load(parameters_file)
        hidden_trajectories = []
        trajectories = []
        trajectories_actions = []
        l2f.initial_parameters(device, env, params)
        l2f.parameters_from_json(device, env, json.dumps(parameters_json_input), params)
        params_json = json.loads(l2f.parameters_to_json(device, env, params))
        for trajectory_i in range(N_TRAJECTORIES):
            hidden_states = []
            states = []
            actions = []
            policy.reset()
            l2f.sample_initial_state(device, env, params, state, rng)
            states.append(shrink_state(copy(state)))
            for step_i in range(N_STEPS):
                l2f.observe(device, env, params, state, observation, rng)
                obs = np.array(observation.observation)[:22] # concatenating position, orientation (rotation matrix), linear velocity, angular velocity, and previous action (note in newer versions of l2f the most recent action follows right after the angular velocity)
                action = policy.evaluate_step(np.array([obs]))[0]
                hidden_state = policy.layers[1].state[0]
                hidden_states.append(hidden_state)
                l2f.step(device, env, params, state, action, next_state, rng)
                state.assign(next_state)
                states.append(shrink_state(copy(state)))
                actions.append(copy(action))
            hidden_trajectories.append(hidden_states)
            trajectories.append(states[:-1])
            trajectories_actions.append(actions)
        drone_path = os.path.join(trajectories_path, f"{drone_i}")
        os.makedirs(drone_path, exist_ok=True)
        np.savez(os.path.join(drone_path, "trajectories.npz"), np.array(trajectories).astype(np.float32))
        np.savez(os.path.join(drone_path, "hidden_trajectories.npz"), np.array(hidden_trajectories).astype(np.float32))
        with open(os.path.join(drone_path, "parameters.json"), "w") as f:
            json.dump(params_json, f, indent=4)


def load():
    hidden_trajectories = []
    trajectories = []
    parameters = []
    for drone in tqdm(list(filter(lambda drone: os.path.exists(os.path.join(trajectories_path, drone, "parameters.json")), os.listdir(trajectories_path)))):
        drone_path = os.path.join(trajectories_path, drone)
        hidden_trajectories.append(np.load(os.path.join(drone_path, "hidden_trajectories.npz"))["arr_0"])
        trajectories.append(np.load(os.path.join(drone_path, "trajectories.npz"))["arr_0"])
        with open(os.path.join(drone_path, "parameters.json"), "r") as f:
            parameters.append(json.load(f))
    hidden_trajectories = np.array(hidden_trajectories)
    trajectories = np.array(trajectories)
    return parameters, trajectories, hidden_trajectories