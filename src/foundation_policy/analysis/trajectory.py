from foundation_model import QuadrotorPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R
import l2f
import os
from  copy import copy
import matplotlib.pyplot as plt
import json

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

N_STEPS = 500
N_TRAJECTORIES = 1
N_DRONES = 100
ANIMATION_INTERVAL = 20

experiment = "2025-04-16_20-10-58"

with open(os.path.join(os.path.dirname(__file__), "models.json"), "r") as f:
    models = json.load(f)

for drone_i in range(N_DRONES):
    with open(f"src/foundation_policy/dynamics_parameters_{experiment}/{drone_i}.json", "r") as parameters_file:
        parameters = json.load(parameters_file)
    hidden_trajectories = []
    trajectories = []
    prediction_trajectories = []
    animations = []
    l2f.initial_parameters(device, env, params)
    l2f.parameters_from_json(device, env, json.dumps(parameters), params)
    params_json = json.loads(l2f.parameters_to_json(device, env, params))
    max_action = params_json["dynamics"]["action_limit"]["max"]
    thrusts = [np.dot(tc, [max_action ** i for i in range(len(tc))]) for tc in params_json["dynamics"]["rotor_thrust_coefficients"]]
    thrust = np.sum(thrusts)
    thrust2weight = thrust / (params_json["dynamics"]["mass"] * 9.81)
    ground_truth = {"t2w": thrust2weight}
    for trajectory_i in range(N_TRAJECTORIES):
        hidden_states = []
        states = []
        predictions = []
        policy.reset()
        l2f.sample_initial_state(device, env, params, state, rng)
        states.append(copy(state))
        for step_i in range(N_STEPS):
            l2f.observe(device, env, params, state, observation, rng)
            obs = np.array(observation.observation)[:22] # concatenating position, orientation (rotation matrix), linear velocity, angular velocity, and previous action (note in newer versions of l2f the most recent action follows right after the angular velocity)
            action = policy.evaluate_step(np.array([obs]))[0]
            hidden_state = policy.layers[1].state[0]
            hidden_states.append(hidden_state)
            prediction = {k:np.dot(hidden_state, np.array(model["mean"]["coef"])) + model["mean"]["intercept"] for k, model in models.items()}
            predictions.append(prediction)
            print("step: ", step_i, " position", state.position, " orientation", state.orientation, " linear_velocity", state.linear_velocity, " angular_velocity", state.angular_velocity, " rpm", state.rpm)
            l2f.step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
            states.append(copy(state))
        hidden_trajectories.append(hidden_states)
        trajectories.append(states)
        prediction_trajectories.append(predictions)
        import requests, zipfile
        import io, imageio.v3 as iio

        def state_to_dict(s, dt):
            return {
                "state": {
                    "position":         [0, 0, 0],
                    "orientation":      s.orientation,   # w-x-y-z
                    "linear_velocity":  s.linear_velocity,
                    "angular_velocity": s.angular_velocity,
                    "rpm":              s.rpm
                },
                "action": [0, 0, 0, 0],
                "dt": dt
            }

        trajectory_payload = [state_to_dict(s, params_json["integration"]["dt"]) for s in states[::ANIMATION_INTERVAL]]       # states just computed

        payload = {
            "parameters": params_json,
            "trajectory": trajectory_payload
        }

        url = "http://localhost:13339/render_trajectory"

        with open(os.path.join(os.path.dirname(__file__), "ui.js"), "rb") as ui_file:
            resp = requests.post(
                url,
                data={"width":  "2000", "height": "2000"},
                files={
                    "data": ("data.json", json.dumps(payload), "application/json"),
                    "ui": ("ui.js", ui_file, "application/javascript")
                }
            )
        resp.raise_for_status()

        frames = []
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in sorted(zf.namelist()):
                frames.append(iio.imread(zf.read(name), extension=".png"))
        animations.append(frames)

    hidden_trajectories = np.array(hidden_trajectories)
    prediction_trajectories = {k:np.array([[step[k] for step in traj] for traj in prediction_trajectories]) for k in models.keys()}


    # first_hidden_states = hidden_trajectories[:, 0]
    # first_hidden_states_standardized = (first_hidden_states - np.mean(first_hidden_states, axis=0)) / np.std(first_hidden_states, axis=0)
    # U, S, Vt = np.linalg.svd(first_hidden_states_standardized, full_matrices=False)
    # var_ratio = S**2 / (S**2).sum()
    # cum = np.cumsum(var_ratio)
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(1, len(cum)+1), cum)
    # plt.xlabel("First N Principal Components")
    # plt.ylabel("Cumulative Explained Variance")
    # plt.title("Cumulative Explained Variance")
    # plt.savefig(f"src/foundation_policy/analysis/cumulative_explained_variance.png", dpi=600)
    # plt.show()


    for trajectory_i, (states, hidden_states) in enumerate(zip(trajectories, hidden_trajectories)):
        animations = animations[trajectory_i]
        states = states[1:]
        position_error = np.array([np.linalg.norm(s.position) for s in states])
        position_error = position_error / np.max(position_error)
        orientation_error = np.array([2 * np.arccos(s.orientation[0]) for s in states])
        orientation_error = orientation_error / np.max(orientation_error)
        linear_velocity_error = np.array([np.linalg.norm(s.linear_velocity) for s in states])
        linear_velocity_error = linear_velocity_error / np.max(linear_velocity_error)
        angular_velocity_error = np.array([np.linalg.norm(s.angular_velocity) for s in states])
        angular_velocity_error = angular_velocity_error / np.max(angular_velocity_error)

        x = np.arange(len(position_error))

        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
        current_ax = 0
        ax_anim = axs[current_ax]
        current_ax += 1
        alpha_scale = 0.8
        zoom = 10 * 2
        ax_width = ax_anim.get_position().width
        ax_height = ax_anim.get_position().height
        height = len(states) * ax_height / ax_width
        for i, f in enumerate(animations):
            # f[:, :, 0] = 255
            # f[:, :, -1] = 255
            w, h = f.shape[1], f.shape[0]
            animation_space = len(states)/len(animations)
            o = animation_space * i
            offset = animation_space / 3
            ax_anim.imshow(f, alpha=alpha_scale, interpolation='none', extent=[o + offset - animation_space/2 * zoom, o + offset + animation_space/2 * zoom, height/2 - animation_space/2 * zoom, height/2 + animation_space/2 * zoom], aspect='auto', clip_on=False)
        ax_anim.set_axis_off()
        ax_anim.set_xlim(0, len(states))
        ax_anim.set_ylim(0, height)
        ax_anim.set_aspect('equal')
        ax = axs[current_ax]
        current_ax += 1
        ax.plot(position_error, label="position error")
        ax.plot(orientation_error, label="orientation error")
        ax.plot(linear_velocity_error, label="linear velocity error")
        ax.plot(angular_velocity_error, label="angular velocity error")
        ax.set_ylabel("Error (Relative to Maximum)")
        ax.legend()
        ax = axs[current_ax]
        current_ax += 1
        ax.plot(prediction_trajectories["t2w"][trajectory_i], label="Predicted")
        ax.plot(np.arange(len(prediction_trajectories["t2w"][trajectory_i])), [ground_truth["t2w"]] * len(prediction_trajectories["t2w"][trajectory_i]), label="Ground Truth")
        ax.set_ylabel("Thrust to Weight Ratio")
        ax.legend()
        ax = axs[current_ax]
        ax.set_title(f"Hidden States for Drone {drone_i}, Trajectory {trajectory_i}")
        current_ax += 1
        hidden_states_standardized = (hidden_states - np.mean(hidden_states, axis=0)) / np.std(hidden_states, axis=0)
        im = ax.imshow(hidden_states_standardized.T,
            aspect='auto',
            origin='lower',
            cmap='inferno',
            # cmap='magma',
            # cmap='hot',
            interpolation='none',
            extent=[x[0], x[-1], 0, 15]
        )
        ax.set_ylabel("Hidden Dimension")
        ax.set_xlabel("Time Step")
        # fig.colorbar(im, ax=ax, label='activation')
        # fig.colorbar(im, ax=axs, label="activation", location="right", pad=0.02, fraction=0.04)
        plt.savefig(f"src/foundation_policy/analysis/figures/trajectory_{drone_i}_{trajectory_i}.png", dpi=600)
        # plt.show()
    
