from generate_data import load, generate_data
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from probe import get_ground_truths

ANIMATION_INTERVAL = 10
MODEL = "t2w_full"

default_options = {
    "camera_position": [0.0, 0.5, 1]
}

def render(params_json, states, options=default_options):
    import requests, zipfile
    import io, imageio.v3 as iio

    def state_to_dict(s, dt):
        return {
            "state": {
                "position":         [0, 0, 0],
                "orientation":      s[3:3+4].tolist(),   # w-x-y-z
                "linear_velocity":  s[3+4:3+4+3].tolist(),
                "angular_velocity": s[3+4+3:3+4+3+3].tolist(),
                "rpm":              s[3+4+3+3:3+4+3+3+4].tolist(),
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
                "ui": ("ui.js", ui_file, "application/javascript"),
                "options": ("options.json", json.dumps(options), "application/json")
            }
        )
    resp.raise_for_status()

    frames = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in sorted(zf.namelist()):
            frames.append(iio.imread(zf.read(name), extension=".png"))
    return frames

if __name__ == "__main__":
    # all_parameters, all_trajectories, all_hidden_trajectories = load()
    all_parameters, all_trajectories, all_hidden_trajectories = zip(*generate_data(10, 1, 500, save=False, initial_position=[0, 3, 0], position_clip=1.0))
    models_path = os.path.join(os.path.dirname(__file__), "models.json")
    with open(models_path, "r") as f:
        models = json.load(f)

    for drone_i, (parameters, ground_truths, trajectories, hidden_trajectories) in enumerate(zip(all_parameters, get_ground_truths(all_parameters), all_trajectories, all_hidden_trajectories)):
        prediction_trajectories = {k:[] for k in models.keys()}
        ground_truth_trajectories = {k:[] for k in models.keys()}
        for trajectory in hidden_trajectories:
            predictions = {k:[] for k in models.keys()}
            ground_truths_trajectory = {k:[] for k in models.keys()}
            for hidden_state in trajectory:
                for k, model in models.items():
                    prediction = np.dot(hidden_state, np.array(model["mean"]["coef"])) + model["mean"]["intercept"]
                    predictions[k].append(prediction)
                    ground_truths_trajectory[k].append(ground_truths[k.split("_")[0]])
            for k in predictions.keys():
                prediction_trajectories[k].append(predictions[k])
                ground_truth_trajectories[k].append(ground_truths_trajectory[k])
        for k in prediction_trajectories.keys():
            prediction_trajectories[k] = np.array(prediction_trajectories[k])
            ground_truth_trajectories[k] = np.array(ground_truth_trajectories[k])
            
        for trajectory_i, (states, hidden_states) in enumerate(zip(trajectories, hidden_trajectories)):
            animations = render(parameters, states)
            states = states[1:]
            position = lambda s: s[:3]
            orientation = lambda s: s[3:3+4]  # w-x-y-z
            linear_velocity = lambda s: s[3+4:3+4+3]
            angular_velocity = lambda s: s[3+4+3:3+4+3+3]

            position_error = np.array([np.linalg.norm(position(s)) for s in states])
            position_error = position_error / np.max(position_error)
            orientation_error = np.array([2 * np.arccos(orientation(s)[0]) for s in states])
            orientation_error = orientation_error / np.max(orientation_error)
            linear_velocity_error = np.array([np.linalg.norm(linear_velocity(s)) for s in states])
            linear_velocity_error = linear_velocity_error / np.max(linear_velocity_error)
            angular_velocity_error = np.array([np.linalg.norm(angular_velocity(s)) for s in states])
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
            ax.plot(prediction_trajectories[MODEL][trajectory_i], label="Predicted")
            ax.plot(ground_truth_trajectories[MODEL][trajectory_i], label="Ground Truth")
            ax.set_ylim(0, 5)
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
        