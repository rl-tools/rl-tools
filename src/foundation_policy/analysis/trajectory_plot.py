from generate_data import load, generate_data
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import os
import json
from probe import get_ground_truths
from copy import deepcopy

ANIMATION_INTERVAL = 10
MODEL = "t2w_full"
PARAMETERS_UI_CONFIG = {
    "model": "11f470c8206d4ca43bf3f7e1ba1d7acc456d3c34",
    "name": "x500",
    "camera_distance": 2
}
N_DRONES = 10
N_TRAJECTORIES = 1

default_options = {
    "camera_position": [0.0, 0.5, 1]
}

def render(params_json, states, options=default_options):
    import requests, zipfile
    import io, imageio.v3 as iio

    def state_to_dict(s, dt):
        return {
            "state": {
                "position":         [float(s[0]), 0, float(s[2])],
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
    return frames, resp.content

if __name__ == "__main__":
    # all_parameters, all_trajectories, all_hidden_trajectories = load()
    all_parameters, all_trajectories, all_hidden_trajectories = zip(*generate_data(N_DRONES, N_TRAJECTORIES, 500, save=False, position_clip=1.0))
    models_path = os.path.join(os.path.dirname(__file__), "models.json")
    with open(models_path, "r") as f:
        models = json.load(f)

    for drone_i, (parameters, ground_truths, trajectories, hidden_trajectories) in enumerate(zip(all_parameters, get_ground_truths(all_parameters), all_trajectories, all_hidden_trajectories)):
        parameters = deepcopy(parameters)
        parameters["ui"] = PARAMETERS_UI_CONFIG
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
            animations, frames_zip = render(parameters, states)
            with open(f"src/foundation_policy/analysis/figures/trajectory_{drone_i}_{trajectory_i}.zip", "wb") as f:
                f.write(frames_zip)
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

            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2.0, 1, 1, 1.5]})
            current_ax = 0
            ax_anim = axs[current_ax]
            current_ax += 1
            alpha_scale = 0.8
            zoom = 4
            ANIMATION_START = 0
            ANIMATION_END = len(animations)//3
            animation_range = animations[ANIMATION_START:ANIMATION_END]
            animation_space = len(states)/(len(animation_range)-1)
            for i, f in enumerate(animation_range):
                w, h = f.shape[1], f.shape[0]
                o = animation_space * i
                offset = animation_space / 3
                offset = 0
                ax_anim.imshow(f, alpha=alpha_scale, interpolation='none', extent=[o + offset - animation_space/2 * zoom, o + offset + animation_space/2 * zoom, 0, animation_space * zoom], clip_on=False)
            ax_anim.set_xlim(0, len(states))
            ax_anim.set_ylim(animation_space * zoom * 0.4, animation_space * zoom * 1.0)
            ax_anim.set_axis_off()
            ax_anim.set_aspect('equal')
            ax = axs[current_ax]
            current_ax += 1
            ax.plot(position_error, label="Position")
            ax.plot(orientation_error, label="Orientation")
            ax.plot(linear_velocity_error, label="Linear Velocity")
            ax.plot(angular_velocity_error, label="Angular Velocity")
            ax.set_ylabel("Error [relative]")
            ax.legend(loc="upper right")
            ax = axs[current_ax]
            current_ax += 1
            ax.plot(prediction_trajectories[MODEL][trajectory_i], label="Predicted")
            ax.plot(ground_truth_trajectories[MODEL][trajectory_i], label="Ground Truth")
            ax.set_ylim(0, 5)
            ax.set_ylabel("Thrust to Weight Ratio")
            ax.legend(loc="lower right")
            ax = axs[current_ax]
            # ax.set_title(f"Hidden States for Drone {drone_i}, Trajectory {trajectory_i}")
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
            plt.tight_layout(h_pad=0.2)
            plt.subplots_adjust(hspace=0.1)
            offset = 0 #animation_space / 3.0
            half_width = 0.5 * animation_space * zoom
            x0_top = 0
            x1_top = len(states)
            
            total_frames = len(animations)
            frames_to_states_ratio = len(states) / total_frames if total_frames > 0 else 0.0
            x0_bottom = float(np.clip(ANIMATION_START * frames_to_states_ratio, 0, len(states)))
            x1_bottom = float(np.clip(ANIMATION_END * frames_to_states_ratio, 0, len(states)))

            ax_first_ts = axs[1]
            
            anim_pos = ax_anim.get_position()
            ts_pos = ax_first_ts.get_position()
            
            def data_to_fig(x, y, ax):
                display_coords = ax.transData.transform([(x, y)])[0]
                return fig.transFigure.inverted().transform(display_coords)
            
            control_offset = 1.0 * (ts_pos.y1 - anim_pos.y0)
            
            for i, (x_top, x_bottom) in enumerate([(x0_top, x0_bottom), (x1_top, x1_bottom)]):
                start_fig_x, start_fig_y = data_to_fig(x_top, ax_anim.get_ylim()[0]*3/4, ax_anim)
                if i == 0:
                    start_fig_x -= 0.03
                    start_fig_y += 0.1
                else:
                    start_fig_x += 0.03
                    # start_fig_y += 0.1
                end_fig = data_to_fig(x_bottom, ax_first_ts.get_ylim()[1], ax_first_ts)
                
                ctrl1_fig = (start_fig_x, start_fig_y + control_offset)
                ctrl2_fig = (end_fig[0], end_fig[1] - control_offset)
                
                verts = [(start_fig_x, start_fig_y), ctrl1_fig, ctrl2_fig, (end_fig[0], end_fig[1])]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path = Path(verts, codes)
                
                patch = patches.PathPatch(path, facecolor='none', edgecolor='0.3', 
                                        linewidth=1.2, alpha=0.8, transform=fig.transFigure)
                fig.patches.append(patch)
            # fig.colorbar(im, ax=ax, label='activation')
            # fig.colorbar(im, ax=axs, label="activation", location="right", pad=0.02, fraction=0.04)
            plt.savefig(f"src/foundation_policy/analysis/figures/trajectory_{drone_i}_{trajectory_i}.png", dpi=600)
            # plt.show() error
        