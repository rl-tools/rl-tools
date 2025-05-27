from foundation_model import QuadrotorPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R
import l2f
import os
from  copy import copy
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

N_STEPS = 500
ANIMATION_INTERVAL = 20
N_TRAJECTORIES = 100
N_DRONES = 100
name = {
    "t2w": "Thrust-to-Weight Ratio",
}


def shrink_state(state):
    return state.position


def generate_data():
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


    experiment = "2025-04-16_20-10-58"

    with open(os.path.join(os.path.dirname(__file__), "models.json"), "r") as f:
        models = json.load(f)
    
    parameters = []
    hidden_trajectories_drones = []
    trajectories_drones = []
    for drone_i in tqdm(range(N_DRONES)):
        with open(f"src/foundation_policy/dynamics_parameters_{experiment}/{drone_i}.json", "r") as parameters_file:
            parameters_json_input = json.load(parameters_file)
        hidden_trajectories = []
        trajectories = []
        trajectories_actions = []
        l2f.initial_parameters(device, env, params)
        l2f.parameters_from_json(device, env, json.dumps(parameters_json_input), params)
        params_json = json.loads(l2f.parameters_to_json(device, env, params))
        parameters.append(params_json)
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
                # prediction = {k:np.dot(hidden_state, np.array(model["mean"]["coef"])) + model["mean"]["intercept"] for k, model in models.items()}
                # predictions.append(prediction)
                # print("step: ", step_i, " position", state.position, " orientation", state.orientation, " linear_velocity", state.linear_velocity, " angular_velocity", state.angular_velocity, " rpm", state.rpm)
                l2f.step(device, env, params, state, action, next_state, rng)
                state.assign(next_state)
                states.append(shrink_state(copy(state)))
                actions.append(copy(action))
            hidden_trajectories.append(hidden_states)
            trajectories.append(states[:-1])
            trajectories_actions.append(actions)
        # prediction_trajectories = {k:np.array([[step[k] for step in traj] for traj in prediction_trajectories]) for k in models.keys()}
        os.makedirs(f"src/foundation_policy/analysis/trajectories/{drone_i}", exist_ok=True)
        np.savez(f"src/foundation_policy/analysis/trajectories/{drone_i}/trajectories.npz", np.array(trajectories).astype(np.float32))
        np.savez(f"src/foundation_policy/analysis/trajectories/{drone_i}/hidden_trajectories.npz", np.array(hidden_trajectories).astype(np.float32))
        with open(f"src/foundation_policy/analysis/trajectories/{drone_i}/parameters.json", "w") as f:
            json.dump(params_json, f, indent=4)

        trajectories_drones.append(trajectories)
        hidden_trajectories_drones.append(hidden_trajectories)

    trajectories = np.array(trajectories_drones)
    hidden_trajectories = np.array(hidden_trajectories_drones)
    return parameters, trajectories, hidden_trajectories


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
def get_ground_truths(parameters):
    for params_json in parameters:
        max_action = params_json["dynamics"]["action_limit"]["max"]
        thrusts = [np.dot(tc, [max_action ** i for i in range(len(tc))]) for tc in params_json["dynamics"]["rotor_thrust_coefficients"]]
        thrust = np.sum(thrusts)
        thrust2weight = thrust / (params_json["dynamics"]["mass"] * 9.81)
        ground_truth = {"t2w": thrust2weight}
        yield ground_truth

def model(dependents, drone_trajectories, hidden_trajectories, final_step_only=False):
    ground_truths = list(get_ground_truths(parameters))
    Xs = {}
    Xs_final = {}
    ys = {}
    ys_final = {}
    prediction_trajectories = {}
    models = {}
    for dependent in dependents:
        def prepare_data(final=False, mask=False):
            Y_trajs = np.array([[[gt[dependent] for _ in range(N_STEPS)] for _ in range(len(drone_trajectories))] for (gt, drone_trajectories) in zip(ground_truths, trajectories)])
            X_trajs = hidden_trajectories
            positions = np.array([[[s.position for s in trajectory] for trajectory in trajectories] for trajectories in drone_trajectories])
            if final:
                X_trajs = X_trajs[:, :, -1:, :]
                Y_trajs = Y_trajs[:, :, -1:]
                positions = positions[:, :, -1:, :]
            final_positions = positions[:, :, -1, :]
            final_position_offset = np.linalg.norm(final_positions, axis=-1)
            final_position_mask = final_position_offset < 0.2
            print(f"Mask rate: {np.mean(final_position_mask):.2f}")
            mask_broadcast = np.repeat(final_position_mask[:, :, np.newaxis], positions.shape[2], axis=-1)
            assert Y_trajs.shape == X_trajs.shape[:3]
            mask = mask_broadcast.reshape(-1)
            X = X_trajs.reshape(-1, X_trajs.shape[-1])[mask, :]
            y = Y_trajs.reshape(-1)[mask]
            return X, y
        def train(X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            print(f"    Train:")
            print(f"        Mean squared error: ", mean_squared_error(y_test, y_pred_test))
            print(f"        R2 score: ", r2_score(y_test, y_pred_test))
            return model
        def test(name, X, y, model):
            y_pred = X @ model.coef_ + model.intercept_
            print(f"    {name}")
            print(f"        Mean squared error: ", mean_squared_error(y, y_pred))
            print(f"        R2 score final: ", r2_score(y, y_pred))

        X, y = prepare_data(final=False, mask=True)
        Xs[dependent] = X
        ys[dependent] = y

        model = train(X, y)
        model_final = train(*prepare_data(final=True, mask=True))

        X_final, y_final = prepare_data(final=True, mask=True)
        test("Full model, final step test", X_final, y_final, model)
        test("Final model, final step test", X_final, y_final, model_final)


        # y_pred = X @ model.coef_ + model.intercept_
        # abs_error = np.abs(y - y_pred)
        # X_train_abs_error, X_test_abs_error, y_train_abs_error, y_test_abs_error = train_test_split(X, abs_error, test_size=0.2, random_state=42)
        # model_std = LinearRegression()
        # model_std.fit(X_train_abs_error, y_train_abs_error)
        # abs_error_pred = model_std.predict(X_test_abs_error)
        # print(f"{dependent}: Mean squared error (std): ", mean_squared_error(y_test_abs_error, abs_error_pred))
        # print("    R2 score (std): ", r2_score(y_test_abs_error, abs_error_pred))

        # prediction_trajectories[dependent] = X_trajs @ model.coef_ + model.intercept_

        models[dependent] = {
            "mean_full": {"coef": model.coef_.tolist(), "intercept": model.intercept_},
            "mean_final": {"coef": model_final.coef_.tolist(), "intercept": model_final.intercept_},
            # "std": {"coef": model_std.coef_.tolist(), "intercept": model_std.intercept_}
        }
    return models, prediction_trajectories, ground_truths, Xs, ys

def plot_model(dependent, model, y, X):
    true = y
    pred = X @ model["coef"] + model["intercept"]
    COLOR_PRIMARY = "#7DB9B6"
    COLOR_GREY = "#3B75AF" #(59,117,175)
    COLOR_GREY = "#347271" #(59,117,175)
    COLOR_BLACK = "#000000"
    SCATTER_ALPHA = 0.10
    SCATTER_SIZE  = 0.5
    LINEWIDTH     = 2
    fig, ax = plt.subplots(figsize=(5, 3), dpi=600)
    for i in range(1):
        ax.scatter(
            true,
            pred,
            alpha=SCATTER_ALPHA if i == 0 else 1,
            s=SCATTER_SIZE if i == 0 else 0.05,
            color=COLOR_GREY if i == 0 else COLOR_BLACK,
            edgecolors="none",
            zorder=6,
            rasterized=True,
            # label="Predictions"
        )
    ax.scatter(
        [],
        [],
        color=COLOR_GREY,
        edgecolors="none",
        zorder=6,
        label="Predictions"
    )
    n_bins = 20
    bins = np.linspace(min(true), max(true), n_bins+1)
    idx = np.digitize(true,bins)-1
    centers = 0.5*(bins[:-1]+bins[1:])
    mean_bin = np.array([pred[idx==j].mean() if np.any(idx==j) else np.nan for j in range(n_bins)])
    std_bin = np.array([pred[idx==j].std(ddof=0) if np.any(idx==j) else np.nan for j in range(n_bins)])
    mask = ~np.isnan(mean_bin)
    # ax.plot(centers[mask],mean_bin[mask]-std_bin[mask],color=COLOR_PRIMARY,alpha=1.0,linewidth=1,zorder=5)
    # ax.plot(centers[mask],mean_bin[mask]+std_bin[mask],color=COLOR_PRIMARY,alpha=1.0,linewidth=1,zorder=5)
    ax.fill_between(
        centers[mask],
        mean_bin[mask] - std_bin[mask],
        mean_bin[mask] + std_bin[mask],
        color=COLOR_PRIMARY,
        alpha=0.75,
        linewidth=0,
        zorder=5,
        label="Mean prediction ± $\sigma$",
    )
    # ax.plot(
    #     centers[mask],
    #     mean_bin[mask],
    #     color=COLOR_PRIMARY,
    #     linewidth=1,
    #     label="Mean prediction",
    #     zorder=4,
    # )
    ax.plot(
        [min(true), max(true)],
        [min(true), max(true)],
        color="red",
        linestyle="--",
        linewidth=1,
        label="Identity",
        zorder=6,
    )
    ax.set_xlim(min(true), max(true))
    ax.set_ylim(min(true), max(true))
    # ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Linear Probe: {name[dependent]}")
    ax.margins(x=0)
    fig.tight_layout()
    fig.legend(
        loc="lower right",
        bbox_to_anchor=(1, -0.02),
        bbox_transform=ax.transAxes,
        # fontsize=10,
        frameon=False,
    )
    plt.savefig(f"src/foundation_policy/analysis/figures/analyze_{dependent}.pdf", bbox_inches="tight")


def render(params_json, states):
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
    return frames


def plot():
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
    

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    parameters, trajectories, hidden_trajectories = generate_data()
    dependents = ["t2w"]
    models, prediction_trajectories, gts, X, y = model(dependents, trajectories, hidden_trajectories)
    plot_model("t2w", models["t2w"]["mean_full"], y["t2w"], X["t2w"])

    with open("src/foundation_policy/analysis/models.json", "w") as f:
        json.dump(models, f, indent=4)


    
    # for params_json, trajectories, hidden_trajectories in zip(parameters, trajectories, hidden_trajectories):
    #     for trajectory in trajectories:
    #         # animations = render(params_json, trajectory)
    #         prediction_trajectories = {k:np.array([[step[k] for step in traj] for traj in trajectories]) for k in dependents}
    #         ground_truth = gts[0]
    #         plot()


