import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

ANIMATION_INTERVAL = 20
name = {
    "t2w_full_full": "Thrust-to-Weight Ratio (Model: Full, Data: Full)",
    "t2w_full_final": "Thrust-to-Weight Ratio (Model: Full, Data: Final)",
    "t2w_final_full": "Thrust-to-Weight Ratio (Model: Final, Data: Full)",
    "t2w_final_final": "Thrust-to-Weight Ratio (Model: Final, Data: Final)",
}
def get_ground_truths(parameters):
    for params_json in parameters:
        max_action = params_json["dynamics"]["action_limit"]["max"]
        thrusts = [np.dot(tc, [max_action ** i for i in range(len(tc))]) for tc in params_json["dynamics"]["rotor_thrust_coefficients"]]
        thrust = np.sum(thrusts)
        thrust2weight = thrust / (params_json["dynamics"]["mass"] * 9.81)
        ground_truth = {"t2w": thrust2weight}
        yield ground_truth

def model(dependents, parameters, trajectories, hidden_trajectories, final_step_only=False):
    N_STEPS = hidden_trajectories.shape[2]
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
            positions = trajectories[:, :, :, :3]
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

        model = train(X, y)
        model_final = train(*prepare_data(final=True, mask=True))

        X_final, y_final = prepare_data(final=True, mask=True)
        test("Full model, final step test", X_final, y_final, model)
        test("Final model, final step test", X_final, y_final, model_final)

        Xs[dependent] = {"full": X, "final": X_final}
        ys[dependent] = {"full": y, "final": y_final}


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
            "mean_full": {"coef": model.coef_.tolist(), "intercept": float(model.intercept_)},
            "mean_final": {"coef": model_final.coef_.tolist(), "intercept": float(model_final.intercept_)},
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



if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    trajectories_path = os.path.join(os.path.dirname(__file__), "trajectories")
    dependents = ["t2w"]
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
    models, prediction_trajectories, gts, X, y = model(dependents, parameters, trajectories, hidden_trajectories)
    indices = np.arange(X["t2w"]["full"].shape[0])
    np.random.shuffle(indices)
    subsample_indices = indices[:len(indices)//500]
    X_full = X["t2w"]["full"][subsample_indices]
    y_full = y["t2w"]["full"][subsample_indices]
    plot_model("t2w_full_full", models["t2w"]["mean_full"], y_full, X_full)
    plot_model("t2w_full_final", models["t2w"]["mean_full"], y["t2w"]["final"], X["t2w"]["final"])
    plot_model("t2w_final_final", models["t2w"]["mean_final"], y["t2w"]["final"], X["t2w"]["final"])
    plot_model("t2w_final_full", models["t2w"]["mean_final"], y_full, X_full)

    with open("src/foundation_policy/analysis/models.json", "w") as f:
        json.dump(models, f, indent=4)


    
    # for params_json, trajectories, hidden_trajectories in zip(parameters, trajectories, hidden_trajectories):
    #     for trajectory in trajectories:
    #         # animations = render(params_json, trajectory)
    #         prediction_trajectories = {k:np.array([[step[k] for step in traj] for traj in trajectories]) for k in dependents}
    #         ground_truth = gts[0]
    #         plot()



# def plot():
#     for trajectory_i, (states, hidden_states) in enumerate(zip(trajectories, hidden_trajectories)):
#         animations = animations[trajectory_i]
#         states = states[1:]
#         position_error = np.array([np.linalg.norm(s.position) for s in states])
#         position_error = position_error / np.max(position_error)
#         orientation_error = np.array([2 * np.arccos(s.orientation[0]) for s in states])
#         orientation_error = orientation_error / np.max(orientation_error)
#         linear_velocity_error = np.array([np.linalg.norm(s.linear_velocity) for s in states])
#         linear_velocity_error = linear_velocity_error / np.max(linear_velocity_error)
#         angular_velocity_error = np.array([np.linalg.norm(s.angular_velocity) for s in states])
#         angular_velocity_error = angular_velocity_error / np.max(angular_velocity_error)

#         x = np.arange(len(position_error))

#         fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
#         current_ax = 0
#         ax_anim = axs[current_ax]
#         current_ax += 1
#         alpha_scale = 0.8
#         zoom = 10 * 2
#         ax_width = ax_anim.get_position().width
#         ax_height = ax_anim.get_position().height
#         height = len(states) * ax_height / ax_width
#         for i, f in enumerate(animations):
#             # f[:, :, 0] = 255
#             # f[:, :, -1] = 255
#             w, h = f.shape[1], f.shape[0]
#             animation_space = len(states)/len(animations)
#             o = animation_space * i
#             offset = animation_space / 3
#             ax_anim.imshow(f, alpha=alpha_scale, interpolation='none', extent=[o + offset - animation_space/2 * zoom, o + offset + animation_space/2 * zoom, height/2 - animation_space/2 * zoom, height/2 + animation_space/2 * zoom], aspect='auto', clip_on=False)
#         ax_anim.set_axis_off()
#         ax_anim.set_xlim(0, len(states))
#         ax_anim.set_ylim(0, height)
#         ax_anim.set_aspect('equal')
#         ax = axs[current_ax]
#         current_ax += 1
#         ax.plot(position_error, label="position error")
#         ax.plot(orientation_error, label="orientation error")
#         ax.plot(linear_velocity_error, label="linear velocity error")
#         ax.plot(angular_velocity_error, label="angular velocity error")
#         ax.set_ylabel("Error (Relative to Maximum)")
#         ax.legend()
#         ax = axs[current_ax]
#         current_ax += 1
#         ax.plot(prediction_trajectories["t2w"][trajectory_i], label="Predicted")
#         ax.plot(np.arange(len(prediction_trajectories["t2w"][trajectory_i])), [ground_truth["t2w"]] * len(prediction_trajectories["t2w"][trajectory_i]), label="Ground Truth")
#         ax.set_ylabel("Thrust to Weight Ratio")
#         ax.legend()
#         ax = axs[current_ax]
#         ax.set_title(f"Hidden States for Drone {drone_i}, Trajectory {trajectory_i}")
#         current_ax += 1
#         hidden_states_standardized = (hidden_states - np.mean(hidden_states, axis=0)) / np.std(hidden_states, axis=0)
#         im = ax.imshow(hidden_states_standardized.T,
#             aspect='auto',
#             origin='lower',
#             cmap='inferno',
#             # cmap='magma',
#             # cmap='hot',
#             interpolation='none',
#             extent=[x[0], x[-1], 0, 15]
#         )
#         ax.set_ylabel("Hidden Dimension")
#         ax.set_xlabel("Time Step")
#         # fig.colorbar(im, ax=ax, label='activation')
#         # fig.colorbar(im, ax=axs, label="activation", location="right", pad=0.02, fraction=0.04)
#         plt.savefig(f"src/foundation_policy/analysis/figures/trajectory_{drone_i}_{trajectory_i}.png", dpi=600)
#         # plt.show()
    