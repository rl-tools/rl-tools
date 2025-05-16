import json
import numpy as np
from tqdm import tqdm

experiment = "2025-04-16_20-10-58"
dynamics_parameters_path = f"src/foundation_policy/dynamics_parameters_{experiment}"
dependents = ["t2w"] #, "return"]
name = {
    "t2w": "Thrust-to-Weight Ratio",
    "t2i": "Torque to Inertia",
    "return": "Return",
}
X = []
ys = {k: [] for k in dependents}

parameters_cache = {}
def get_parameters(dynamics_id):
    if dynamics_id in parameters_cache:
        return parameters_cache[dynamics_id]
    with open(dynamics_parameters_path + f"/{dynamics_id}.json", "r") as parameters_file:
        parameters = json.load(parameters_file)
        parameters_cache[dynamics_id] = parameters
        return parameters

with open(f"src/foundation_policy/hidden_states_{experiment}.json", "r") as f:
    hidden_states = json.load(f)
    for hidden_state in tqdm(hidden_states):
        x = np.array(hidden_state["hidden_state"])
        X.append(x)
        parameters = get_parameters(hidden_state["dynamics_id"])
        for dependent in dependents:
            max_action = parameters["dynamics"]["action_limit"]["max"]
            thrusts = [np.dot(tc, [max_action ** i for i in range(len(tc))]) for tc in parameters["dynamics"]["rotor_thrust_coefficients"]]
            if dependent == "t2w":
                thrust = np.sum(thrusts)
                thrust2weight = thrust / (parameters["dynamics"]["mass"] * 9.81)
                ys[dependent].append(thrust2weight)
            elif dependent == "t2i":
                l = np.linalg.norm(parameters["dynamics"]["rotor_positions"][0])
                torque = thrusts[0] * l * np.sqrt(2)
                inertia = parameters["dynamics"]["J"][0][0]
                torque2inertia = torque / inertia
                ys[dependent].append(torque2inertia)
            elif dependent == "return":
                ys[dependent].append(hidden_state["return"])
            else:
                raise ValueError(f"Unknown dependent: {dependent}")

X = np.array(X)
X_standardized = X # (X - X.mean(axis=0)) / X.std(axis=0)
U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)
var_ratio = S**2 / (S**2).sum()
cum = np.cumsum(var_ratio)

import matplotlib.pyplot as plt
plt.plot(cum)
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.savefig(f"src/foundation_policy/analysis/figures/cumulative_explained_variance_non_standardized.png", dpi=600)
plt.close()

p = S / S.sum()
erank = np.exp(-(p * np.log(p)).sum())
pr = (S.sum()**2) / (S**2).sum()
print(f"Effective rank (Shannon): {erank:.2f}")
print(f"Participation ratio     : {pr:.2f}")

ys = {k:np.array(yy) for k, yy in ys.items()}

# train least squares regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
models = {}
for dependent in dependents:
    y = ys[dependent]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{dependent}: Mean squared error: ", mean_squared_error(y_test, y_pred))
    print("    R2 score: ", r2_score(y_test, y_pred))

    percentile_cutoff = 0

    preds_train = X_train @ model.coef_ + model.intercept_
    abs_res_train = np.abs(y_train - preds_train)
    abs_res_min = np.percentile(abs_res_train, percentile_cutoff)
    abs_res_max = np.percentile(abs_res_train, 100-percentile_cutoff)
    abs_res_train = np.clip(abs_res_train, abs_res_min, abs_res_max)
    preds_test = X_test @ model.coef_ + model.intercept_
    abs_res_test = np.abs(y_test - preds_test)
    abs_res_test = np.clip(abs_res_test, abs_res_min, abs_res_max)
    
    # for pred, y_true in zip(preds, y):
    #     print(f"pred: {pred}, y_true: {y_true}")
    model_std = LinearRegression()
    model_std.fit(X_train, abs_res_train)
    abs_res_test_pred = model_std.predict(X_test)
    print(f"{dependent}: Mean squared error (std): ", mean_squared_error(abs_res_test, abs_res_test_pred))
    print("    R2 score (std): ", r2_score(abs_res_test, abs_res_test_pred))

    models[dependent] = {
        "mean": {"coef": model.coef_.tolist(), "intercept": model.intercept_},
        "std": {"coef": model_std.coef_.tolist(), "intercept": model_std.intercept_}
    }

    abs_res_train_pred = X_train @ model_std.coef_ + model_std.intercept_

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # plt.scatter(y_test, preds_test, alpha=0.02, s=1)
    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    # plt.xlim([min(y_test), max(y_test)])
    # plt.ylim([min(y_test), max(y_test)])
    # plt.xlabel("True")
    # plt.ylabel("Predicted")
    # plt.title(f"True vs Predicted {name[dependent]} (std) ratio")
    # ax.set_aspect('equal')
    # plt.savefig(f"src/foundation_policy/analysis/figures/analyze_{dependent}.png", dpi=600)
    # plt.show()
    true = y
    pred = X @ model.coef_ + model.intercept_
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
    bins = np.linspace(min(y_test), max(y_test), n_bins+1)
    idx = np.digitize(y_test,bins)-1
    centers = 0.5*(bins[:-1]+bins[1:])
    mean_bin = np.array([preds_test[idx==j].mean() if np.any(idx==j) else np.nan for j in range(n_bins)])
    std_bin = np.array([preds_test[idx==j].std(ddof=0) if np.any(idx==j) else np.nan for j in range(n_bins)])
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
    # plt.show()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(y_train, preds_train, alpha=0.01, s=1)
    # plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color="red", linestyle="--")
    # plt.xlabel("True")
    # plt.ylabel("Predicted")
    # plt.title(f"True vs Predicted {name[dependent]} ratio")
    # plt.savefig(f"src/foundation_policy/analysis/figures/analyze_{dependent}.png", dpi=600)
    # plt.show()

with open(f"src/foundation_policy/analysis/models.json", "w") as f:
    f.write(json.dumps(models))
