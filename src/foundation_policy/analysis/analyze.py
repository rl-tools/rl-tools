import json
import numpy as np
from tqdm import tqdm

experiment = "2025-04-16_20-10-58"
dynamics_parameters_path = f"src/foundation_policy/dynamics_parameters_{experiment}"
dependents = ["t2w"] #, "return"]
name = {
    "t2w": "Thrust to Weight",
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
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)
var_ratio = S**2 / (S**2).sum()
cum = np.cumsum(var_ratio)

import matplotlib.pyplot as plt
plt.plot(cum)
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.savefig(f"src/foundation_policy/analysis/cumulative_explained_variance.png", dpi=600)

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
    models[dependent] = {"coef": model.coef_.tolist(), "intercept": model.intercept_}

    preds = X @ model.coef_ + model.intercept_
    # for pred, y_true in zip(preds, y):
    #     print(f"pred: {pred}, y_true: {y_true}")

    import matplotlib.pyplot as plt
    plt.scatter(y, preds, alpha=0.01, s=1)
    min_val = min(y)
    max_val = max(y)
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted {name[dependent]} ratio")
    plt.savefig(f"src/foundation_policy/analysis/analyze_{dependent}.png", dpi=600)
    plt.show()

with open(f"src/foundation_policy/analysis/models.json", "w") as f:
    f.write(json.dumps(models))
