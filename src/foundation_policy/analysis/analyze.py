import json
import numpy as np
from tqdm import tqdm

experiment = "2025-04-16_20-10-58"
dynamics_parameters_path = f"src/foundation_policy/dynamics_parameters_{experiment}"

X = []
y = []
with open(f"src/foundation_policy/hidden_states_{experiment}.json", "r") as f:
    hidden_states = json.load(f)
    for hidden_state in tqdm(hidden_states):
        x = np.array(hidden_state["hidden_state"])
        X.append(x)
        with open(dynamics_parameters_path + f"/{hidden_state["dynamics_id"]}.json", "r") as parameters_file:
            parameters = json.load(parameters_file)
            max_action = parameters["dynamics"]["action_limit"]["max"]
            thrust = sum([np.dot(tc, [max_action ** i for i in range(len(tc))]) for tc in parameters["dynamics"]["rotor_thrust_coefficients"]])
            thrust2weight = thrust / (parameters["dynamics"]["mass"] * 9.81)
            y.append(thrust2weight)

X = np.array(X)
y = np.array(y)

# train least squares regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("R2 score: ", r2_score(y_test, y_pred))
preds = X @ model.coef_ + model.intercept_

for pred, y_true in zip(preds, y):
    print(f"pred: {pred}, y_true: {y_true}")

import matplotlib.pyplot as plt

plt.scatter(y, preds, alpha=0.01, s=1)
# identity line
min_val = min(y)
max_val = max(y)
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
plt.xlabel("True thrust-to-weight ratio")
plt.ylabel("Predicted thrust-to-weight ratio")
plt.title("True vs Predicted thrust-to-weight ratio")
plt.savefig("src/foundation_policy/analysis/thrust_to_weight_ratio.png", dpi=600)
plt.show()
