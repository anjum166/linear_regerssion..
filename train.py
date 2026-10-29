import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.linear_regression import LinearRegressionScratch


# ---------- Manual metrics (no library functions) ----------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ---------- 1. Load dataset ----------
data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=["MedHouseVal"]).values
y = df["MedHouseVal"].values

# ---------- 2. Train/test split + standardization ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- 3. Train from-scratch model ----------
model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# ---------- 4. Cost convergence plot ----------
plt.figure(figsize=(8, 5))
plt.plot(model.cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Gradient descent convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/cost_convergence.png", dpi=150)
plt.close()

# ---------- 5. Evaluate from-scratch model ----------
y_pred_scratch = model.predict(X_test)
mse_scratch = mse(y_test, y_pred_scratch)
r2_scratch = r2_score_manual(y_test, y_pred_scratch)

# ---------- 6. scikit-learn comparison ----------
sk_model = LinearRegression().fit(X_train, y_train)
y_pred_sk = sk_model.predict(X_test)
mse_sk = mse(y_test, y_pred_sk)
r2_sk = r2_score_manual(y_test, y_pred_sk)

print("From-scratch model")
print(f"  MSE: {mse_scratch:.6f}   R²: {r2_scratch:.6f}")
print(f"  Intercept: {model.theta[0]:.6f}")
print(f"  Coefficients: {np.round(model.theta[1:], 6)}")
print()
print("scikit-learn model")
print(f"  MSE: {mse_sk:.6f}   R²: {r2_sk:.6f}")
print(f"  Intercept: {sk_model.intercept_:.6f}")
print(f"  Coefficients: {np.round(sk_model.coef_, 6)}")
print()
print(f"Max |coef difference|: "
      f"{np.max(np.abs(model.theta[1:] - sk_model.coef_)):.2e}")
print(f"R² difference: {abs(r2_scratch - r2_sk):.2e}")