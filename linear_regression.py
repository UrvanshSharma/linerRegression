import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------- DATA ----------------
data = {
    "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
    "Marks":[35,40,50,55,65,70,75,85,88,95]
}

df = pd.DataFrame(data)
print("\nDataset Preview\n", df)

X = df[["Hours_Studied"]]
y = df["Marks"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
print(f"\n{'MODEL PERFORMANCE':^40}")
print("="*40)
print(f"MSE        : {mean_squared_error(y_test, y_pred):.3f}")
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"RMSE : {rmse:.3f}")
print(f"MAE        : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R2 Score   : {r2_score(y_test, y_pred):.3f}")
print(f"Slope      : {model.coef_[0]:.3f}")
print(f"Intercept  : {model.intercept_:.3f}")
print(f"Equation   : Marks = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}")
print("="*40)

# ---------------- SAFE PREDICTION ----------------
new_hours = pd.DataFrame({"Hours_Studied":[7.5]})
predicted_marks = model.predict(new_hours)
print("\nPredicted Marks for 7.5 hours:", round(predicted_marks[0],2))


# ---------------- PLOT ----------------
plt.style.use("seaborn-v0_8-muted")
plt.figure(figsize=(10,6))

# scatter
plt.scatter(
    X, y,
    s=100,
    edgecolor="black",
    alpha=0.85,
    label="Actual Data"
)

# smooth regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_line = model.predict(X_line)

plt.plot(
    X_line, y_line,
    linewidth=3,
    linestyle="--",
    label="Regression Trend"
)

plt.title("Study Hours vs Marks — Linear Regression", fontsize=15, weight="bold")
plt.xlabel("Hours Studied", fontsize=12)
plt.ylabel("Marks", fontsize=12)

# add R2 on chart
plt.text(1, 90, f"R² = {r2_score(y_test, y_pred):.3f}", fontsize=11)

plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("study_regression_plot.png", dpi=300)
plt.show()
