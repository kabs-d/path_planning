import numpy as np

# Given formula: y = 499.6871 + 1171.8366 / (x + 0.5336)
def rational_decay_formula(x):
    return 499.6871 + 1171.8366 / (x + 0.5336)

# Compute values for x = 1..18
x_vals = np.arange(1, 19)
y_vals = rational_decay_formula(x_vals)

# Print results
print("Computed values (y = 499.6871 + 1171.8366 / (x + 0.5336)):\n")
for xi, yi in zip(x_vals, y_vals):
    print(f"x={xi:2d} -> y={yi:.2f}")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# Data
x = np.array([1,2,3,4,5,6,7,8,9,10,11])
y = np.array([1264, 961, 833,757, 710, 680, 656, 640, 624, 609, 599])

# Candidate functions
def linear(x, a, b): return a * x + b
def quadratic(x, a, b, c): return a * x**2 + b * x + c
def exponential(x, a, b): return a * np.exp(b * x)
def logarithmic(x, a, b): return a + b * np.log(x)
def power_law(x, a, b): return a * (x**b)
def exp_decay_floor(x, a, b, c): return c + a * np.exp(b * x)
def rational_decay(x, a, b, c): return c + a / (x + b)

models = {
    "Linear": linear,
    "Quadratic": quadratic,
    "Exponential": exponential,
    "Logarithmic": logarithmic,
    "Power Law": power_law,
    "ExpDecay+Floor": exp_decay_floor,
    "RationalDecay": rational_decay
}

results = {}

for name, func in models.items():
    try:
        if name == "Exponential":
            p0 = (1e3, -0.5)
        elif name == "ExpDecay+Floor":
            p0 = (700, -0.5, 600)
        elif name == "RationalDecay":
            p0 = (1000, 1, 600)
        elif name == "Quadratic":
            p0 = (1, 1, 1)
        else:
            p0 = (1, 1)

        params, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)

        y_pred = func(x, *params)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        results[name] = (func, params, rmse, r2)

    except Exception as e:
        print(f"Skipping {name} fit due to error: {e}")

# Pick best model
best_model = min(results.items(), key=lambda kv: kv[1][2])
best_name, (best_func, best_params, best_rmse, best_r2) = best_model

print(f"\nBest Model: {best_name}")
print(f"Parameters: {best_params}")
print(f"RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}")

# Print explicit formula
if best_name == "Linear":
    print(f"Formula: y = {best_params[0]:.4f} * x + {best_params[1]:.4f}")
elif best_name == "Quadratic":
    print(f"Formula: y = {best_params[0]:.4f} * x^2 + {best_params[1]:.4f} * x + {best_params[2]:.4f}")
elif best_name == "Exponential":
    print(f"Formula: y = {best_params[0]:.4f} * exp({best_params[1]:.4f} * x)")
elif best_name == "Logarithmic":
    print(f"Formula: y = {best_params[0]:.4f} + {best_params[1]:.4f} * ln(x)")
elif best_name == "Power Law":
    print(f"Formula: y = {best_params[0]:.4f} * x^{best_params[1]:.4f}")
elif best_name == "ExpDecay+Floor":
    print(f"Formula: y = {best_params[2]:.4f} + {best_params[0]:.4f} * exp({best_params[1]:.4f} * x)")
elif best_name == "RationalDecay":
    print(f"Formula: y = {best_params[2]:.4f} + {best_params[0]:.4f} / (x + {best_params[1]:.4f})")

# Plot
x_fit = np.linspace(min(x), max(x) + 5, 300)
y_fit = best_func(x_fit, *best_params)

plt.scatter(x, y, label="Data")
plt.plot(x_fit, y_fit, label=f"Best fit: {best_name}", color="red")

# Forecast
x_future = np.arange(max(x)+1, max(x)+8)
y_future = best_func(x_future, *best_params)

plt.scatter(x_future, y_future, color="green", marker="x", label="Forecast")
plt.legend()
plt.show()

print("\nForecast (next 7 values):")
for xi, yi in zip(x_future, y_future):
    print(f"x={xi}: {yi:.2f}")
