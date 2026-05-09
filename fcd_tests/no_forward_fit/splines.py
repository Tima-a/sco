import numpy as np
from scipy.interpolate import UnivariateSpline, PPoly
import matplotlib.pyplot as plt

# 1. Dataset
y = np.random.normal(0, 500, 100).cumsum() + 60000 
x = np.arange(len(y))

# 2. Fit the Smoothing Spline
s_level = 1e6 # Bitcoin-scale noise needs a larger s_level
spline = UnivariateSpline(x, y, k=3, s=s_level)

# 3. Correct way to get PP-form from UnivariateSpline
# We must use the full knot vector (t), coefficients (c), and degree (k)
tck = (spline.get_knots(), spline.get_coeffs(), 3)
# To use from_spline, we need the "full" tck tuple which includes boundary knots
# SciPy's UnivariateSpline makes this a bit tricky, so we use this conversion:
pp = PPoly.from_spline(spline._eval_args) 

# 4. Print Equations
n_segments = len(pp.x) - 1
print(f"Total Segments: {n_segments}")

for i in range(min(n_segments, 10)): # Print first 10
    a, b, c, d = pp.c[:, i]
    x0 = pp.x[i]
    x1 = pp.x[i+1]
    equation = f"y = {a:+.4e}(x-{x0})^3 {b:+.4e}(x-{x0})^2 {c:+.4e}(x-{x0}) {d:+.4e}"
    print(f"Seg {i} [{x0:.1f} to {x1:.1f}]: {equation}")

# 5. Plotting
x_smooth = np.linspace(x.min(), x.max(), 500) # Range must match your data (0 to 100)
y_smooth = spline(x_smooth)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=10, color='gray', alpha=0.5, label="Data")
plt.plot(x_smooth, y_smooth, color='red', label="Smoothed Spline")
plt.legend()
plt.grid(True)
plt.show()