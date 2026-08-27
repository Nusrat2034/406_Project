# /// script
# dependencies = [
#     "numpy",
#     "matplotlib"
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Main Loss Function Curve (Red/Brown)
# Equation: sin(2x) + 0.5*cos(4x) - 0.1*x^2 for -1 <= x <= 1
x_curve = np.linspace(-1, 1, 400)
y_curve = np.sin(2 * x_curve) + 0.5 * np.cos(4 * x_curve) - 0.1 * (x_curve**2)
ax.plot(x_curve, y_curve, color="#c0392b", linewidth=2.5, label=r"$L(x) = \sin(2x) + 0.5\cos(4x) - 0.1x^2$")

# 2. FGSM Straight Line (Blue)
# Equation: 0.526000798 * x for -0.22122 <= x <= 0.8271
x_fgsm_line = np.linspace(-0.22122, 0.8271, 100)
y_fgsm_line = 0.526000798 * x_fgsm_line
ax.plot(x_fgsm_line, y_fgsm_line, color="#2980b9", linewidth=2.5, label="FGSM (Single Step Jump)")

# 3. PGD Multi-step Points (Green points along the curve)
pgd_points_x = [-0.33, -0.12, 0.08, 0.16, 0.27, 0.62, 0.82]
pgd_points_y = [
    np.sin(2 * x) + 0.5 * np.cos(4 * x) - 0.1 * (x**2) for x in pgd_points_x
]
ax.plot(
    pgd_points_x,
    pgd_points_y,
    "o",
    color="#27ae60",
    markersize=6,
    label="PGD Steps",
)

# 4. FGSM Start and End Points (Grey dots)
fgsm_dots_x = [-0.22122, 0.8271]
fgsm_dots_y = [0.526000798 * x for x in fgsm_dots_x]
ax.plot(fgsm_dots_x, fgsm_dots_y, "o", color="#7f8c8d", markersize=6)

# 5. Text Annotations matching the Desmos setup
ax.text(0.35, 1.05, "PGD", color="#4a235a", fontsize=11, fontweight="bold")
ax.text(0.37, 0.85, "( Multi Step Prediction )", color="#2980b9", fontsize=10)
ax.text(0.85, 0.25, "FGSM", color="#000000", fontsize=11, fontweight="bold")
ax.text(0.85, 0.05, "( Single Step Jump )", color="#c0392b", fontsize=10)

# Configure plot boundaries and grid to match the Desmos view
ax.set_xlim(-1.7, 2.2)
ax.set_ylim(-1.7, 1.5)

# Axis lines through origin (x=0, y=0)
ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)

# Major and Minor grid setup
ax.grid(True, which="both", color="#dcdde1", linestyle="-", linewidth=0.7)
ax.set_xticks(np.arange(-1.5, 2.5, 0.5))
ax.set_yticks(np.arange(-1.5, 2.0, 0.5))

# Labels and Title
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("Loss L(x)", fontsize=12)
ax.set_title("PGD vs. FGSM Attack Methods Visualization", fontsize=14, pad=12)

# Display Legend
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

plt.tight_layout()
plt.savefig('FGSM-vs-PGD.png')