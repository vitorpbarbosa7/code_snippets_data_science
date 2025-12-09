import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Population Stability Index (PSI) - Full Example + Visualization + Save
# ============================================================

def psi(expected, actual, eps=1e-6):
    """
    Efficient PSI calculation given bin proportions or counts.
    expected, actual: arrays with either counts or proportions.
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)

    # normalize to proportions
    expected = expected / expected.sum()
    actual   = actual / actual.sum()

    # avoid log(0)
    ratio = (actual + eps) / (expected + eps)

    return np.sum((actual - expected) * np.log(ratio))


# ------------------------------------------------------------
# Baseline distribution (10 bins)
# ------------------------------------------------------------
expected = np.array([0.05, 0.05, 0.10, 0.10, 0.10,
                     0.15, 0.15, 0.10, 0.10, 0.10])

# ------------------------------------------------------------
# Five example "actual" cases
# ------------------------------------------------------------
cases = {
    "case1_same": expected,  # identical
    "case2_small_shift": np.array([0.04, 0.05, 0.09, 0.11, 0.11,
                                   0.16, 0.16, 0.10, 0.09, 0.09]),
    "case3_medium_shift": np.array([0.03, 0.04, 0.08, 0.09, 0.10,
                                    0.16, 0.17, 0.11, 0.11, 0.11]),
    "case4_big_shift": np.array([0.02, 0.03, 0.06, 0.07, 0.09,
                                 0.18, 0.20, 0.13, 0.11, 0.11]),
    "case5_extreme_shift": np.array([0.00, 0.01, 0.03, 0.05, 0.07,
                                     0.18, 0.22, 0.16, 0.14, 0.14]),
}

# ------------------------------------------------------------
# Compute PSI values and prepare for plotting
# ------------------------------------------------------------
psi_values = {}
for name, actual in cases.items():
    value = psi(expected, actual)
    psi_values[name] = value
    print(f"{name:20s} -> PSI: {value:.4f}")

# ------------------------------------------------------------
# Plot stacked bars to show distribution differences
# ------------------------------------------------------------
case_names = list(cases.keys())
case_labels = [f"{psi_values[name]:.3f}" for name in case_names]  # PSI value as x-label

# Stack bar data: rows=cases, columns=bins
stack_data = np.array([cases[name] for name in case_names])

fig, ax = plt.subplots(figsize=(10, 5))

bottom = np.zeros(len(case_names))
colors = plt.cm.viridis(np.linspace(0, 1, expected.size))

for i in range(expected.size):
    ax.bar(case_labels, stack_data[:, i], bottom=bottom, color=colors[i], edgecolor='none')
    bottom += stack_data[:, i]

ax.set_ylabel("Proportion per Bin")
ax.set_xlabel("Cases (PSI value)")
ax.set_title("Population Stability Index (PSI) – Distribution Drift Visualization")
ax.legend([f"Bin {i+1}" for i in range(expected.size)], title="Bins", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# ------------------------------------------------------------
# Save the figure locally
# ------------------------------------------------------------
plt.savefig("psi_visualization.png", dpi=300)
print("Plot saved as psi_visualization.png")

plt.show()

