import numpy as np

# ============================================================
# Population Stability Index (PSI) - Full Example
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
# Compute and display PSI results
# ------------------------------------------------------------
for name, actual in cases.items():
    value = psi(expected, actual)
    print(f"{name:20s} -> PSI: {value:.4f}")

# Expected output (approx):
# case1_same           -> PSI: 0.0000
# case2_small_shift    -> PSI: 0.0086
# case3_medium_shift   -> PSI: 0.0240
# case4_big_shift      -> PSI: 0.0995
# case5_extreme_shift  -> PSI: 0.8224

