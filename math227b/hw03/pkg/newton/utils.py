from matplotlib import pyplot as plot
import numpy as np


import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(errors):
    """
    Plot Newton convergence diagnostics:
      (1) error history ||x_k - x*||
      (2) quadratic convergence ratio e_{k+1}/e_k^2

    Non-asymptotic region is marked on the plots.
    """

    errors = np.array(errors)

    # Compute quadratic ratios
    ratios = errors[1:] / errors[:-1]**2
    iterations = np.arange(len(errors))
    ratio_iter = np.arange(len(ratios))

    # Heuristic detection of asymptotic region:
    # Look for region where ratio stabilizes (relative change small)
    rel_change = np.abs(np.diff(ratios) / ratios[:-1])
    asymptotic_start = None
    for i in range(len(rel_change)):
        if np.all(rel_change[i:i+3] < 0.2):  # 3-step stability window
            asymptotic_start = i
            break

    if asymptotic_start is None:
        asymptotic_start = len(ratios)  # no clear asymptotic region found

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: error history ---
    axes[0].semilogy(iterations, errors, 'o-', label=r"$e_k = \|x_k - x^*\|$")
    axes[0].axvspan(0, asymptotic_start, color='red', alpha=0.1, label="non-asymptotic")
    axes[0].axvspan(asymptotic_start, len(errors), color='green', alpha=0.1, label="asymptotic")

    axes[0].set_xlabel("Iteration k")
    axes[0].set_ylabel(r"Error $\|x_k - x^*\|$")
    axes[0].set_title("Newton Error Convergence")
    axes[0].grid(True, which="both")
    axes[0].legend()

    # --- Plot 2: quadratic ratio ---
    axes[1].semilogy(ratio_iter, ratios, 'o-', label=r"$e_{k+1}/e_k^2$")

    if asymptotic_start < len(ratios):
        ref_value = np.mean(ratios[asymptotic_start:])
        axes[1].semilogy(ratio_iter, ref_value * np.ones_like(ratio_iter),
                         '--', label="constant reference")

    axes[1].axvspan(0, asymptotic_start, color='red', alpha=0.1, label="non-asymptotic")
    axes[1].axvspan(asymptotic_start, len(ratios), color='green', alpha=0.1, label="asymptotic")

    axes[1].set_xlabel("Iteration k")
    axes[1].set_ylabel(r"$e_{k+1}/e_k^2$")
    axes[1].set_title("Quadratic Convergence Ratio")
    axes[1].grid(True, which="both")
    axes[1].legend()

    fig.suptitle("Convergence of Newton's Method (Error-Based Diagnostics)")
    plt.tight_layout()
    plt.show()
