from matplotlib import pyplot as plot
import numpy as np


def plot_errors(errors):
    """
    Plot Newton convergence diagnostics:
      (1) error history ||x_k - x*||
      (2) log-log plot of e_{k+1} vs e_k with slope-2 reference

    Non-asymptotic region is marked on the plots.
    """

    errors = np.array(errors)

    iterations = np.arange(len(errors))

    # Prepare log-log data: e_{k+1} vs e_k
    e_k = errors[:-1]
    e_k1 = errors[1:]

    # Heuristic detection of asymptotic region:
    # Look for region where slope is approximately 2
    slopes = np.log(e_k1[1:] / e_k1[:-1]) / np.log(e_k[1:] / e_k[:-1])
    asymptotic_start = None
    for i in range(len(slopes)):
        if np.all(np.abs(slopes[i:i+3] - 2.0) < 0.3):  # 3-step window near slope 2
            asymptotic_start = i
            break

    if asymptotic_start is None:
        asymptotic_start = len(e_k)

    fig, axes = plot.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: error history ---
    axes[0].semilogy(iterations, errors, 'o-', label=r"$e_k = \|x_k - x^*\|$")
    axes[0].axvspan(0, asymptotic_start, color='red', alpha=0.1, label="non-asymptotic")
    axes[0].axvspan(asymptotic_start, len(errors), color='green', alpha=0.1, label="asymptotic")

    axes[0].set_xlabel("Iteration k")
    axes[0].set_ylabel(r"Error $\|x_k - x^*\|$")
    axes[0].set_title("Newton Error Convergence")
    axes[0].grid(True, which="both")
    axes[0].legend()

    # --- Plot 2: log-log error plot ---
    axes[1].loglog(e_k, e_k1, '-o', label=r"$(e_k, e_{k+1})$")

    # Slope-2 reference line: e_{k+1} = C e_k^2
    if asymptotic_start < len(e_k):
        C = e_k1[asymptotic_start] / e_k[asymptotic_start]**2
    else:
        C = e_k1[0] / e_k[0]**2

    ref_x = np.array([e_k.min(), e_k.max()])
    ref_y = C * ref_x**2
    axes[1].loglog(ref_x, ref_y, '--', label="slope = 2 reference")

    axes[1].set_xlabel(r"$e_k$")
    axes[1].set_ylabel(r"$e_{k+1}$")
    axes[1].set_title("Log–Log Error Plot (Quadratic Convergence)")
    axes[1].grid(True, which="both")
    axes[1].legend()

    fig.suptitle("Convergence of Newton's Method (Error-Based Diagnostics)")
    plot.tight_layout()
    plot.show()
