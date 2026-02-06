from matplotlib import pyplot as plt
import numpy as np


def plot_errors(errors, title_suffix=""):
    """
    Plot Newton convergence diagnostics:
      (1) Error history (semilog)
      (2) Log–log plot of e_{k+1} vs e_k with slope-2 reference

    This version is clean, with no region highlighting.
    """

    errors = np.array(errors)
    iterations = np.arange(len(errors))

    # Prepare log-log data: e_{k+1} vs e_k
    e_k = errors[:-1]
    e_k1 = errors[1:]

    # --- Slope-2 reference line ---
    C = e_k1[0] / (e_k[0]**2)  # use first ratio as reference
    ref_x = np.array([e_k.min(), e_k.max()])
    ref_y = C * ref_x**2

    # --- Create figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- Plot 1: Error history ----------
    axes[0].semilogy(iterations, errors, 'o-', label=r"$e_k = \|x_k - x^*\|$")
    axes[0].set_xlabel("Iteration $k$")
    axes[0].set_ylabel(r"Error $\|x_k - x^*\|$")
    axes[0].set_title(f"Semi-log Error Convergence: {title_suffix}")
    axes[0].grid(True, which="both")
    axes[0].legend(fontsize=9)

    # ---------- Plot 2: Log–Log Error Plot ----------
    axes[1].loglog(e_k, e_k1, 'o-', label=r"$(e_k, e_{k+1})$")
    axes[1].loglog(ref_x, ref_y, '--', label="Quadratic slope reference")
    axes[1].set_xlabel(r"$e_k$")
    axes[1].set_ylabel(r"$e_{k+1}$")
    axes[1].set_title(f"Log–Log Error vs. Previous Error: {title_suffix}")
    axes[1].grid(True, which="both")
    axes[1].legend(fontsize=9)

    plt.suptitle("Newton Convergence Diagnostics", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.show()
