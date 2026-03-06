import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from pkg.ivp.solver import f_linear, solve_pc
from analysis.stability_regions import (plot_stability_predictor,
    plot_stability_corrector,
    plot_stability_predictor_corrector,
    plot_stability_overlay)
from analysis.error import analyze_pc_order, analyze_true_lte


def exact_solution_linear_system(t, A, y0):
    """
    Exact solution y(t) = exp(A t) y0 for the linear system y' = A y.
    """
    At = A * t
    return expm(At) @ y0


def solve_example_system():
    
    A = np.array([[-5.0,   3.0],
                  [100.0, -301.0]])
    y0 = np.array([52.29, 83.82])
    
    t0, T = 0.0, 0.1
    h = 1e-3
    
    # Predictor–Corrector solution
    t, Y = solve_pc(f_linear, (t0, T), y0, h, A)

    # Exact reference
    Y_ref = np.zeros_like(Y)
    for k, tk in enumerate(t):
        Y_ref[k] = expm(A * tk) @ y0

    # SciPy solve_ivp reference
    def f_ivp(t, y):
        return f_linear(t, y, A)

    sol = solve_ivp(
        f_ivp,
        (t0, T),
        y0,
        method="RK45",
        t_eval=t,
        rtol=1e-9,
        atol=1e-12
    )

    Y_ivp = sol.y.T

    # Plot results
    plt.figure(figsize=(8, 5))

    # Predictor–Corrector
    plt.plot(t, Y[:, 0], "b-", label="y1 (PC)")
    plt.plot(t, Y[:, 1], "r-", label="y2 (PC)")

    # Exact
    plt.plot(t, Y_ref[:, 0], "b--", label="y1 (exact)")
    plt.plot(t, Y_ref[:, 1], "r--", label="y2 (exact)")

    # SciPy solve_ivp
    plt.plot(t, Y_ivp[:, 0], "b:", label="y1 (solve_ivp)")
    plt.plot(t, Y_ivp[:, 1], "r:", label="y2 (solve_ivp)")

    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Predictor–Corrector vs exact vs solve_ivp (linear stiff system)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
	## A. plot stability regions.
	# plot_stability_predictor()
	# plt.show()

	# plot_stability_corrector()
	# plt.show()

	# plot_stability_predictor_corrector()
	# plt.show()

	# plot_stability_overlay()
	# plt.show()

	## B. solve the given system using the predictor–corrector scheme
	# solve_example_system()

	## C. analyze LTE vs h.
	analyze_pc_order()
	analyze_true_lte()

	## D. analyze error on other function.
	# =============================================================================
	# TEST SYSTEMS: Both GLOBAL error (analyze_pc_order) and LOCAL LTE (analyze_true_lte)
	# =============================================================================

