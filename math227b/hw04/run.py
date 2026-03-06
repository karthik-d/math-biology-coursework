import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pkg.ivp.solver import f_linear, solve_pc
from analysis.stability_regions import (plot_stability_predictor,
	plot_stability_corrector,
	plot_stability_predictor_corrector,
	plot_stability_overlay)


def exact_solution_linear_system(t, A, y0):
    """
    Exact solution y(t) = exp(A t) y0 for the linear system y' = A y,
    using matrix exponential for reference (small system).
    """
    At = A * t
    return expm(At) @ y0


def solve_example_system():
	
	A = np.array([[-5.0,   3.0],
				[100.0, -301.0]])
	y0 = np.array([52.29, 83.82])
	t0, T = 0.0, 0.1   # short interval because system is stiff
	h = 1e-3
	t, Y = solve_pc(f_linear, (t0, T), y0, h, A)
	Y_ref = np.zeros_like(Y)
	
	for k, tk in enumerate(t):
		Y_ref[k] = expm(A * tk) @ y0

	plt.figure(figsize=(8, 5))
	plt.plot(t, Y[:, 0], "b-", label="y1 (PC)")
	plt.plot(t, Y[:, 1], "r-", label="y2 (PC)")
	plt.plot(t, Y_ref[:, 0], "b--", label="y1 (exact)")
	plt.plot(t, Y_ref[:, 1], "r--", label="y2 (exact)")
	plt.xlabel("t")
	plt.ylabel("y")
	plt.title("Predictor–Corrector solution vs exact (linear system)")
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
    
	## B. solve the given system using the predictor–corrector scheme and compare to exact solution.
	solve_example_system()

    
