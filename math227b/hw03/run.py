import numpy as np

from pkg.newton.newton import newton_system
from pkg.newton.utils import plot_errors, plot_newton_convergence, plot_residual_heatmap
from analysis.basic import q1b_visual_check_f1, q1b_visual_check_f2, q1c_f1, q1c_f2, q1c_f3
from analysis.two_gene import (compute_bistable_solutions, sweep_n, sweep_alpha_max, sweep_ecx, 
							   basin_of_attraction, test_newton_convergence)


if __name__ == "__main__":

	# --------------------------
	# 0. Unit tests.
	# Unit tests are defined in pkg/test and can be run using `python -m unittest discover -s pkg/test` from the root directory.
	# --------------------------


	# --------------------------
	# 1. Basic Tests.
	# --------------------------

	# A. convergence tests for chosen functions.
	test_newton_convergence()
		
	# B. visual verification.
	# function N4.
	F, J, x0, x_true = q1b_visual_check_f1()
	plot_residual_heatmap(F, J, x0, x_true, title="Residual Heatmap")
	plot_newton_convergence(F, J, x0, x_true, title="Convergence History")

	# function N2.
	F, J, x0, x_true = q1b_visual_check_f2()
	plot_residual_heatmap(F, J, x0, x_true, title="Residual Heatmap", grid_bounds=(0, 3))
	plot_newton_convergence(F, J, x0, x_true, title="Convergence History")

	
	# --------------------------
	# 2. Two-Gene Network.
	# --------------------------

	# print("=== Step 1: Test Newton solver on known functions ===")
	# test_newton_code()

	print("=== Step 2: Compute bistable steady states ===")
	compute_bistable_solutions()

	print("=== Step 3: Parameter sweeps ===")
	# sweep_n()
	# sweep_alpha_max()
	# sweep_ecx()

	print("=== Step 4: Basin of attraction ===")
	# params = {
	#     'alpha_min': 0.1,
	#     'alpha_max': 5.5,
	#     'alpha_deg': 1.0,
	#     'beta_min': 0.1,
	#     'beta_max': 4.5,
	#     'beta_deg': 0.9,
	#     'e_cx': 1.0,
	#     'e_cy': 1.5,
	#     'n': 4
	# }
	# basin_of_attraction(params)
    


