import numpy as np

from pkg.descent.steepest_descent import steepest_descent
from pkg.descent.utils import check_hessian_pd
from pkg.rk4.rk4 import rk4_solver
from pkg.rk4.utils import plot_solutions

from analysis.basic import test_function, run_order_analysis_clean, run_order_analysis_regions
from analysis.descent import rosenbrock_functions


if __name__=='__main__':

	# unload test function.
	f, y, params = test_function()

	## 1. BASIC TEST: RK4.
	# h=0.1
	# # run solver.
	# t_rk, y_rk = rk4_solver(f, params['t0'], params['y0'], h, params['tf'])

	# # analytical solution.
	# t_ex = np.linspace(params['t0'], params['tf'], 400)
	# y_ex = y(t_ex)

	# # plot.
	# plot_solutions(t_rk, y_rk, t_ex, y_ex)

	# # Error at t = 4
	# y_exact_tf = y(params['tf'])
	# rk_error = abs(y_rk[-1] - y_exact_tf)
	# print(rk_error)

	
	## 2. SYSTEMATIC TEST: RK4.
	# h_values = np.logspace(-4, 0, 50)
	# errors, p_values = run_order_analysis(f, y, params, h_values)

	# Clean convergence analysis
	# h_vals, errors, p_vals = run_order_analysis_clean(f, y, params)

	# # Region analysis
	# h_vals2, errors2, p_vals2, good_region = run_order_analysis_regions(f, y, params)

	
	## 3. BASIC TEST: STEEPEST DESCENT.
	f, grad, hess = rosenbrock_functions()

	# Check Hessian at minimizer
	x_star = np.array([1.0, 1.0])
	H, eigvals = check_hessian_pd(hess, x_star)

	print("Hessian at (1,1):")
	print(H)
	print("Eigenvalues:", eigvals)

	# Initial conditions
	x0_list = [np.array([1.2, 1.2]), np.array([-1.2, 1.0])]

	for x0 in x0_list:
		print("\nInitial guess:", x0)
		xmin, path = steepest_descent(f, grad, x0)
		print("Computed minimizer:", xmin)
		print("Function value:", f(xmin))
		print("Gradient norm:", np.linalg.norm(grad(xmin)))





