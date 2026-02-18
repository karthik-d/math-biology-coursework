import numpy as np

from pkg.rk4.rk4 import rk4_solver
from pkg.rk4.utils import plot_solutions
from analysis.basic import test_function, run_order_analysis_clean, run_order_analysis_regions


if __name__=='__main__':

	# unload test function.
	f, y, params = test_function()

	## 1. BASIC TEST.
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

	
	## 2. SYSTEMATIC TEST.
	# h_values = np.logspace(-4, 0, 50)
	# errors, p_values = run_order_analysis(f, y, params, h_values)

	# Clean convergence analysis
	h_vals, errors, p_vals = run_order_analysis_clean(f, y, params)

	# Region analysis
	h_vals2, errors2, p_vals2, good_region = run_order_analysis_regions(f, y, params)


