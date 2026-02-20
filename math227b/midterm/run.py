import numpy as np

from pkg.descent.steepest_descent import steepest_descent
from pkg.descent import utils as descent_utils
from pkg.rk4.rk4 import rk4_solver
from pkg.rk4 import utils as rk4_utils

from analysis.rk4 import test_function
from analysis.descent import rosenbrock_functions, f1_shifted_quadratic, f2_mixed_poly_exp


if __name__=='__main__':

	## 1A. GIVEN FUNCTIION TESTS
	# f = lambda t, y: y - t**2 + 1
	# y_exact = lambda t: (t + 1)**2 - 0.5*np.exp(t)
	# t0, y0, tf = 0.0, 0.5, 4.0
	# h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
	# rk4_utils.plot_rk4_solutions(f, y_exact, t0, y0, tf, h_values)
	# rk4_utils.plot_rk4_local_and_global_error(f, y_exact, t0, y0, tf, h_values)

	# # h_values = np.logspace(0, -4, 500)
	# # rk4_utils.rk4_relative_error_heatmap(f, y_exact, t0, y0, tf, h_values)

	# h_values = np.logspace(-4, 0, 100)
	# local_errors, global_errors = rk4_utils.plot_loglog_error_with_slope(f, y_exact, t0, y0, tf, h_values)
	# rk4_utils.plot_piecewise_order(h_values, local_errors, global_errors)

	## 1B. OTHER FUNCTION TESTS

	# ---------------- Quadratic growth ----------------
	f1 = lambda t, y: np.cos(t)
	y_exact1 = lambda t: np.sin(t)
	t0_1, y0_1, tf_1 = 0.0, 0.0, 8.0
	h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]

	# Plot RK4 solutions and errors for quadratic growth
	rk4_utils.plot_rk4_solutions(f1, y_exact1, t0_1, y0_1, tf_1, h_values, title="Sinusoidal")
	rk4_utils.plot_rk4_local_and_global_error(f1, y_exact1, t0_1, y0_1, tf_1, h_values, title="Sinusoidal")
	h_values = np.logspace(-4, 0, 100)
	local_errors, global_errors = rk4_utils.plot_loglog_error_with_slope(f1, y_exact1, t0_1, y0_1, tf_1, h_values, title="Sinusoidal")

	# ---------------- Exponential decay ----------------
	f2 = lambda t, y: -2*y
	y_exact2 = lambda t: np.exp(-2*t)
	t0_2, y0_2, tf_2 = 0.0, 1.0, 2.0
	h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]

	# Plot RK4 solutions and errors for exponential decay
	rk4_utils.plot_rk4_solutions(f2, y_exact2, t0_2, y0_2, tf_2, h_values, title="Exponential Decay")
	rk4_utils.plot_rk4_local_and_global_error(f2, y_exact2, t0_2, y0_2, tf_2, h_values, title="Exponential Decay")
	h_values = np.logspace(-4, 0, 100)
	local_errors, global_errors = rk4_utils.plot_loglog_error_with_slope(f2, y_exact2, t0_2, y0_2, tf_2, h_values, title="Exponential Decay")
	

	# 2. SYSTEMATIC TEST: RK4.
	# convergence analysis
	# h_vals, errors, p_vals = run_order_analysis_clean(f, y, params)
	# # Region analysis
	# h_vals2, errors2, p_vals2, good_region = run_order_analysis_regions(f, y, params)

	
	# 3. BASIC TEST: STEEPEST DESCENT.
	# f, grad, hess = rosenbrock_functions()

	# # Check Hessian at minimizer
	# x_star = np.array([1.0, 1.0])
	# H, eigvals = check_hessian_pd(hess, x_star)

	# print("Hessian at (1,1):")
	# print(H)
	# print("Eigenvalues:", eigvals)

	# # Initial conditions
	# x0_list = [np.array([1.2, 1.2]), np.array([-1.2, 1.0])]

	# for x0 in x0_list:
	# 	print("\nInitial guess:", x0)
	# 	xmin, path = steepest_descent(f, grad, x0)
	# 	print("Computed minimizer:", xmin)
	# 	print("Function value:", f(xmin))
	# 	print("Gradient norm:", np.linalg.norm(grad(xmin)))


	## PRE-TEST 1: Other test functions.
	# f, grad = f1_shifted_quadratic()
	# x_star = np.array([1.0, -2.0])
	# x0 = np.array([3.0, 1.0])
	# xmin, path = steepest_descent(f, grad, x0)
	# descent_utils.plot_iterations_summary(path, grad, x_star, title="Shifted Quadratic: Iteration Summary")
	# descent_utils.plot_trajectory_contour(f, grad, path, xlim=(0,4), ylim=(-4,2), title="Shifted Quadratic: Trajectory over contour")

	# f, grad = f2_mixed_poly_exp()
	# x_star = np.array([0.0, 0.0])
	# x0 = np.array([0.2, 1.0])
	# xmin, path = steepest_descent(f, grad, x0)
	# descent_utils.plot_iterations_summary(path, grad, x_star, title="Mixed Exponential: Iteration Summary")
	# descent_utils.plot_trajectory_contour(f, grad, path, xlim=(-0.3,0.3), ylim=(-0.5,1.2), title="Mixed Exponential: Trajectory over contour")


	# ## MAIN TEST: ROSENBROCK.
	# f, grad, hess = rosenbrock_functions()
	# # descent_utils.plot_basin(
	# # 	f, grad,
	# # 	x_range=(-9, 11),
	# # 	y_range=(-49, 51),
	# # 	grid_size=20
	# # )
	# x_star = np.array([1, 1])
	# # starting pt. 1
	# x0 = np.array([1.2, 1.2])
	# xmin, path = steepest_descent(f, grad, x0)
	# descent_utils.plot_iterations_summary(path, grad, x_star, title="Rosenbrock from (1.2, 1.2): Iteration Summary")
	# descent_utils.plot_trajectory_contour(f, grad, path, xlim=(0.75, 1.25), ylim=(0.75, 1.25), title="Rosenbrock from (1.2, 1.2): Trajectory over contour")
	# # starting pt. 2
	# x0 = np.array([-1.2, 1])
	# xmin, path = steepest_descent(f, grad, x0)
	# descent_utils.plot_iterations_summary(path, grad, x_star, title="Rosenbrock from (-1.2, 1): Iteration Summary")
	# descent_utils.plot_trajectory_contour(f, grad, path, xlim=(-1.5, 1.25), ylim=(0.75, 1.25), title="Rosenbrock from (-1.2, 1): Trajectory over contour")






