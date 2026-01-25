import math
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as anp
import numpy as np

import jax
import jax.numpy as jnp

from pkg.lagrange import lagrange
from analysis import utils


# ---- Function to compute theoretical max derivative (n-th) ----
def max_nth_derivative(f, n, a, b, num_points=1000):
    """Estimate max of n-th derivative numerically using finite differences."""
    # Simple finite difference approximation
    dx = (b - a) / (num_points - 1)
    xs = [a + i*dx for i in range(num_points)]
    # Recursive derivative approximation
    y_vals = [f(xi) for xi in xs]
    for _ in range(n):
        y_vals = [(y_vals[i+1] - y_vals[i]) / dx for i in range(len(y_vals)-1)]
    return max(abs(yi) for yi in y_vals)


def analytical_error_bound(f_name, n, x_nodes, x_test):
    """
    Analytical Lagrange bound using log-space to avoid overflow.
    f_name: 'poly', 'sin', 'runge'
    """
    if f_name == 'poly':
        max_f_deriv = 0.0   # polynomial of degree <= n → exact
    elif f_name == 'sin':
        max_f_deriv = 1.0   # max |sin^(n)(x)| = 1
    elif f_name == 'runge':
        max_f_deriv = math.factorial(n) * 25**n  # bound at x=0
    else:
        raise ValueError("Unknown function")

    fact = math.factorial(n)
    x_nodes_arr = np.array(x_nodes)
    error_bound = []
    eps = 1e-16
    for x in x_test:
        diff = x - x_nodes_arr
        diff = np.where(diff == 0, eps, diff)
        log_omega = np.sum(np.log(np.abs(diff)))
        log_bound = np.log(max_f_deriv + 1e-16) + log_omega - np.log(fact)
        error_bound.append(np.exp(log_bound))

    return np.array(error_bound)



# def theoretical_error_bound(f, n, x_nodes, x_test, num_dr_eval=500):
#     """
#     Approximate theoretical Lagrange interpolation bound efficiently.

#     f : callable OR list of precomputed f(x_i)
#     n : number of nodes
#     x_nodes : interpolation nodes
#     x_test : points where bound is evaluated
#     """
#     # Evaluate f at nodes if callable
#     if callable(f):
#         fvals = np.array([f(xi) for xi in x_nodes])
#     else:
#         fvals = np.array(f)

#     # Compute divided differences
#     a = lagrange.divided_difference(list(x_nodes), list(fvals))
#     # a[-1] is f[x_0,...,x_n] = f^{(n)}(ξ)/n! (numerical approximation)
#     coeff = abs(a[-1])

#     # Compute |prod(x - x_i)| safely in log-space
#     x_nodes_arr = np.array(x_nodes)
#     error_bound = []
#     eps = 1e-16  # avoid log(0)
#     for x in x_test:
#         diff = x - x_nodes_arr
#         diff = np.where(diff == 0, eps, diff)
#         log_prod = np.sum(np.log(np.abs(diff)))
#         error_bound.append(coeff * np.exp(log_prod))

#     return np.array(error_bound)


# ---- Function to compute interpolation results ----
def interpolation_results(f, a, b, n, x_test):
    x_nodes = utils.generate_nodes(a, b, n)
    p_vals = [lagrange.interpolate(x_nodes, f, x) for x in x_test]
    f_vals = [f(x) for x in x_test]
    errors = [abs(fv - pv) for fv, pv in zip(f_vals, p_vals)]
    max_error = max(errors)
    plt.show()
    rms_error = math.sqrt(sum(e**2 for e in errors)/len(errors))
    return p_vals, f_vals, errors, max_error, rms_error, x_nodes


# ---- Main analysis function ----
def analyze_error(f, a, b, func_name, func_str):
	n_values = list(range(3, 15))
	num_test_points = 500
	x_test = [a + i*(b-a)/(num_test_points-1) for i in range(num_test_points)]

	max_errors = []
	rms_errors = []
	max_bounds = []

	p_vals_all = []
	f_vals_all = []
	for n in n_values:
		# Interpolation
		p_vals, f_vals, errors, max_err, rms_err, x_nodes = interpolation_results(f, a, b, n, x_test)
		max_errors.append(max_err)
		rms_errors.append(rms_err)
		p_vals_all.append(p_vals)
		f_vals_all.append(f_vals)

		# Theoretical bound
		bound = analytical_error_bound(func_str, n, x_nodes, x_test)
		max_bounds.append(max(bound))

	# ---- Plot max and RMS error vs n ----
	plt.figure(figsize=(8,6))
	plt.plot(n_values, max_errors, 'o-', label='Max error', alpha=0.8)
	plt.plot(n_values, rms_errors, 's-', label='RMS error', alpha=0.8)
	plt.plot(n_values, max_bounds, '^-', label='Max theoretical bound', alpha=0.8)
	plt.xlabel('Number of nodes n')
	plt.ylabel('Error')
	# plt.yscale('log')
	plt.title(f'Lagrange Interpolation: Error vs n for {func_name}')
	plt.legend()
	plt.grid(True)
	plt.show()
		
	# ---- Plot actual function and interpolated values ----
	print(len(p_vals_all))
	plt.figure(figsize=(16,16))
	for i, n in enumerate(n_values):
		plt.subplot(4, 3, i+1)
		plt.plot(x_test, p_vals_all[i], 'k--', label=f'Interpolated')
		plt.plot(x_test, f_vals_all[i], 'b', label=f'Analytical', alpha=0.3)
		plt.xlabel('x')
		plt.ylabel('f(x)')
		plt.title(f'n={n}')
		plt.grid(True)
	plt.suptitle(f'Lagrange Interpolation: Interpolation vs Actual for {func_name}')
	plt.legend()
	plt.show()



