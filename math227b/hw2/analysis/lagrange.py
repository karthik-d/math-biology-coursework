import math
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as anp
import numpy as np

from pkg.lagrange import lagrange


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


# ---- Function to generate equally spaced nodes ----
def generate_nodes(a, b, n):
    return [a + i*(b-a)/(n-1) for i in range(n)]


# ---- Function to compute theoretical Lagrange error ----
# def theoretical_error_bound(f, n, x_nodes, x_test, num_dr_eval=500):
#     """
#     Compute theoretical Lagrange error bound using auto-diff for derivatives.
#     Handles vector inputs using elementwise_grad.
#     """
#     # 1. Build nth derivative function
#     df = f
#     for _ in range(n):
#         df = autograd.elementwise_grad(df)

#     # 2. Sample dense grid to estimate max |f⁽ⁿ⁾(x)|
#     a, b = x_nodes[0], x_nodes[-1]
#     xs = anp.linspace(a, b, num_dr_eval)
#     deriv_vals = anp.abs(df(xs))  # elementwise_grad works on vector xs
#     max_f_deriv = float(anp.max(deriv_vals))

#     # 3. Compute error bound at each x_test
#     fact = math.factorial(n)
#     error_bound = []
#     for x in x_test:
#         omega = 1.0
#         for xi in x_nodes:
#             omega *= (x - xi)
#         error_bound.append(abs(max_f_deriv * omega / fact))

#     return error_bound


def theoretical_error_bound(f, n, x_nodes, x_test, num_dr_eval=500):
    """
    Compute theoretical Lagrange interpolation error bound safely using auto-diff.
    Uses log-space for the node polynomial to prevent overflow.
    
    Parameters
    ----------
    f : callable
        Autograd-compatible function (uses anp operations)
    n : int
        Number of nodes
    x_nodes : list of floats
        Equally spaced interpolation nodes
    x_test : list/array of floats
        Points in [a,b] where to evaluate the bound
    num_dr_eval : int
        Number of points to evaluate n-th derivative for max estimate
        
    Returns
    -------
    error_bound : np.array
        Theoretical error bound at each x_test[i]
    """
    # 1. Compute nth derivative using auto-diff
    df = f
    for _ in range(n):
        df = autograd.elementwise_grad(df)

    # 2. Evaluate max absolute n-th derivative on dense grid
    a, b = x_nodes[0], x_nodes[-1]
    xs_dense = anp.linspace(a, b, num_dr_eval)
    deriv_vals = anp.abs(df(xs_dense))
    max_f_deriv = float(anp.max(deriv_vals))

    # 3. Factorial term
    fact = math.factorial(n)

    # 4. Compute node polynomial safely in log-space
    error_bound = []
    for x in x_test:
        # log(|prod(x - xi)|) = sum(log(|x - xi|))
        log_omega = np.sum(np.log(np.abs(x - np.array(x_nodes))))
        omega = np.exp(log_omega)
        error_bound.append(max_f_deriv * omega / fact)

    return np.array(error_bound)


# ---- Function to compute interpolation results ----
def interpolation_results(f, a, b, n, x_test):
    x_nodes = generate_nodes(a, b, n)
    p_vals = [lagrange.interpolate(x_nodes, f, x) for x in x_test]
    f_vals = [f(x) for x in x_test]
    errors = [abs(fv - pv) for fv, pv in zip(f_vals, p_vals)]
    max_error = max(errors)
    rms_error = math.sqrt(sum(e**2 for e in errors)/len(errors))
    return p_vals, f_vals, errors, max_error, rms_error, x_nodes


# ---- Main analysis function ----
def analyze_error(f, a, b, func_name):
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
		bound = theoretical_error_bound(f, n, x_nodes, x_test)
		max_bounds.append(max(bound))

	# ---- Plot max and RMS error vs n ----
	plt.figure(figsize=(8,6))
	plt.plot(n_values, max_errors, 'o-', label='Max error', alpha=0.8)
	plt.plot(n_values, rms_errors, 's-', label='RMS error', alpha=0.8)
	plt.plot(n_values, max_bounds, '^-', label='Max theoretical bound', alpha=0.8)
	plt.xlabel('Number of nodes n')
	plt.ylabel('Error')
	plt.yscale('log')
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



