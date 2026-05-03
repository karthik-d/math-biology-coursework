import math
import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from analysis.lagrange import analyze_error as l_analyze_error
from analysis.cubic import analyze_error as c_analyze_error
from pkg.cubic.cubic import interpolate as build_cubic_spline
from pkg.lagrange.lagrange import compute_polynomial, interpolate, divided_difference


# ---- Q1(d) test functions ----
def q1d_f1(x):
    return jnp.power(x, 3)*3 + jnp.power(x, 2)*4 + 2*x + 1

def q1d_f2(x):
    return jnp.sin(x)

def q1d_f3(x):
    return 1 / (1 + 25*jnp.power(x, 2))


# ---- Q2 test functions ----
def q2_f1(x):
	return anp.sin(x)

def q2_f2(x):
	return anp.power(x, 2) * anp.power((1 - anp.array(x)), 2)

def q2_f3(x):
    return anp.exp(x)


# ---- Run analysis for each function ----
if __name__ == "__main__":


	## 1. TEST DIVIDED DIFFERENCE.
	# def f(x):
	# 	return np.sin(x) + x**2
	# x = np.linspace(0, 2*np.pi, 8)
	# fvals = [f(xi) for xi in x]
	# table = divided_difference(x, fvals)
	# plt.figure(figsize=(6,5))
	# plt.imshow(table, origin='lower', cmap='viridis', interpolation='nearest')
	# plt.colorbar(label='Divided difference value')
	# plt.xlabel('Order j')
	# plt.ylabel('Index i')
	# plt.title('Divided Difference Table for f(x) = sin(x) + x^2')
	# plt.show()


	## 2. TEST NESTED MULTIPLICATION.
	# # Newton-form polynomial:
	# # P(x) = 1 + 2*(x-0) - 1*(x-0)*(x-1) + 3*(x-0)*(x-1)*(x+1) + 0.5*(x-0)*(x-1)*(x+1)*(x-2)
	# a = [1.0, 2.0, -1.0, 3.0, 0.5]
	# x_nodes = [0.0, 1.0, -1.0, 2.0]  # 4 nodes for 5 coefficients
	# x_test = np.linspace(-2, 3, 400)
	# P_vals = [compute_polynomial(a, x_nodes, xi) for xi in x_test]
	# P_exact = [
	# 	a[0]
	# 	+ a[1]*(x-x_nodes[0])
	# 	+ a[2]*(x-x_nodes[0])*(x-x_nodes[1])
	# 	+ a[3]*(x-x_nodes[0])*(x-x_nodes[1])*(x-x_nodes[2])
	# 	+ a[4]*(x-x_nodes[0])*(x-x_nodes[1])*(x-x_nodes[2])*(x-x_nodes[3])
	# 	for x in x_test
	# ]
	# plt.figure(figsize=(8,5))
	# plt.plot(x_test, P_vals, 'r--', label='Interpolated')
	# plt.plot(x_test, P_exact, 'k-', label='Exact', alpha=0.5)
	# plt.title('Nested multiplication visual check (Complex Newton Polynomial)')
	# plt.xlabel('x')
	# plt.ylabel('P(x)')
	# plt.legend()
	# plt.grid(True)
	# plt.show()

	## 3. Lagrange Interpolation Basic Tests.
	# # Test function (non-polynomial)
	# def f(x):
	# 	return np.sin(3*x) + 0.3*x**2

	# # Interpolation nodes
	# n = 10
	# x_nodes = np.linspace(-1, 1, n)
	# x_plot = np.linspace(-1, 1, 400)
	# f_vals = f(x_plot)
	# p_vals = [interpolate(list(x_nodes), f, x) for x in x_plot]
	# f_nodes = f(x_nodes)
	# plt.figure()
	# plt.plot(x_plot, f_vals, label="True function $f(x)=\\sin(3x)+0.3x^2$")
	# plt.plot(x_plot, p_vals, "--", label="Interpolating polynomial $P(x)$")
	# plt.scatter(x_nodes, f_nodes, color="red", zorder=3, label="Interpolation nodes")
	# plt.xlabel("x")
	# plt.ylabel("y")
	# plt.legend()
	# plt.title("Lagrange Interpolation of a Non-Polynomial Function")
	# plt.show()

	# error = np.abs(f_vals - p_vals)
	# plt.figure()
	# plt.plot(x_plot, error)
	# plt.xlabel("x")
	# plt.ylabel(r"$|f(x) - P(x)|$")
	# plt.title("Interpolation Error")
	# plt.show()

	## 4. Lagrange Interpolation Analysis.

	functions = [
	    (q1d_f1, -1.0, 1.0, "3x^3 + 4x^2 + 2x + 1", 'poly'),
	    (q1d_f2, 0.0, 2*math.pi, "sin(x)", 'sin'),
	    (q1d_f3, -1.0, 1.0, "1/(1+25x^2)", 'runge')
	]

	for f, a, b, name, string in functions:
	    l_analyze_error(f, a, b, name, string)


	### 5. Cubic Spline Analysis.
		
	# functions = [
	# 	(q2_f1, 0.0, 2*math.pi, "sin(x)"),
	# 	(q2_f2, -1.0, 1.0, "x^2 . (1-x)^2"),
	# 	(q2_f3, -1.0, 1.0, "exp(x)")
	# ]

	# for f, a, b, name in functions:
	# 	analyze_error(f, a, b, name, bc_type="complete")
			
	# for f, a, b, name in functions:
	# 	analyze_error(f, a, b, name, bc_type="not-a-knot")