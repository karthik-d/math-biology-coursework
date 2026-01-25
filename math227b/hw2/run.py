import math
import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt

from analysis.lagrange import analyze_error
from analysis.cubic import analyze_error
from pkg.cubic.cubic import interpolate as build_cubic_spline


# ---- Q1(d) test functions ----
def q1d_f1(x):
    return 3*x**3 + 4*x**2 + 2*x + 1

def q1d_f2(x):
    return anp.sin(x)

def q1d_f3(x):
    return 1 / (1 + 25*x**2)


# ---- Q2 test functions ----
def q2_f1(x):
	return anp.sin(x)

def q2_f2(x):
	return anp.power(x, 2) * anp.power((1 - anp.array(x)), 2)

def q2_f3(x):
    return anp.exp(x)


# ---- Run analysis for each function ----
if __name__ == "__main__":

	### 1. Lagrange Interpolation Analysis.

	# functions = [
	#     (q1d_f1, -1.0, 1.0, "3x^3 + 4x^2 + 2x + 1"),
	#     (q1d_f2, 0.0, 2*math.pi, "sin(x)"),
	#     (q1d_f3, -1.0, 1.0, "1/(1+25x^2)")
	# ]

	# for f, a, b, name in functions:
	#     analyze_error(f, a, b, name)


	### 2. Cubic Spline Analysis.
     
	functions = [
		(q2_f1, 0.0, 2*math.pi, "sin(x)"),
		(q2_f2, -1.0, 1.0, "x^2 . (1-x)^2"),
		(q2_f3, -1.0, 1.0, "exp(x)")
	]

	for f, a, b, name in functions:
		analyze_error(f, a, b, name, bc_type="natural")
            
	for f, a, b, name in functions:
		analyze_error(f, a, b, name, bc_type="not-a-knot")