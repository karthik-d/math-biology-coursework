import numpy as np	
from matplotlib import pyplot as plt

from pkg.rk4.rk4 import rk4_solver


def test_function():
	
	# ode function.
	def f(t, y):
		return y - t**2 + 1

	# analytical soln. function.
	def y(t):
		return (t + 1)**2 - 0.5 * np.exp(t)

	params = dict(
		t0 = 0.0,
		tf = 4.0,
		y0 = 0.5
	)
	return f, y, params


def compute_errors(f, y_exact_func, t0, y0, tf, h_values):
    """
    Compute RK4 errors at tf for different step sizes.

    Returns
    -------
    errors : list
        List of errors corresponding to each h
    """
    errors = []
    for h in h_values:
        t_rk, y_rk = rk4_solver(f, t0, y0, h, tf)
        y_exact_tf = y_exact_func(tf)
        err = abs(y_rk[-1] - y_exact_tf)
        errors.append(err)
    return errors


def estimate_order(h_values, errors):
    """
    Estimate order of accuracy p from consecutive errors.

    Parameters
    ----------
    h_values : list or ndarray
        Step sizes
    errors : list or ndarray
        Errors at corresponding step sizes

    Returns
    -------
    p_values : list
        Estimated order between consecutive step sizes
    """
    p_values = []
    for i in range(len(errors)-1):
        p = np.log(errors[i]/errors[i+1]) / np.log(h_values[i]/h_values[i+1])
        p_values.append(p)
    return p_values





