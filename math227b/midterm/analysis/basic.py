import numpy as np	


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