def divided_difference(x, fvals):
	"""
	compute the divided difference f[x0, x1, ..., xn].

	Takes:
	x : list or array of floats
		Points x0, x1, ..., xn
	fvals : list or array of floats
		Function values f(x0), f(x1), ..., f(xn)

	Returns:
	float
		The divided difference f[x0, x1, ..., xn]
	"""

	n = len(x)
	dd = fvals.copy() 

	for k in range(1, n):
		for i in range(n - k):
			dd[i] = (dd[i + 1] - dd[i]) / (x[i + k] - x[i])

	return dd[0]


def compute_polynomial(a, x_nodes, x):
    """
    evaluate Newton-form polynomial using nested multiplication.

    Takes:
    a : list or array of floats
        Coefficients a1, a2, ..., a_{n+1}
    x_nodes : list or array of floats
        Nodes x1, x2, ..., xn
    x : float
        Point where polynomial is evaluated

    Returns:
    float
        P(x)
    """

    n = len(a)
    value = a[-1]

    for k in range(n - 2, -1, -1):
        value = value * (x - x_nodes[k]) + a[k]

    return value


def interpolate(x_nodes, f, x):
	"""
	Lagrange interpolation at x.

	Takes:
	x_nodes : list of floats
		Interpolation nodes
	f : callable OR list of floats
		Function f(x) or precomputed values f(x_i)
	x : float
		Evaluation point

	Returns:
	float
		Interpolated value P(x)
	"""
	# Allow either f(x) or precomputed values
	if callable(f):
		fvals = [f(xi) for xi in x_nodes]
	else:
		fvals = f

	a = divided_difference(x_nodes, fvals)
	return compute_polynomial(a, x_nodes, x)