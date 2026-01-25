def divided_difference(x, fvals):
    """
    Compute divided difference table (2D).

    Returns
    -------
    table : list of lists
        table[i][j] = f[x_i, ..., x_{i+j}]
    """
    n = len(x)
    table = [[0.0]*n for _ in range(n)]

    # first column: f[x_i]
    for i in range(n):
        table[i][0] = fvals[i]

    # build table
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])

    return table[0]


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