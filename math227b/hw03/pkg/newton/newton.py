import numpy as np


def newton_system(F, x0, J=None, tol=1e-16, max_iter=50):
	"""
	Solve F(x) = 0 using Newton's method for systems.

	Parameters
	----------
	F : callable
		Function F(x) returning array-like of shape (n,).
	x0 : array-like
		Initial guess (n,).
	J : callable, optional
		Function J(x) returning Jacobian matrix (n, n).
		If None, finite-difference Jacobian is used.
	tol : float
		Convergence tolerance on ||F(x)||.
	max_iter : int
		Maximum number of iterations.

	Returns
	-------
	x : ndarray
		Approximate solution.
	info : dict
		Dictionary with convergence info.
	"""

	x = np.asarray(x0, dtype=float)

	def finite_diff_jacobian(F, x, eps=1e-16):
		n = len(x)
		J = np.zeros((n, n))
		Fx = F(x)
		for j in range(n):
			dx = np.zeros(n)
			dx[j] = eps
			J[:, j] = (F(x + dx) - Fx) / eps
		return J

	residuals_l = []
	for k in range(max_iter):
		Fx = np.asarray(F(x), dtype=float)
		normF = np.linalg.norm(Fx)
		residuals_l.append(normF)

		if normF < tol:
			return x, {"converged": True, "iterations": k, "residual": normF, "residual_history": residuals_l}

		if J is None:
			Jx = finite_diff_jacobian(F, x)
		else:
			Jx = np.asarray(J(x), dtype=float)

		try:
			delta = np.linalg.solve(Jx, -Fx)
		except np.linalg.LinAlgError:
			return x, {"converged": False, "reason": "Jacobian singular"}

		x = x + delta

	return x, {"converged": False, "iterations": max_iter, "residual": np.linalg.norm(F(x)), "residual_history": residuals_l}
