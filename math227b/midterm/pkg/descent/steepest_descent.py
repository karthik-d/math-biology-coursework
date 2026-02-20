import numpy as np


def backtracking_line_search(f, grad, x, p, alpha0=1.0, rho=0.5, c=1e-4):
    alpha = alpha0
    fx = f(x)
    gTp = grad(x).dot(p)

    while f(x + alpha*p) > fx + c*alpha*gTp:
        alpha *= rho
    return alpha


def steepest_descent(f, grad, x0, tol=1e-6, maxiter=int(1e7)):
	x = np.array(x0, dtype=float)
	history = [x.copy()]

	for k in range(maxiter):
		g = grad(x)
		norm_g = np.linalg.norm(g)

		if norm_g < tol:
			print(f"Converged in {k} iterations.")
			print(x)
			return x, np.array(history)

		p = -g  # steepest descent direction
		alpha = backtracking_line_search(f, grad, x, p)
		x = x + alpha*p
		history.append(x.copy())

	print("Maximum iterations reached.")
	print(x)
	return x, np.array(history)
