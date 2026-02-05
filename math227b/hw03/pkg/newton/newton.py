import numpy as np

def newton_system(F, x0, J=None, x_true=None, tol=1e-15, max_iter=50):
    """
    Solve F(x) = 0 using Newton's method for systems.

    Also records:
      - function residuals ||F(x_k)||
      - error residuals ||x_k - x*|| for quadratic convergence study

    Parameters
    ----------
    F : callable
        Function F(x) returning array-like of shape (n,).
    x0 : array-like
        Initial guess (n,).
    J : callable, optional
        Function J(x) returning Jacobian matrix (n, n).
        If None, finite-difference Jacobian is used.
    x_true : array-like, optional
        Ground-truth solution x*. If None, final iterate is used.
    tol : float
        Convergence tolerance on ||F(x)||.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x : ndarray
        Approximate solution.
    info : dict
        Dictionary with convergence info and histories.
    """

    x = np.asarray(x0, dtype=float)

    def finite_diff_jacobian(F, x, eps=1e-15):
        n = len(x)
        J = np.zeros((n, n))
        Fx = F(x)
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = eps
            J[:, j] = (F(x + dx) - Fx) / eps
        return J

    residuals_l = []     # ||F(x_k)||
    iterates = []        # store x_k for later error computation

    for k in range(max_iter):
        Fx = np.asarray(F(x), dtype=float)
        normF = np.linalg.norm(Fx)

        residuals_l.append(normF)
        iterates.append(x.copy())

        if normF < tol:
            break

        if J is None:
            Jx = finite_diff_jacobian(F, x)
        else:
            Jx = np.asarray(J(x), dtype=float)

        try:
            delta = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            return x, {
                "converged": False,
                "reason": "Jacobian singular",
                "residual_history": residuals_l
            }

        x = x + delta

    # Decide reference solution x*
    if x_true is None:
        x_star = iterates[-1]
    else:
        x_star = np.asarray(x_true, dtype=float)

    # Compute error history ||x_k - x*||
    error_history = [np.linalg.norm(xk - x_star) for xk in iterates]

    return x, {
        "converged": residuals_l[-1] < tol,
        "iterations": len(residuals_l),
        "residual": residuals_l[-1],
        "residual_history": residuals_l,
        "error_history": error_history
    }
