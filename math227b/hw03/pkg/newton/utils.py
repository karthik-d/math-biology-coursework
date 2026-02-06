from matplotlib import pyplot as plt
import numpy as np

from .newton import newton_system


def plot_errors(errors, title_suffix=""):
	"""
	Plot Newton convergence diagnostics:
		(1) Error history (semilog)
		(2) Log–log plot of e_{k+1} vs e_k with slope-2 reference
	"""

	errors = np.array(errors)
	iterations = np.arange(len(errors))

	# Prepare log-log data: e_{k+1} vs e_k
	e_k = errors[:-1]
	e_k1 = errors[1:]

	# --- Slope-2 reference line ---
	C = e_k1[0] / (e_k[0]**2)  # use first ratio as reference
	ref_x = np.array([e_k.min(), e_k.max()])
	ref_y = C * ref_x**2

	# --- Create figure ---
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	# ---------- Plot 1: Error history ----------
	axes[0].semilogy(iterations, errors, 'o-', label=r"$e_k = \|x_k - x^*\|$")
	axes[0].set_xlabel("Iteration $k$")
	axes[0].set_ylabel(r"Error $\|x_k - x^*\|$")
	axes[0].set_title(f"Semi-log Error Convergence: {title_suffix}")
	axes[0].grid(True, which="both")
	axes[0].legend(fontsize=9)

	# ---------- Plot 2: Log–Log Error Plot ----------
	axes[1].loglog(e_k, e_k1, 'o-', label=r"$(e_k, e_{k+1})$")
	axes[1].loglog(ref_x, ref_y, '--', label="Quadratic slope reference")
	axes[1].set_xlabel(r"$e_k$")
	axes[1].set_ylabel(r"$e_{k+1}$")
	axes[1].set_title(f"Log–Log Error vs. Previous Error: {title_suffix}")
	axes[1].grid(True, which="both")
	axes[1].legend(fontsize=9)

	plt.suptitle("Newton Convergence Diagnostics", fontsize=14, y=1.03)
	plt.tight_layout()
	plt.show()


def plot_residual_heatmap(F, J, x0, x_true=None, grid_bounds=(0, 2), grid_points=100, title="Residual Heatmap"):
	"""
	Visual demonstration of Newton's method: heatmap of ||F(x,y)|| over a 2D grid
	with trajectory from initial guess to final solution.

	Parameters
	----------
	F : callable
		Function F(x) returning array-like of shape (2,).
	J : callable
		Jacobian of F.
	x0 : array-like
		Initial guess for Newton's method.
	x_true : array-like, optional
		Ground-truth solution (for marking), default uses Newton final iterate.
	grid_bounds : tuple
		(min, max) bounds for x and y axes.
	grid_points : int
		Number of points per axis in the grid.
	title : str
		Figure title.
	"""
	# Run Newton and get trajectory
	x_sol, info = newton_system(F, x0, J)
	trajectory = info["trajectory"]
	print(info)

	# Convert trajectory to array for easy indexing
	traj = np.array(trajectory)

	# Grid for heatmap
	x_vals = np.linspace(grid_bounds[0], grid_bounds[1], grid_points)
	y_vals = np.linspace(grid_bounds[0], grid_bounds[1], grid_points)
	X, Y = np.meshgrid(x_vals, y_vals)
	Z = np.zeros_like(X)

	# Compute residuals
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Z[i,j] = np.linalg.norm(F([X[i,j], Y[i,j]]))

	# Plot heatmap
	plt.figure(figsize=(6,5))
	plt.contourf(X, Y, Z, levels=50, cmap='viridis')
	plt.colorbar(label=r"$\|F(x,y)\|$")

	# Plot trajectory
	if len(traj) > 1:
		plt.plot(traj[:-1,0], traj[:-1,1], 'w--', label='Trajectory', marker='o')  # dotted line for trajectory
	# final point.
	plt.scatter(traj[-1,0], traj[-1,1], color='red', s=50, label='Newton solution', zorder=5, alpha=0.4) 
	# mark initial guess
	plt.scatter(traj[0,0], traj[0,1], color='blue', s=40, label='Initial guess', zorder=5, alpha=0.4)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title(title)
	plt.legend()
	plt.grid(False)
	plt.show()

	return x_sol, info, traj


def plot_newton_convergence(F, J, x0, x_true=None, tol=1e-15, max_iter=50, title="Newton Convergence"):
    """
    Plot the residual and error history for a Newton method run.

    Parameters
    ----------
    F : callable
        Function F(x) returning array-like of shape (n,).
    J : callable
        Jacobian of F.
    x0 : array-like
        Initial guess.
    x_true : array-like, optional
        Ground-truth solution (for computing error history). Defaults to final iterate.
    tol : float
        Tolerance for Newton convergence.
    max_iter : int
        Maximum number of iterations.
    title : str
        Figure title.
    """
    x_sol, info = newton_system(F, x0, J, x_true=x_true, tol=tol, max_iter=max_iter)

    iterations = np.arange(len(info["residual_history"]))

    plt.figure(figsize=(6,5))
    plt.semilogy(iterations, info["residual_history"], 'o-', label=r"Residual ||F(x_k)||")
    plt.semilogy(iterations, info["error_history"], 's-', label=r"Error ||x_k - x*||")
    plt.xlabel("Iteration k")
    plt.ylabel("Value (log scale)")
    plt.title(title)
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    return x_sol, info
