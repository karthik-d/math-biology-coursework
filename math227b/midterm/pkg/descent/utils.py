import numpy as np
import matplotlib.pyplot as plt

from .steepest_descent import steepest_descent


def check_hessian_pd(hess, x):
    H = hess(x)
    eigvals = np.linalg.eigvals(H)
    return H, eigvals


def plot_iterations_summary(path, grad, x_star, title="Iteration Summary"):
	"""
	Plots step length, gradient norm, and distance to minimizer in one figure.

	Parameters:
	-----------
	path : array-like
		Array of iterates, shape (num_iter, n)
	grad : function
		Gradient function, grad(x)
	x_star : array-like
		True minimizer
	title : str
		Figure title
	"""
	path = np.array(path[::10])
	iterations = np.arange(len(path))

	# Step length (norm of move)
	step_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
	step_lengths = np.insert(step_lengths, 0, 0)  # insert 0 for first iteration

	# Gradient norms
	grad_norms = [np.linalg.norm(grad(x)) for x in path]

	# Distance to minimizer
	dists = [np.linalg.norm(x - x_star) for x in path]

	plt.figure(figsize=(8,6))

	plt.plot(iterations, step_lengths, marker='.', color='tab:blue', label=r'Step length $||x_{k+1}-x_k||$', alpha=0.7)
	plt.plot(iterations, grad_norms, marker='s', color='tab:orange', label=r'Gradient norm $||grad f(x_k)||$', alpha=0.7)
	plt.plot(iterations, dists, marker='^', color='tab:green', label='Distance to x*', alpha=0.7)

	plt.xlabel("Iteration")
	plt.ylabel("Value (log scale)")
	plt.yscale('log')
	plt.title(title)
	plt.grid(True, which="both", ls="--", lw=0.5)
	plt.legend()
	plt.tight_layout()
	plt.show()


# ------------------------------------------------
# 2. 2D contour with trajectory overlay
# ------------------------------------------------
def plot_trajectory_contour(f, path, xlim=(-1,5), ylim=(-3,3), title="Trajectory over contour"):
    path = np.array(path)
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = f([X[i,j], Y[i,j]])

    plt.figure()
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.plot(path[:,0], path[:,1], marker='o', color='r', label="Iterations", alpha=0.4, markersize=1)
    plt.scatter(path[0,0], path[0,1], color='blue', marker='s', label="Start", s=50, zorder=5)
    plt.scatter(path[-1,0], path[-1,1], color='green', marker='*', label="End", s=50, zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_basin(f, grad, x_range=(-2, 2), y_range=(-1, 3),
	grid_size=50, tol=1e-4, max_iter=int(1e7), representative_points=None):
    """
    Plots a basin-of-attraction heat map for the Rosenbrock function, colored
    by number of iterations to converge. Points that do not converge are shown
    in a separate color. Representative paths are overlaid.

    Parameters
    ----------
    f : function
        Scalar function f(x)
    grad : function
        Gradient function grad(x)
    x_range : tuple
        x-axis limits (min, max)
    y_range : tuple
        y-axis limits (min, max)
    grid_size : int
        Number of initial points along each axis
    max_iter : int
        Maximum iterations for steepest descent
    tol : float
        Convergence tolerance
    representative_points : list of np.ndarray, optional
        Initial points for which to plot paths
    """

    x_vals = np.linspace(x_range[0], x_range[1], grid_size)
    y_vals = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    iter_map = np.full_like(X, fill_value=max_iter, dtype=float)

    x_star = np.array([1.0, 1.0])

    # Compute iteration count for each starting point
    for i in range(grid_size):
        print(i, grid_size)
        for j in range(grid_size):
            x0 = np.array([X[i,j], Y[i,j]])
            try:
                xmin, path = steepest_descent(f, grad, x0, maxiter=max_iter)
                dist = np.linalg.norm(xmin - x_star)
                if dist < tol:
                    iter_map[i,j] = len(path)
                else:
                    iter_map[i,j] = max_iter  # failed to converge
            except Exception:
                iter_map[i,j] = max_iter  # treat exceptions as failure

    # Evaluate function for contours
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i,j] = f([X[i,j], Y[i,j]])

    # Plot heat map of iterations
    plt.figure(figsize=(8,6))
    cmap = plt.cm.viridis
    # Set failed points to NaN so they can be colored differently
    iter_plot = np.copy(iter_map)
    iter_plot[iter_plot >= max_iter] = np.nan
    heat = plt.imshow(iter_plot, extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                      origin='lower', alpha=0.7, cmap=cmap, aspect='auto')

    # Colorbar for iterations
    cbar = plt.colorbar(heat)
    cbar.set_label("Iterations to converge")

    # Overlay function contours
    CS = plt.contour(X, Y, Z, levels=20, cmap='magma', alpha=0.6)

    # Overlay non-converged points in red
    failed = np.where(iter_map >= max_iter)
    plt.scatter(X[failed], Y[failed], color='red', s=20, marker='x', label="Did not converge")

    # Overlay representative paths
    if representative_points is not None:
        for x0 in representative_points:
            try:
                _, path = steepest_descent(f, grad, x0, tol=tol, maxiter=max_iter)
                path = np.array(path)
                plt.plot(path[:,0], path[:,1], marker='o', markersize=3,
                         label=f"Path from {x0}", alpha=0.8)
            except Exception:
                continue

    # Mark global minimum
    plt.scatter(x_star[0], x_star[1], color='green', marker='*', s=150, label="Global minimum")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rosenbrock Basin of Attraction (Iterations to Converge)")
    plt.legend()
    plt.grid(True)
    plt.show()