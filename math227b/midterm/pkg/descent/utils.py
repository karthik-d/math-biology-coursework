import numpy as np
import matplotlib.pyplot as plt


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
	path = np.array(path)
	iterations = np.arange(len(path))

	# Step length (norm of move)
	step_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
	step_lengths = np.insert(step_lengths, 0, 0)  # insert 0 for first iteration

	# Gradient norms
	grad_norms = [np.linalg.norm(grad(x)) for x in path]

	# Distance to minimizer
	dists = [np.linalg.norm(x - x_star) for x in path]

	plt.figure(figsize=(8,6))

	plt.plot(iterations, step_lengths, marker='o', color='tab:blue', label=r'Step length $||x_{k+1}-x_k||$', alpha=0.7)
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
# 4. 2D contour with trajectory overlay
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
    plt.plot(path[:,0], path[:,1], marker='o', color='r', label="Iterations", alpha=0.4)
    plt.scatter(path[0,0], path[0,1], color='blue', marker='s', label="Start", s=50, zorder=5)
    plt.scatter(path[-1,0], path[-1,1], color='green', marker='*', label="End", s=50, zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()