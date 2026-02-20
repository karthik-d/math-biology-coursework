import numpy as np
import matplotlib.pyplot as plt


def check_hessian_pd(hess, x):
    H = hess(x)
    eigvals = np.linalg.eigvals(H)
    return H, eigvals


# ------------------------------------------------
# 1. Step length vs iteration
# ------------------------------------------------
def plot_step_length(path, title="Step length vs iteration"):
    path = np.array(path)
    # Approximate step length: norm of move
    step_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    plt.figure()
    plt.plot(np.arange(1, len(path)), step_lengths, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Step length (||x_{k+1}-x_k||)")
    plt.title(title)
    plt.grid(True)
    plt.show()


# ------------------------------------------------
# 2. Gradient norm vs iteration
# ------------------------------------------------
def plot_grad_norm_vs_iter(path, grad, title="Gradient norm vs iteration"):
    path = np.array(path)
    grad_norms = [np.linalg.norm(grad(x)) for x in path]
    plt.figure()
    plt.plot(np.arange(len(path)), grad_norms, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("||grad f(x_k)||")
    plt.title(title)
    plt.yscale('log')
    plt.grid(True)
    plt.show()


# ------------------------------------------------
# 3. Distance to minimizer vs iteration
# ------------------------------------------------
def plot_dist_to_min(path, x_star, title="Distance to minimizer vs iteration"):
    path = np.array(path)
    dists = [np.linalg.norm(x - x_star) for x in path]
    plt.figure()
    plt.plot(np.arange(len(path)), dists, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("||x_k - x*||")
    plt.title(title)
    plt.yscale('log')
    plt.grid(True)
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
    plt.plot(path[:,0], path[:,1], marker='o', color='r', label="Iterates")
    plt.scatter(path[0,0], path[0,1], color='blue', marker='s', label="Start")
    plt.scatter(path[-1,0], path[-1,1], color='green', marker='*', label="End")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()