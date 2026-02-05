import numpy as np
import matplotlib.pyplot as plt


import pkg.newton.test_functions as tf
from pkg.newton.newton import newton_system
from pkg.newton.utils import plot_errors


# --------------------------
# 1. Define the system and Jacobian
# --------------------------
def two_gene_network(params):
    """
    Returns F(x,y) and J(x,y) for Newton's method.
    params: dict with keys:
        alpha_min, alpha_max, alpha_deg
        beta_min, beta_max, beta_deg
        e_cx, e_cy, n
    """
    alpha_min = params['alpha_min']
    alpha_max = params['alpha_max']
    alpha_deg = params['alpha_deg']
    beta_min = params['beta_min']
    beta_max = params['beta_max']
    beta_deg = params['beta_deg']
    e_cx = params['e_cx']
    e_cy = params['e_cy']
    n = params['n']

    def F(v):
        x, y = v
        f1 = alpha_min + (alpha_max - alpha_min) * y**n / (e_cy**n + y**n) - alpha_deg * x
        f2 = beta_min + (beta_max - beta_min) * x**n / (e_cx**n + x**n) - beta_deg * y
        return np.array([f1, f2])

    def J(v):
        x, y = v
        df1_dx = -alpha_deg
        df1_dy = (alpha_max - alpha_min) * n * y**(n-1) * e_cy**n / (e_cy**n + y**n)**2
        df2_dx = (beta_max - beta_min) * n * x**(n-1) * e_cx**n / (e_cx**n + x**n)**2
        df2_dy = -beta_deg
        return np.array([[df1_dx, df1_dy],
                         [df2_dx, df2_dy]])

    return F, J


# --------------------------
# 2. Code testing / verification
# --------------------------
def test_newton_code():
    """
    Test Newton solver on known functions.
    """
    from pkg.newton import test_functions as tf
    test_list = [tf.q1c_f1(), tf.q1c_f2(), tf.q1c_f3()]
    for i, (F, J, x0, x_true) in enumerate(test_list, 1):
        sol, info = newton_system(F, x0, J, x_true)
        print(f"Test function {i}: solution = {sol}, converged = {info['converged']}")
        # Plot residuals for quadratic convergence
        plot_errors(info['error_history'])


# --------------------------
# 3. Compute example steady-state solutions in bistable regime
# --------------------------
def compute_bistable_solutions():
    """
    Finds and visualizes all three steady states (low stable, unstable, high stable)
    for a slightly asymmetric two-gene network. Overlays solutions on nullclines.
    Marks stability using Jacobian eigenvalues and colors initial guesses accordingly.
    Adds a vector field to show the flow and basins of attraction.
    """
    # Slightly asymmetric parameters
    params = {
        'alpha_min': 0.01,
        'alpha_max': 5.5,
        'alpha_deg': 1.0,
        'beta_min': 0.01,
        'beta_max': 4.5,
        'beta_deg': 0.9,
        'e_cx': 1.0,
        'e_cy': 1.5,
        'n': 4
    }

    F, J = two_gene_network(params)

    # Initial guesses for the three steady states (low, unstable, high)
    guesses = [
        np.array([0.5, 1.0]),   # low stable
        np.array([1.5, 1.8]),   # unstable
        np.array([3.5, 3.5])    # high stable
    ]

    solutions = []
    labels = ['Low', 'Unstable', 'High']
    colors = ['blue', 'orange', 'red']
    stability = []

    # Solve with Newton
    for x0, color, label in zip(guesses, colors, labels):
        sol, info = newton_system(F, x0, J)
        solutions.append(sol)
        print(f"Initial guess {x0} → solution {sol}, converged={info['converged']}")
        # plot_errors(info['error_history'])

        # Check stability via Jacobian eigenvalues
        J_sol = J(sol)
        eigvals = np.linalg.eigvals(J_sol)
        stable = np.all(np.real(eigvals) < 0)
        stability.append(stable)

    # Nullclines
    y_vals = np.linspace(-1, 7, 400)
    x_nullcline = (params['alpha_min'] + (params['alpha_max'] - params['alpha_min']) * y_vals**params['n'] /
                   (params['e_cy']**params['n'] + y_vals**params['n'])) / params['alpha_deg']

    x_vals = np.linspace(-1, 7, 400)
    y_nullcline = (params['beta_min'] + (params['beta_max'] - params['beta_min']) * x_vals**params['n'] /
                   (params['e_cx']**params['n'] + x_vals**params['n'])) / params['beta_deg']

    # --- Vector field / phase portrait ---
    X, Y = np.meshgrid(np.linspace(-1, 7, 25), np.linspace(-1, 7, 25))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            f_val = F([X[i,j], Y[i,j]])
            U[i,j] = f_val[0]
            V[i,j] = f_val[1]
    # Normalize arrows for better visualization
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-6)
    V_norm = V / (magnitude + 1e-6)

    # Plotting
    plt.figure(figsize=(8,8))

    # Vector field
    plt.quiver(X, Y, U_norm, V_norm, color='gray', alpha=0.5)

    # Nullclines
    plt.plot(x_nullcline, y_vals, 'b--', label='x-nullcline')
    plt.plot(x_vals, y_nullcline, 'r--', label='y-nullcline')

    # Overlay steady states with stability marker
    for sol, color, label, stable in zip(solutions, colors, labels, stability):
        marker = 'o' if stable else 's'  # circle=stable, square=unstable
        plt.scatter(sol[0], sol[1], c=color, s=120, edgecolors='k',
                    marker=marker, label=f'{label} steady state {"(stable)" if stable else "(unstable)"}')

    # Overlay initial guesses with same color, lighter, smaller
    for guess, color in zip(guesses, colors):
        plt.scatter(guess[0], guess[1], c=color, s=60, alpha=0.5, label='Initial guess')

    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Two-Gene Network: Converged Solutions on Phase Portrait')
    plt.grid(True)
    plt.xlim(-1, 7)
    plt.ylim(-1, 7)

    # Legend inside top-left
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()



# --------------------------
# 4. Parameter sweep for n and alpha_max
# --------------------------
def sweep_n():
    """
    Sweep Hill coefficient n for the asymmetric two-gene network.
    Overlay steady states from Newton on analytical nullcline predictions.
    """
    # --- Fixed parameters ---
    alpha_max_fixed = 5.5
    beta_max_fixed = 4.0
    n_ref = 4
    params_ref = {
        'alpha_min': 0.01, 'alpha_max': alpha_max_fixed, 'alpha_deg': 1.0,
        'beta_min': 0.01, 'beta_max': beta_max_fixed, 'beta_deg': 0.8,
        'e_cx': 1.0, 'e_cy': 2.0,
        'n': n_ref
    }

    n_values = np.linspace(1, 6, 50)
    x_low, y_low, x_high, y_high = [], [], [], []

    for n in n_values:
        params = params_ref.copy()
        params['n'] = n
        F, J = two_gene_network(params)
        sol_low, _ = newton_system(F, np.array([0.5,1.0]), J)
        sol_high, _ = newton_system(F, np.array([4.0,4.0]), J)
        x_low.append(sol_low[0]); y_low.append(sol_low[1])
        x_high.append(sol_high[0]); y_high.append(sol_high[1])

    plt.figure(figsize=(7,5))
    plt.scatter(n_values, x_low, c='blue', s=30, label='x low branch')
    plt.scatter(n_values, y_low, c='green', s=30, label='y low branch')
    plt.scatter(n_values, x_high, c='blue', s=30, marker='s', label='x high branch')
    plt.scatter(n_values, y_high, c='green', s=30, marker='s', label='y high branch')

    # Analytical nullclines at reference n
    y_vals = np.linspace(-1, 7, 400)
    x_nullcline = (params_ref['alpha_min'] + (params_ref['alpha_max'] - params_ref['alpha_min']) *
                   y_vals**n_ref / (params_ref['e_cy']**n_ref + y_vals**n_ref)) / params_ref['alpha_deg']
    plt.plot(x_nullcline, y_vals, 'b--', alpha=0.5, label='x-nullcline (analytical)')
    x_vals = np.linspace(-1, 7, 400)
    y_nullcline = (params_ref['beta_min'] + (params_ref['beta_max'] - params_ref['beta_min']) *
                   x_vals**n_ref / (params_ref['e_cx']**n_ref + x_vals**n_ref)) / params_ref['beta_deg']
    plt.plot(x_vals, y_nullcline, 'g--', alpha=0.5, label='y-nullcline (analytical)')

    plt.xlabel('Hill coefficient n')
    plt.ylabel('Steady state value')
    plt.title('Steady states vs Hill coefficient n')
    plt.grid(True)
    plt.legend()
    plt.show()


def sweep_alpha_max():
    """
    Sweep activation strength alpha_max for the asymmetric two-gene network.
    Overlay steady states from Newton on analytical nullcline predictions.
    """
    # --- Fixed parameters ---
    alpha_max_ref = 5.5
    beta_max_fixed = 4.0
    n_fixed = 4
    params_ref = {
        'alpha_min': 0.01, 'alpha_max': alpha_max_ref, 'alpha_deg': 1.0,
        'beta_min': 0.01, 'beta_max': beta_max_fixed, 'beta_deg': 0.8,
        'e_cx': 1.0, 'e_cy': 2.0,
        'n': n_fixed
    }

    alpha_values = np.linspace(0.5, 6, 50)
    x_low, y_low, x_high, y_high = [], [], [], []

    for alpha_max in alpha_values:
        params = params_ref.copy()
        params['alpha_max'] = alpha_max
        F, J = two_gene_network(params)
        sol_low, _ = newton_system(F, np.array([0.5,1.0]), J)
        sol_high, _ = newton_system(F, np.array([4.0,4.0]), J)
        x_low.append(sol_low[0]); y_low.append(sol_low[1])
        x_high.append(sol_high[0]); y_high.append(sol_high[1])

    plt.figure(figsize=(7,5))
    plt.scatter(alpha_values, x_low, c='blue', s=30, label='x low branch')
    plt.scatter(alpha_values, y_low, c='green', s=30, label='y low branch')
    plt.scatter(alpha_values, x_high, c='blue', s=30, marker='s', label='x high branch')
    plt.scatter(alpha_values, y_high, c='green', s=30, marker='s', label='y high branch')

    # Analytical nullclines at reference alpha_max
    y_vals = np.linspace(-1, 7, 400)
    x_nullcline = (params_ref['alpha_min'] + (params_ref['alpha_max'] - params_ref['alpha_min']) *
                   y_vals**n_fixed / (params_ref['e_cy']**n_fixed + y_vals**n_fixed)) / params_ref['alpha_deg']
    plt.plot(x_nullcline, y_vals, 'b--', alpha=0.5, label='x-nullcline (analytical)')
    x_vals = np.linspace(-1, 7, 400)
    y_nullcline = (params_ref['beta_min'] + (params_ref['beta_max'] - params_ref['beta_min']) *
                   x_vals**n_fixed / (params_ref['e_cx']**n_fixed + x_vals**n_fixed)) / params_ref['beta_deg']
    plt.plot(x_vals, y_nullcline, 'g--', alpha=0.5, label='y-nullcline (analytical)')

    plt.xlabel('Activation strength alpha_max')
    plt.ylabel('Steady state value')
    plt.title('Steady states vs alpha_max')
    plt.grid(True)
    plt.legend()
    plt.show()


def sweep_ecx():
    """
    Sweep cross-activation threshold e_cx for the asymmetric two-gene network.
    Overlay steady states from Newton on analytical nullcline predictions.
    """
    # --- Fixed parameters ---
    alpha_max_fixed = 5.5
    beta_max_fixed = 4.0
    n_fixed = 4
    params_ref = {
        'alpha_min': 0.01, 'alpha_max': alpha_max_fixed, 'alpha_deg': 1.0,
        'beta_min': 0.01, 'beta_max': beta_max_fixed, 'beta_deg': 0.8,
        'e_cx': 1.0, 'e_cy': 2.0,
        'n': n_fixed
    }

    e_cx_values = np.linspace(0.5, 3, 50)
    x_low, y_low, x_high, y_high = [], [], [], []

    for e_cx in e_cx_values:
        params = params_ref.copy()
        params['e_cx'] = e_cx
        F, J = two_gene_network(params)
        sol_low, _ = newton_system(F, np.array([0.5,1.0]), J)
        sol_high, _ = newton_system(F, np.array([4.0,4.0]), J)
        x_low.append(sol_low[0]); y_low.append(sol_low[1])
        x_high.append(sol_high[0]); y_high.append(sol_high[1])

    plt.figure(figsize=(7,5))
    plt.scatter(e_cx_values, x_low, c='blue', s=30, label='x low branch')
    plt.scatter(e_cx_values, y_low, c='green', s=30, label='y low branch')
    plt.scatter(e_cx_values, x_high, c='blue', s=30, marker='s', label='x high branch')
    plt.scatter(e_cx_values, y_high, c='green', s=30, marker='s', label='y high branch')

    # Analytical nullclines at reference e_cx
    y_vals = np.linspace(-1, 7, 400)
    x_nullcline = (params_ref['alpha_min'] + (params_ref['alpha_max'] - params_ref['alpha_min']) *
                   y_vals**n_fixed / (params_ref['e_cy']**n_fixed + y_vals**n_fixed)) / params_ref['alpha_deg']
    plt.plot(x_nullcline, y_vals, 'b--', alpha=0.5, label='x-nullcline (analytical)')
    x_vals = np.linspace(-1, 7, 400)
    y_nullcline = (params_ref['beta_min'] + (params_ref['beta_max'] - params_ref['beta_min']) *
                   x_vals**n_fixed / (params_ref['e_cx']**n_fixed + x_vals**n_fixed)) / params_ref['beta_deg']
    plt.plot(x_vals, y_nullcline, 'g--', alpha=0.5, label='y-nullcline (analytical)')

    plt.xlabel('Cross-activation threshold e_cx')
    plt.ylabel('Steady state value')
    plt.title('Steady states vs e_cx')
    plt.grid(True)
    plt.legend()
    plt.show()


# --------------------------
# 5. Newton-based basin-of-attraction
# --------------------------
def basin_of_attraction_grid(param_set, x_range=(0,5), y_range=(0,5), grid_size=50):
    """
    2D grid of initial guesses, colored by which steady state Newton converges to.
    """
    F, J = two_gene_network(param_set)
    X = np.linspace(*x_range, grid_size)
    Y = np.linspace(*y_range, grid_size)
    basin = np.zeros((grid_size, grid_size))

    for i, x0 in enumerate(X):
        for j, y0 in enumerate(Y):
            sol, info = newton_system(F, np.array([x0, y0]), J)
            # classify solution: closer to low or high branch
            # use thresholds at midpoint of expected low/high states
            low_guess = np.array([0.01,0.01])
            high_guess = np.array([5.0,5.0])
            if np.linalg.norm(sol - low_guess) < np.linalg.norm(sol - high_guess):
                basin[j, i] = 0  # low branch
            else:
                basin[j, i] = 1  # high branch

    plt.figure()
    plt.imshow(basin, origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
               cmap='coolwarm', alpha=0.8)
    plt.colorbar(label='Steady state branch (0=low, 1=high)')
    plt.xlabel('Initial x')
    plt.ylabel('Initial y')
    plt.title('Basin of attraction (Newton convergence)')
    plt.show()


# --------------------------
# 6. Main analysis workflow
# --------------------------
if __name__ == "__main__":
    # print("=== Step 1: Test Newton solver on known functions ===")
    # test_newton_code()

    print("=== Step 2: Compute bistable steady states ===")
    # compute_bistable_solutions()

    print("=== Step 3: Parameter sweeps ===")
    sweep_alpha_max()
    sweep_n()
    sweep_ecx()

    print("=== Step 4: Basin of attraction ===")
    params_bistable = {
        'alpha_min': 0.01, 'alpha_max': 5.0, 'alpha_deg': 1.0,
        'beta_min': 0.01, 'beta_max': 5.0, 'beta_deg': 1.0,
        'e_cx': 1.0, 'e_cy': 1.0, 'n': 3
    }
    # basin_of_attraction_grid(params_bistable)
