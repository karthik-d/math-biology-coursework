import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import analysis.basic as tf
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
def test_newton_convergence():
    """
    Test Newton solver on known functions.
    """
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
        'alpha_min': 0.1,
        'alpha_max': 5.5,
        'alpha_deg': 1.0,
        'beta_min': 0.1,
        'beta_max': 4.5,
        'beta_deg': 0.9,
        'e_cx': 1.0,
        'e_cy': 1.5,
        'n': 4
    }

    F, J = two_gene_network(params)

    # Initial guesses for the three steady states (low, unstable, high)
    guesses = [
        np.array([-0.2, -0.2]),		# low stable
        np.array([1, 1]),   		# unstable
        np.array([3.5, 3.5]) 		# high stable
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
        plot_errors(info['error_history'], title_suffix=f'{label} State')

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
# -------------------------
def sweep_alpha_max():
    """
    Sweep activation strength alpha_max for the asymmetric two-gene network.
    Plot Newton-inferred and analytical steady states and phase portraits
    with detailed legends similar to the reference style.
    """

    # --- Fixed parameters ---
    beta_max_fixed = 4.0
    n_fixed = 4
    params_ref = {
        'alpha_min': 0.1, 'alpha_max': 5.5, 'alpha_deg': 1.0,
        'beta_min': 0.1, 'beta_max': beta_max_fixed, 'beta_deg': 0.8,
        'e_cx': 1.0, 'e_cy': 2.0,
        'n': n_fixed
    }

    alpha_values = np.linspace(1.5, 6, 25)

    x_low, y_low = [], []
    x_unst, y_unst = [], []
    x_high, y_high = [], []

    # --- Newton steady states ---
    for alpha_max in alpha_values:
        params = params_ref.copy()
        params['alpha_max'] = alpha_max
        F, J = two_gene_network(params)
        sol_low, _ = newton_system(F, [0.1,0.1], J)
        sol_unst, _ = newton_system(F, [1,2.3], J)
        sol_high, _ = newton_system(F, [4,4], J)
        x_low.append(sol_low[0]); y_low.append(sol_low[1])
        x_unst.append(sol_unst[0]); y_unst.append(sol_unst[1])
        x_high.append(sol_high[0]); y_high.append(sol_high[1])

    # --- Analytical steady states ---
    x_low_ref, y_low_ref = [], []
    x_unst_ref, y_unst_ref = [], []
    x_high_ref, y_high_ref = [], []

    for alpha_max in alpha_values:
        params_ref['alpha_max'] = alpha_max
        n = n_fixed

        def x_null(y):
            return (params_ref['alpha_min'] + (params_ref['alpha_max'] - params_ref['alpha_min']) *
                    y**n / (params_ref['e_cy']**n + y**n)) / params_ref['alpha_deg']

        def y_null(x):
            return (params_ref['beta_min'] + (params_ref['beta_max'] - params_ref['beta_min']) *
                    x**n / (params_ref['e_cx']**n + x**n)) / params_ref['beta_deg']

        sol_low  = fsolve(lambda v:[v[0]-x_null(v[1]),v[1]-y_null(v[0])],[0.1,0.1])
        sol_unst = fsolve(lambda v:[v[0]-x_null(v[1]),v[1]-y_null(v[0])],[1,2.3])
        sol_high = fsolve(lambda v:[v[0]-x_null(v[1]),v[1]-y_null(v[0])],[4,4])

        x_low_ref.append(sol_low[0]); y_low_ref.append(sol_low[1])
        x_unst_ref.append(sol_unst[0]); y_unst_ref.append(sol_unst[1])
        x_high_ref.append(sol_high[0]); y_high_ref.append(sol_high[1])

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    ax1, ax2 = axes

    # --- SUBPLOT 1: alpha_max vs steady states ---
    ax1.scatter(alpha_values, x_low,  c='blue',   s=20, marker='s', label='x low (Newton)')
    ax1.scatter(alpha_values, y_low,  c='blue',   s=20, label='y low (Newton)')
    ax1.scatter(alpha_values, x_unst, c='orange', s=20, marker='s', label='x unstable (Newton)')
    ax1.scatter(alpha_values, y_unst, c='orange', s=20, label='y unstable (Newton)')
    ax1.scatter(alpha_values, x_high, c='green',  s=20, marker='s', label='x high (Newton)')
    ax1.scatter(alpha_values, y_high, c='green',  s=20, label='y high (Newton)')

    ax1.plot(alpha_values, x_low_ref,  'b-', alpha=0.7, label='x low (analytical)')
    ax1.plot(alpha_values, x_unst_ref, 'orange', alpha=0.7, label='x unstable (analytical)')
    ax1.plot(alpha_values, x_high_ref, 'g-', alpha=0.7, label='x high (analytical)')
    ax1.plot(alpha_values, y_low_ref,  'b--', alpha=0.7, label='y low (analytical)')
    ax1.plot(alpha_values, y_unst_ref, 'orange', ls='--', alpha=0.7, label='y unstable (analytical)')
    ax1.plot(alpha_values, y_high_ref, 'g--', alpha=0.7, label='y high (analytical)')

    ax1.set_xlabel('alpha_max'); ax1.set_ylabel('Steady state value')
    ax1.set_title('Steady states vs alpha_max')
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # --- SUBPLOT 2: phase portrait ---
    selected_indices = [0, len(alpha_values)//2, -1]
    phase_colors = ['purple', 'teal', 'darkred']

    line_handles = []

    for idx, c in zip(selected_indices, phase_colors):
        alpha_max = alpha_values[idx]
        params = params_ref.copy(); params['alpha_max'] = alpha_max
        n = n_fixed

        y_vals = np.linspace(-0.02,7,200)
        x_null = (params['alpha_min'] + (params['alpha_max'] - params['alpha_min']) * y_vals**n /
                  (params['e_cy']**n + y_vals**n)) / params['alpha_deg']
        x_vals = np.linspace(-0.02,7,200)
        y_null = (params['beta_min'] + (params['beta_max'] - params['beta_min']) * x_vals**n /
                  (params['e_cx']**n + x_vals**n)) / params['beta_deg']

        l1, = ax2.plot(x_null, y_vals, c=c, lw=1.5, label=f'x-null alpha_max={alpha_max:.2f}', alpha=0.8)
        l2, = ax2.plot(x_vals, y_null, c=c, lw=1.5, ls='--', label=f'y-null alpha_max={alpha_max:.2f}', alpha=0.8)
        line_handles.extend([l1, l2])

        ax2.scatter(x_low[idx], y_low[idx], c=c, s=40, marker='o')
        ax2.scatter(x_unst[idx], y_unst[idx], c=c, s=40, marker='x')
        ax2.scatter(x_high[idx], y_high[idx], c=c, s=40, marker='s')

    # Marker legend proxies
    low_proxy = plt.Line2D([0],[0], marker='o', color='k', linestyle='None', label='Low steady state')
    unst_proxy = plt.Line2D([0],[0], marker='x', color='k', linestyle='None', label='Unstable steady state')
    high_proxy = plt.Line2D([0],[0], marker='s', color='k', linestyle='None', label='High steady state')

    ax2.legend(handles=line_handles + [low_proxy, unst_proxy, high_proxy], fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.2))
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_title('Phase portrait (x vs y) for selected alpha_max')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
     

def sweep_n():
	"""
	Sweep Hill coefficient n for the asymmetric two-gene network.
	Plot Newton-inferred steady states and analytical steady states (low, unstable, high),
	along with phase portraits for selected n values.
	"""
	# --- Fixed parameters ---
	params_ref = {
		'alpha_min': 0.1, 'alpha_max': 5.5, 'alpha_deg': 1.0,
		'beta_min': 0.1, 'beta_max': 4.0, 'beta_deg': 0.8,
		'e_cx': 1.0, 'e_cy': 2.0,
		'n': 4
	}

	n_values = np.linspace(1, 10, 10)  # sweep n
	x_low, y_low = [], []
	x_unst, y_unst = [], []
	x_high, y_high = [], []

	# --- Newton-inferred steady states ---
	for n in n_values:
		params = params_ref.copy()
		params['n'] = n
		
		F, J = two_gene_network(params)
		sol_high, _ = newton_system(F, [4.0, 4.0], J)

		if n < 3:
			x_high.append(sol_high[0]); y_high.append(sol_high[1])
			x_unst.append(np.nan); y_unst.append(np.nan)
			x_low.append(np.nan); y_low.append(np.nan)
		else:
			if n < 5:
				mid_guess = [1.2, 1.0]
			else:
				mid_guess = [1, 1.5]

			sol_unst, _ = newton_system(F, mid_guess, J)
			sol_low, _ = newton_system(F, [0.1, 0.1], J)

			x_low.append(sol_low[0]); y_low.append(sol_low[1])
			x_unst.append(sol_unst[0]); y_unst.append(sol_unst[1])
			x_high.append(sol_high[0]); y_high.append(sol_high[1])

	# --- Analytical steady states ---
	x_low_ref, y_low_ref = [], []
	x_unst_ref, y_unst_ref = [], []
	x_high_ref, y_high_ref = [], []

	for n in n_values:
		params = params_ref.copy()
		params['n'] = n

		def x_null(y):
			return (params['alpha_min'] + (params['alpha_max'] - params['alpha_min']) *
					y**n / (params['e_cy']**n + y**n)) / params['alpha_deg']

		def y_null(x):
			return (params['beta_min'] + (params['beta_max'] - params['beta_min']) *
					x**n / (params['e_cx']**n + x**n)) / params['beta_deg']

		sol_high = fsolve(lambda v: [v[0]-x_null(v[1]), v[1]-y_null(v[0])], [4.0, 4.0])
		x_high_ref.append(sol_high[0]); y_high_ref.append(sol_high[1])

		if n < 3:
			x_unst_ref.append(np.nan); y_unst_ref.append(np.nan)
			x_low_ref.append(np.nan); y_low_ref.append(np.nan)
		else:
			if n < 5:
				mid_guess = [1.2, 1.0]
			else:
				mid_guess = [1, 1.5]

			sol_unst = fsolve(lambda v: [v[0]-x_null(v[1]), v[1]-y_null(v[0])], mid_guess)
			sol_low = fsolve(lambda v: [v[0]-x_null(v[1]), v[1]-y_null(v[0])], [0.1, 0.1])

			x_unst_ref.append(sol_unst[0]); y_unst_ref.append(sol_unst[1])
			x_low_ref.append(sol_low[0]); y_low_ref.append(sol_low[1])

		# --- Plots ---
	fig, axes = plt.subplots(1, 2, figsize=(14,6))

	# ---------- SUBPLOT 1 ----------
	ax1 = axes[0]
	ax1.scatter(n_values, x_low, c='blue', s=20, marker='s', label='x low (Newton)')
	ax1.scatter(n_values, y_low, c='blue', s=20, label='y low (Newton)')
	ax1.scatter(n_values, x_unst, c='orange', s=20, marker='s', label='x unstable (Newton)')
	ax1.scatter(n_values, y_unst, c='orange', s=20, label='y unstable (Newton)')
	ax1.scatter(n_values, x_high, c='green', s=20, marker='s', label='x high (Newton)')
	ax1.scatter(n_values, y_high, c='green', s=20, label='y high (Newton)')

	ax1.plot(n_values, x_low_ref, 'b-', alpha=0.7, label='x low (analytical)')
	ax1.plot(n_values, x_unst_ref, 'orange', alpha=0.7, label='x unstable (analytical)')
	ax1.plot(n_values, x_high_ref, 'g-', alpha=0.7, label='x high (analytical)')
	ax1.plot(n_values, y_low_ref, 'b--', alpha=0.7, label='y low (analytical)')
	ax1.plot(n_values, y_unst_ref, 'orange', ls='--', alpha=0.7, label='y unstable (analytical)')
	ax1.plot(n_values, y_high_ref, 'g--', alpha=0.7, label='y high (analytical)')

	ax1.set_xlabel('Hill coefficient n')
	ax1.set_ylabel('Steady state value')
	ax1.set_title('Steady states vs n')
	ax1.grid(True)
	ax1.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.0, 0.6))

	# ---------- SUBPLOT 2 ----------
	ax2 = axes[1]
	selected_indices = [0, len(n_values)//2, -1]

	# Independent color scheme for subplot 2
	phase_colors = ['purple', 'teal', 'darkred']

	line_handles = []

	for idx, c in zip(selected_indices, phase_colors):
		n = n_values[idx]
		params = params_ref.copy(); params['n'] = n

		y_vals = np.linspace(-0.02,7,200)
		x_null = (params['alpha_min'] + (params['alpha_max'] - params['alpha_min']) * y_vals**n /
					(params['e_cy']**n + y_vals**n)) / params['alpha_deg']
		x_vals = np.linspace(-0.02,7,200)
		y_null = (params['beta_min'] + (params['beta_max'] - params['beta_min']) * x_vals**n /
					(params['e_cx']**n + x_vals**n)) / params['beta_deg']

		l1, = ax2.plot(x_null, y_vals, c=c, lw=1.5, label=f'x-null n={n:.2f}', alpha=0.8)
		l2, = ax2.plot(x_vals, y_null, c=c, lw=1.5, ls='--', label=f'y-null n={n:.2f}', alpha=0.8)
		line_handles.extend([l1, l2])

		ax2.scatter(x_low[idx], y_low[idx], c=c, s=40, marker='o')
		if not np.isnan(x_unst[idx]):
			ax2.scatter(x_unst[idx], y_unst[idx], c=c, s=40, marker='x')
		if not np.isnan(x_high[idx]):
			ax2.scatter(x_high[idx], y_high[idx], c=c, s=40, marker='s')

	# Marker legend entries
	low_proxy = plt.Line2D([0], [0], marker='o', color='k', linestyle='None', label='Low steady state')
	unst_proxy = plt.Line2D([0], [0], marker='x', color='k', linestyle='None', label='Unstable steady state')
	high_proxy = plt.Line2D([0], [0], marker='s', color='k', linestyle='None', label='High steady state')

	# Combine line + marker legends
	ax2.legend(handles=line_handles + [low_proxy, unst_proxy, high_proxy],
			   fontsize=8, loc='best')

	ax2.set_xlabel('x'); ax2.set_ylabel('y')
	ax2.set_title('Phase portrait (x vs y) for selected n')
	ax2.grid(True)

	plt.tight_layout()
	plt.show()


def sweep_ecx():
	"""
	Sweep activation threshold e_cx for the asymmetric two-gene network.
	Plot Newton-inferred and analytical steady states with detailed legend.
	"""

	params_ref = {
		'alpha_min': 0.1, 'alpha_max': 5.5, 'alpha_deg': 1.0,
		'beta_min': 0.1, 'beta_max': 4.0, 'beta_deg': 0.8,
		'e_cx': 1.0, 'e_cy': 2.0,
		'n': 4
	}

	ecx_values = np.linspace(1, 4, 20)
	x_low, y_low = [], []; x_unst, y_unst = [], []; x_high, y_high = [], []

	# --- Newton ---
	for ecx in ecx_values:
		params = params_ref.copy(); params['e_cx'] = ecx
		if ecx < 2:
			mid_guess = [1,1.5]
		elif ecx < 3.4:
			mid_guess = [2,1.5]
		else:
			mid_guess = [3.7, 2.8] 
		
		F, J = two_gene_network(params)
		sol_low,_ = newton_system(F,[0.1,0.1],J)
		sol_unst,_ = newton_system(F,mid_guess,J)
		sol_high,_ = newton_system(F,[4,4],J)
		x_low.append(sol_low[0]); y_low.append(sol_low[1])
		x_unst.append(sol_unst[0]); y_unst.append(sol_unst[1])
		x_high.append(sol_high[0]); y_high.append(sol_high[1])

	# --- Analytical ---
	x_low_ref, y_low_ref = [], []; x_unst_ref, y_unst_ref = [], []; x_high_ref, y_high_ref = [], []
	for ecx in ecx_values:
		params = params_ref.copy(); params['e_cx'] = ecx
		n = params['n']
		if ecx < 2:
			mid_guess = [1,1.5]
		elif ecx < 3.4:
			mid_guess = [2,1.5]
		else:
			mid_guess = [3.7, 2.8] 

		def x_null(y):
			return (params['alpha_min'] + (params['alpha_max'] - params['alpha_min'])*y**n /
					(params['e_cy']**n + y**n)) / params['alpha_deg']
		def y_null(x):
			return (params['beta_min'] + (params['beta_max'] - params['beta_min'])*x**n /
					(params['e_cx']**n + x**n)) / params['beta_deg']

		sol_low  = fsolve(lambda v:[v[0]-x_null(v[1]),v[1]-y_null(v[0])],[0.1,0.1])
		sol_unst = fsolve(lambda v:[v[0]-x_null(v[1]),v[1]-y_null(v[0])],mid_guess)
		sol_high = fsolve(lambda v:[v[0]-x_null(v[1]),v[1]-y_null(v[0])],[4,4])

		x_low_ref.append(sol_low[0]); y_low_ref.append(sol_low[1])
		x_unst_ref.append(sol_unst[0]); y_unst_ref.append(sol_unst[1])
		x_high_ref.append(sol_high[0]); y_high_ref.append(sol_high[1])

	fig, axes = plt.subplots(1,2,figsize=(14,6))
	ax1, ax2 = axes

	# --- SUBPLOT 1 ---
	ax1.scatter(ecx_values, x_low,  c='blue',   s=20, marker='s', label='x low (Newton)')
	ax1.scatter(ecx_values, y_low,  c='blue',   s=20, label='y low (Newton)')
	ax1.scatter(ecx_values, x_unst, c='orange', s=20, marker='s', label='x unstable (Newton)')
	ax1.scatter(ecx_values, y_unst, c='orange', s=20, label='y unstable (Newton)')
	ax1.scatter(ecx_values, x_high, c='green',  s=20, marker='s', label='x high (Newton)')
	ax1.scatter(ecx_values, y_high, c='green',  s=20, label='y high (Newton)')

	ax1.plot(ecx_values, x_low_ref,  'b-', alpha=0.7, label='x low (analytical)')
	ax1.plot(ecx_values, x_unst_ref, 'orange', alpha=0.7, label='x unstable (analytical)')
	ax1.plot(ecx_values, x_high_ref, 'g-', alpha=0.7, label='x high (analytical)')
	ax1.plot(ecx_values, y_low_ref,  'b--', alpha=0.7, label='y low (analytical)')
	ax1.plot(ecx_values, y_unst_ref, 'orange', ls='--', alpha=0.7, label='y unstable (analytical)')
	ax1.plot(ecx_values, y_high_ref, 'g--', alpha=0.7, label='y high (analytical)')

	ax1.set_xlabel('e_cx'); ax1.set_ylabel('Steady state value')
	ax1.set_title('Steady states vs e_cx')
	ax1.grid(True)
	ax1.legend(loc='center left', fontsize=8)

	# --- SUBPLOT 2 ---
	selected_indices = [0, len(ecx_values)//2, -1]
	phase_colors = ['purple','teal','darkred']
	line_handles = []

	for idx, c in zip(selected_indices, phase_colors):
		ecx = ecx_values[idx]
		params = params_ref.copy(); params['e_cx'] = ecx
		n = params['n']

		y_vals = np.linspace(-0.02,7,200)
		x_null = (params['alpha_min'] + (params['alpha_max'] - params['alpha_min']) * y_vals**n /
					(params['e_cy']**n + y_vals**n)) / params['alpha_deg']
		x_vals = np.linspace(-0.02,7,200)
		y_null = (params['beta_min'] + (params['beta_max'] - params['beta_min']) * x_vals**n /
					(params['e_cx']**n + x_vals**n)) / params['beta_deg']

		l1, = ax2.plot(x_null, y_vals, c=c, lw=1.5, label=f'x-null e_cx={ecx:.2f}', alpha=0.8)
		l2, = ax2.plot(x_vals, y_null, c=c, lw=1.5, ls='--', label=f'y-null e_cx={ecx:.2f}', alpha=0.8)
		line_handles.extend([l1,l2])

		ax2.scatter(x_low[idx], y_low[idx], c=c, s=40, marker='o')
		ax2.scatter(x_unst[idx], y_unst[idx], c=c, s=40, marker='x')
		ax2.scatter(x_high[idx], y_high[idx], c=c, s=40, marker='s')

	low_proxy = plt.Line2D([0],[0], marker='o', color='k', linestyle='None', label='Low steady state')
	unst_proxy = plt.Line2D([0],[0], marker='x', color='k', linestyle='None', label='Unstable steady state')
	high_proxy = plt.Line2D([0],[0], marker='s', color='k', linestyle='None', label='High steady state')

	ax2.legend(handles=line_handles + [low_proxy, unst_proxy, high_proxy], fontsize=8, loc='best')
	ax2.set_xlabel('x'); ax2.set_ylabel('y')
	ax2.set_title('Phase portrait (x vs y) for selected e_cx')
	ax2.grid(True)

	plt.tight_layout()
	plt.show()


# --------------------------
# 5. Newton-based basin-of-attraction
# --------------------------
def basin_of_attraction(params, x_range=(-1,7), y_range=(-1,7), grid_size=500):
    """
    Computes and plots Newton basins of attraction for the two-gene network.
    Detects convergence to low, unstable, and high steady states.
    Overlays basins, nullclines, and vector field.
    """

    F, J = two_gene_network(params)

    # --- Reference steady states for classification ---
    # (Use Newton to find them robustly)
    low_ref, _ = newton_system(F, np.array([0.05, 0.05]), J)
    mid_ref, _ = newton_system(F, np.array([1.0, 1.0]), J)     # unstable
    high_ref,_ = newton_system(F, np.array([5.0, 5.0]), J)

    refs = [low_ref, mid_ref, high_ref]  # order matters

    X = np.linspace(*x_range, grid_size)
    Y = np.linspace(*y_range, grid_size)
    basin = np.zeros((grid_size, grid_size))

    # --- Sweep initial guesses ---
    for i, x0 in enumerate(X):
        for j, y0 in enumerate(Y):
            sol, info = newton_system(F, np.array([x0, y0]), J)

            if not info['converged'] or np.any(np.isnan(sol)):
                basin[j, i] = np.nan
            else:
                # classify by closest steady state
                dists = [np.linalg.norm(sol - r) for r in refs]
                basin[j, i] = np.argmin(dists)   # 0=low, 1=unstable, 2=high

    # --- Nullclines ---
    y_vals = np.linspace(*y_range, 400)
    x_nullcline = (params['alpha_min'] +
                   (params['alpha_max'] - params['alpha_min']) *
                   y_vals**params['n'] /
                   (params['e_cy']**params['n'] + y_vals**params['n'])
                  ) / params['alpha_deg']

    x_vals = np.linspace(*x_range, 400)
    y_nullcline = (params['beta_min'] +
                   (params['beta_max'] - params['beta_min']) *
                   x_vals**params['n'] /
                   (params['e_cx']**params['n'] + x_vals**params['n'])
                  ) / params['beta_deg']

    # --- Vector field ---
    Xv, Yv = np.meshgrid(np.linspace(*x_range, 25),
                         np.linspace(*y_range, 25))
    U = np.zeros_like(Xv)
    V = np.zeros_like(Yv)

    for i in range(Xv.shape[0]):
        for j in range(Xv.shape[1]):
            f_val = F([Xv[i,j], Yv[i,j]])
            U[i,j] = f_val[0]
            V[i,j] = f_val[1]

    mag = np.sqrt(U**2 + V**2)
    U /= (mag + 1e-8)
    V /= (mag + 1e-8)

    # --- Plot ---
    plt.figure(figsize=(8,8))

    plt.imshow(basin, origin='lower',
               extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
               cmap='viridis', alpha=0.4)

    plt.colorbar(label='Newton convergence (0=low, 1=unstable, 2=high)')

    # Vector field
    plt.quiver(Xv, Yv, U, V, color='black', alpha=0.7)

    # Nullclines
    plt.plot(x_nullcline, y_vals, 'k--', linewidth=2, label='x-nullcline')
    plt.plot(x_vals, y_nullcline, 'k:',  linewidth=2, label='y-nullcline')

    # Steady states
    plt.scatter(low_ref[0],  low_ref[1],  c='blue',   s=120, edgecolors='k', label='Low (stable)')
    plt.scatter(mid_ref[0],  mid_ref[1],  c='orange', s=120, edgecolors='k', label='Unstable')
    plt.scatter(high_ref[0], high_ref[1], c='red',    s=120, edgecolors='k', label='High (stable)')

    plt.xlabel('Initial x')
    plt.ylabel('Initial y')
    plt.title('Newton Basins of Attraction with Vector Field')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.tight_layout()
    plt.show()

