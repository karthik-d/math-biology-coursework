import matplotlib.pyplot as plt
import numpy as np

from .rk4 import rk4_solver


def plot_solutions(t_rk, y_rk, t_ex, y_ex):
    
    plt.figure(figsize=(8,5))
    plt.plot(t_ex, y_ex, 'k-', label="Analytical", alpha=0.5)
    plt.plot(t_rk, y_rk, 'bo', label="RK4", alpha=0.5)

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Numerical solution of y' = y - t^2 + 1")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_rk4_solutions(f, y_exact_func, t0, y0, tf, h_values, reference_h=None):
	"""Plot RK4 solutions vs exact solution for multiple step sizes."""
	n_h = len(h_values)

	# Reference solution if needed
	if y_exact_func is None and reference_h is not None:
		t_ref, y_ref = rk4_solver(f, t0, y0, reference_h, tf)
		y_exact_func = lambda t: np.interp(t, t_ref, y_ref)

	plt.figure(figsize=(8*n_h, 4))

	for i, h in enumerate(h_values):
		t_num, y_num = rk4_solver(f, t0, y0, h, tf)
		y_exact_vals = y_exact_func(t_num)
		
		plt.subplot(1, n_h, i+1)
		plt.plot(t_num, y_num, 'k-', label="Analytical", alpha=0.5)
		plt.plot(t_num, y_exact_vals, 'bo', label="RK4", alpha=0.5, markersize=2)
		plt.xlabel('t')
		plt.ylabel('y(t)')
		plt.title(f'Solution h={h}')
		plt.grid(True)
		if i == 0:
			plt.legend(fontsize=8)

	plt.suptitle("Solutions")
	plt.tight_layout()
	plt.show()


def plot_rk4_relative_error(f, y_exact_func, t0, y0, tf, h_values, reference_h=None):
	"""Plot relative error |y_num - y_exact| / |y_exact| for multiple step sizes."""
	n_h = len(h_values)

	if y_exact_func is None and reference_h is not None:
		t_ref, y_ref = rk4_solver(f, t0, y0, reference_h, tf)
		y_exact_func = lambda t: np.interp(t, t_ref, y_ref)

	plt.figure(figsize=(20*n_h, 4))

	for i, h in enumerate(h_values):
		t_num, y_num = rk4_solver(f, t0, y0, h, tf)
		y_exact_vals = y_exact_func(t_num)
		rel_error = np.abs(y_num - y_exact_vals) / np.maximum(np.abs(y_exact_vals), 1e-14)
		
		plt.subplot(1, n_h, i+1)
		plt.plot(t_num, rel_error, 'r-')
		plt.xlabel('t')
		plt.ylabel('Relative Error')
		plt.title(f'h={h}')
		plt.grid(True)
            
		if i == 0:
			plt.gca().set_ylabel('Relative Error')
		else:
			plt.gca().set_yticklabels([])

	plt.suptitle("Relative Error")
	plt.tight_layout()
	plt.show()

def plot_rk4_local_error(f, y_exact_func, t0, y0, tf, h_values, reference_h=None):
    """
    Plot estimated local truncation error per step for multiple step sizes (RK4).
    
    Parameters
    ----------
    f : callable
        RHS function f(t, y)
    y_exact_func : callable
        Exact solution y(t)
    t0, y0, tf : float
        Initial conditions
    h_values : list or array
        Step sizes to evaluate
    reference_h : float, optional
        Step size for reference solution if exact solution not available
    """
    n_h = len(h_values)

    # Reference solution if needed
    if y_exact_func is None and reference_h is not None:
        t_ref, y_ref = rk4_solver(f, t0, y0, reference_h, tf)
        y_exact_func = lambda t: np.interp(t, t_ref, y_ref)

    # Dynamically scale figure width
    fig_width = max(5 * n_h, 12)
    plt.figure(figsize=(fig_width, 4))

    for i, h in enumerate(h_values):
        t_num, y_num = rk4_solver(f, t0, y0, h, tf)
        t_half, y_half = rk4_solver(f, t0, y0, h/2, tf)
        y_half_interp = np.interp(t_num, t_half, y_half)

        # RK4 LTE estimate
        local_error_est = np.abs(y_num - y_half_interp) / (2**4 - 1)

        ax = plt.subplot(1, n_h, i+1)
        ax.plot(t_num, local_error_est, 'm-')
        ax.grid(True)
        ax.set_title(f'h = {h}')

        # Only show y-axis label on first subplot
        if i == 0:
            ax.set_ylabel('LTE estimate')
        else:
            ax.set_yticklabels([])

        # Only show x-axis label on the bottom row / last subplot
        ax.set_xlabel('t')

    plt.suptitle("RK4 Local Truncation Error Estimates", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()
    

def rk4_relative_error_heatmap(f, y_exact_func, t0, y0, tf, h_values, n_t_points=200, reference_h=None):
    """
    Generate a dense heatmap of relative error vs t and h for RK4.

    x-axis: t
    y-axis: h (log scale)
    color: relative error
    """
    t_grid = np.linspace(t0, tf, n_t_points)

    # Reference solution if needed
    if y_exact_func is None and reference_h is not None:
        t_ref, y_ref = rk4_solver(f, t0, y0, reference_h, tf)
        y_exact_func = lambda t: np.interp(t, t_ref, y_ref)

    # Build error matrix: rows = h_values, columns = t_grid
    error_matrix = []
    
    for h in h_values:
        t_num, y_num = rk4_solver(f, t0, y0, h, tf)
        y_num_interp = np.interp(t_grid, t_num, y_num)
        y_exact_vals = y_exact_func(t_grid)
        
        # Relative error
        rel_error = np.abs(y_num_interp - y_exact_vals)
        error_matrix.append(rel_error)

    error_matrix = np.log(np.array(error_matrix))

    # Plot heatmap
    plt.figure(figsize=(10,6))

    # imshow expects y-axis to go from bottom to top
    im = plt.imshow(error_matrix, origin='lower', aspect='auto',
                    extent=[t0, tf, h_values[0], h_values[-1]],
                    cmap='viridis', interpolation='nearest')

    plt.colorbar(im, label='log(relative error)')
    plt.xlabel('t')
    plt.ylabel('Step size h')
    plt.yscale('log')  # step sizes often logarithmic
    plt.title('RK4 Relative Error Heatmap')
    plt.show()


def plot_loglog_error_with_slope(f, y_exact_func, t0, y0, tf, h_values):
    global_errors = []
    local_errors = []

    for h in h_values:
        # Global error at tf
        t_num, y_num = rk4_solver(f, t0, y0, h, tf)
        y_tf = y_exact_func(tf)
        global_errors.append(abs(y_num[-1] - y_tf))

        # True local truncation error: one step from exact data
        t_prev = tf - h
        y_exact_prev = y_exact_func(t_prev)
        t_step, y_step = rk4_solver(f, t_prev, y_exact_prev, h, tf)
        y_exact_tf = y_exact_func(tf)
        local_errors.append(abs(y_step[-1] - y_exact_tf))

    global_errors = np.array(global_errors)
    local_errors = np.array(local_errors)

    # Reference lines (correct orders)
    Cg = global_errors[len(global_errors)//2] / h_values[len(h_values)//2]**4
    Cl = local_errors[len(local_errors)//2] / h_values[len(h_values)//2]**5
    ref_global = Cg * h_values**4
    ref_local  = Cl * h_values**5

    # Detect asymptotic region via slope ~4
    slopes = np.diff(np.log(global_errors)) / np.diff(np.log(h_values))
    idx_asymp = np.where((slopes > 3.95) & (slopes < 4.05))[0]

    plt.figure(figsize=(7,6))

    plt.loglog(h_values, global_errors, 'gs-', label='Global error', markersize=3)
    plt.loglog(h_values, local_errors, 'bo-', label='Local truncation error', markersize=3)

    plt.loglog(h_values, ref_global, 'g--', label=r'Ref $\propto h^4$')
    plt.loglog(h_values, ref_local,  'b--', label=r'Ref $\propto h^5$')

    # Shade regimes
    if len(idx_asymp) > 0:
        h_left = h_values[idx_asymp[0]]
        h_right = h_values[idx_asymp[-1]+1]
        plt.axvspan(h_left, h_right, color='green', alpha=0.1, label='Asymptotic')
        plt.axvspan(h_values[0], h_left, color='red', alpha=0.1, label='Round-off dominated')
        plt.axvspan(h_right, h_values[-1], color='orange', alpha=0.1, label='Pre-asymptotic')

    plt.xlabel('Step size h')
    plt.ylabel('Error at t = 4')
    plt.title('Error Regimes for RK4')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    return local_errors, global_errors


def plot_piecewise_order(h_values, local_errors, global_errors):
    p_values_local = []
    p_values_global = []

    for i in range(len(h_values)-1):
        p = np.log(local_errors[i] / local_errors[i+1]) / np.log(h_values[i] / h_values[i+1])
        p_values_local.append(p)

        p = np.log(global_errors[i] / global_errors[i+1]) / np.log(h_values[i] / h_values[i+1])
        p_values_global.append(p)

    p_values_local = np.array(p_values_local)
    p_values_global = np.array(p_values_global)

    # Detect asymptotic region using global error order ≈ 4
    idx_asymp = np.where((p_values_global > 3.8) & (p_values_global < 4.2))[0]

    plt.figure(figsize=(7,5))
    plt.semilogx(h_values[:-1], p_values_local, 'bs-', label='Local error order', alpha=0.6, markersize=3)
    plt.semilogx(h_values[:-1], p_values_global, 'go-', label='Global error order', alpha=0.6, markersize=3)

    plt.axhline(4, color='g', linestyle='--', label='Global order = 4')
    plt.axhline(5, color='b', linestyle='--', label='Local order = 5')

    # Shade regimes
    if len(idx_asymp) > 0:
        h_left = h_values[idx_asymp[0]]
        h_right = h_values[idx_asymp[-1] + 1]

        plt.axvspan(h_values[0], h_left, color='red', alpha=0.12,
                    label='Round-off dominated')
        plt.axvspan(h_left, h_right, color='green', alpha=0.12,
                    label='Asymptotic (truncation-dominated)')
        plt.axvspan(h_right, h_values[-1], color='orange', alpha=0.12,
                    label='Pre-asymptotic (coarse h)')

    plt.xlabel('Step size h')
    plt.ylabel('Estimated order')
    plt.title('Piecewise Convergence Order (RK4)')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
