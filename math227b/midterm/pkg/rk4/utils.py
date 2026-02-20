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
		t_num, y_num = rk4_solver(f, t0, y0, h, tf)
		y_tf = y_exact_func(tf)
		global_errors.append(abs(y_num[-1] - y_tf))
		# LTE.
		t_prev = tf - h
		y_exact_prev = y_exact_func(t_prev)
		# one RK4 step from exact value
		t_step, y_step = rk4_solver(f, t_prev, y_exact_prev, h, tf)
		y_exact_tf = y_exact_func(tf)
		local_errors.append(abs(y_step[-1] - y_exact_tf))
			
	global_errors = np.array(global_errors)
	local_errors = np.array(local_errors)

	# Reference slope h^4 (scaled to match first data point)
	C = global_errors[-1] / h_values[-1]**5
	ref_line_4 = C * h_values**5
	C = local_errors[-1] / h_values[-1]**4
	ref_line_5 = C * h_values**4

	plt.figure(figsize=(6,5))
	plt.loglog(h_values, global_errors, 'gs-', label='global error', alpha=0.7, markersize=2)
	plt.loglog(h_values, local_errors, 'bo-', label='local error', alpha=0.7, markersize=2)
	plt.loglog(h_values, ref_line_5, 'g--', label=r'Reference $\propto h^4$')
	plt.loglog(h_values, ref_line_4, 'b--', label=r'Reference $\propto h^4$')
	plt.xlabel('Step size h')
	plt.ylabel('Error at t = 4')
	plt.title('Global Error vs Step Size (RK4)')
	plt.grid(True, which='both')
	plt.legend()
	plt.show()

	return local_errors, global_errors


def plot_piecewise_order(h_values, local_errors, global_errors):
    p_values_local = []

    for i in range(len(h_values)-1):
        p = np.log(local_errors[i] / local_errors[i+1]) / np.log(h_values[i] / h_values[i+1])
        p_values_local.append(p)
        p = np.log(global_errors[i] / global_errors[i+1]) / np.log(h_values[i] / h_values[i+1])
        p_values_global.append(p)

    p_values_local = np.array(p_values_local)
    p_values_global = np.array(p_values_global)

    plt.figure(figsize=(6,4))
    plt.semilogx(h_values, p_values_local, 'bs-', label='local error', alpha=0.7, markersize=2)
    plt.semilogx(h_values, p_values_global, 'go-', label='global error', alpha=0.7, markersize=2)
    plt.axhline(4, color='b', linestyle='--', label='order = 4')
    plt.axhline(5, color='g', linestyle='--', label='order = 5')
    plt.xlabel('Step size h')
    plt.ylabel('Estimated order')
    plt.title('Piecewise Convergence Order (RK4)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return p_values


def plot_roundoff_vs_truncation(f, y_exact_func, t0, y0, tf):
    h_values = np.logspace(-8, -1, 40)
    errors = []

    for h in h_values:
        t_num, y_num = rk4_solver(f, t0, y0, h, tf)
        y_tf = y_exact_func(tf)
        errors.append(abs(y_num[-1] - y_tf))

    errors = np.array(errors)

    plt.figure(figsize=(6,5))
    plt.loglog(h_values, errors, 'o-', label='RK4 error')

    # Reference slope h^4 (scaled in mid-range)
    mid = len(h_values)//2
    C = errors[mid] / h_values[mid]**4
    plt.loglog(h_values, C*h_values**4, 'k--', label=r'Reference $\propto h^4$')

    plt.xlabel('Step size h')
    plt.ylabel('Error at t = tf')
    plt.title('Roundoff vs Truncation Error Regimes')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    return h_values, errors