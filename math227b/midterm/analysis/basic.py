import numpy as np	
from matplotlib import pyplot as plt

from pkg.rk4.rk4 import rk4_solver


def test_function():
	
	# ode function.
	def f(t, y):
		return y - t**2 + 1

	# analytical soln. function.
	def y(t):
		return (t + 1)**2 - 0.5 * np.exp(t)

	params = dict(
		t0 = 0.0,
		tf = 4.0,
		y0 = 0.5
	)
	return f, y, params


def compute_errors(f, y_exact_func, t0, y0, tf, h_values):
    """
    Compute RK4 errors at tf for different step sizes.

    Returns
    -------
    errors : list
        List of errors corresponding to each h
    """
    errors = []
    for h in h_values:
        t_rk, y_rk = rk4_solver(f, t0, y0, h, tf)
        y_exact_tf = y_exact_func(tf)
        err = abs(y_rk[-1] - y_exact_tf)
        errors.append(err)
    return errors


def estimate_order(h_values, errors):
    """
    Estimate order of accuracy p from consecutive errors.

    Parameters
    ----------
    h_values : list or ndarray
        Step sizes
    errors : list or ndarray
        Errors at corresponding step sizes

    Returns
    -------
    p_values : list
        Estimated order between consecutive step sizes
    """
    p_values = []
    for i in range(len(errors)-1):
        p = np.log(errors[i]/errors[i+1]) / np.log(h_values[i]/h_values[i+1])
        p_values.append(p)
    return p_values


def run_order_analysis_clean(f, y_exact_func, params, expected_order=4):
    """
    Clean RK4 convergence analysis with pre-selected step sizes that 
    show clear 4th-order behavior.
    """

    # Pre-selected step sizes for good RK4 asymptotic region
    h_values = np.linspace(0.01, 1, 200)

    # Compute errors
    errors = compute_errors(f, y_exact_func, params['t0'], params['y0'], params['tf'], h_values)
    p_values = estimate_order(h_values, errors)

    # Plot log-log convergence with reference slope
    plt.figure(figsize=(7,5))
    plt.loglog(h_values, errors, 'o-', label='RK4 error', markersize=6)

    # Reference slope
    h_ref = np.array(h_values)
    error_ref = errors[-1]  # anchor at smallest h
    ref_line = error_ref * (h_ref / h_ref[-1])**expected_order
    plt.loglog(h_ref, ref_line, 'k--', label=f'Reference slope {expected_order}')

    plt.xlabel('Step size h')
    plt.ylabel(f'Error at t={params['tf']}')
    plt.title('RK4 Convergence (Clean Region)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.show()

    return h_values, errors, p_values


def run_order_analysis_regions(f, y_exact_func, params, expected_order=4):
    """
    RK4 convergence analysis showing regions of global error and round-off.
    """

    # Step sizes spanning several orders of magnitude to illustrate both regions
    h_values = np.logspace(-4, 0, 200)  # 0.0001 to ~0.316

    # Compute errors
    errors = compute_errors(f, y_exact_func, params['t0'], params['y0'], params['tf'], h_values)
    p_values = estimate_order(h_values, errors)

    # Estimate "good" region where slope ~ expected_order
    # We'll use a simple method: find consecutive p-values within 20% of expected_order
    good_region = []
    for i, p in enumerate(p_values):
        if 0.8*expected_order <= p <= 1.2*expected_order:
            good_region.append(i)

    # Plot log-log convergence
    plt.figure(figsize=(8,5))
    plt.loglog(h_values, errors, 'o-', label='RK4 error', markersize=5)

    # Reference slope anchored at middle of good region
    if good_region:
        mid_idx = good_region[len(good_region)//2]
        h_ref = np.array(h_values)
        error_ref = errors[mid_idx]
        ref_line = error_ref * (h_ref / h_ref[mid_idx])**expected_order
        plt.loglog(h_ref, ref_line, 'k--', label=f'Reference slope {expected_order}')

        # Highlight global-error region
        plt.axvspan(h_values[good_region[0]], h_values[good_region[-1]], color='green', alpha=0.2, label='Global error region')

    # Highlight round-off dominated region (small h)
    plt.axvspan(h_values[0], h_values[good_region[0]] if good_region else h_values[5], color='red', alpha=0.2, label='Round-off dominated')

    plt.xlabel('Step size h')
    plt.ylabel('Error at t=tf')
    plt.title('RK4 Convergence with Regions')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.show()
    
    return h_values, errors, p_values, good_region




