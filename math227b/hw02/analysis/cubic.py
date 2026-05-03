from autograd import elementwise_grad
from autograd import numpy as anp
import matplotlib.pyplot as plt
import numpy as np

from pkg.cubic import cubic
from analysis import utils


# ---- Function to compute interpolation results ----
def interpolation_results(f, a, b, n, x_test, bc_type="natural"):
    x_nodes = utils.generate_nodes(a, b, n)
    y_nodes = f(x_nodes)

    # ---- Determine boundary condition ----
    if bc_type == "complete":
        # Use first derivatives at endpoints from the original function
        fp_a = elementwise_grad(lambda x: anp.array(f(x)))(a)
        fp_b = elementwise_grad(lambda x: anp.array(f(x)))(b)
        bc = ((1, fp_a), (1, fp_b))
    else:
        # Natural spline (second derivatives zero at endpoints)
        bc = "natural"

    spline = cubic.interpolate(x_nodes, y_nodes, bc_type=bc)

    p_vals = spline(x_test)
    f_vals = f(x_test)

    errors = np.abs(f_vals - p_vals)
    max_error = np.max(errors)
    rms_error = np.sqrt(np.mean(errors**2))

    return p_vals, f_vals, errors, max_error, rms_error, x_nodes, spline


# ---- Main analysis function with derivatives, RMS, and subplot fits ----
def analyze_error(f, a, b, func_name, bc_type="natural"):
    n_values = list(range(5, 55, 2))  # number of spline nodes
    num_test_points = 500
    x_test = np.linspace(a, b, num_test_points)

    # --- Store errors ---
    max_errors_f, rms_errors_f = [], []
    max_errors_f1, rms_errors_f1 = [], []
    max_errors_f2, rms_errors_f2 = [], []

    hs = []

    # --- Store interpolated values for subplot fits ---
    p_vals_all = []
    f_vals_all = []
    n_nodes_all = []

    for n in n_values:
        # Interpolation
        p_vals, f_vals, errors, max_err, rms_err, x_nodes, spline = interpolation_results(
            f, a, b, n, x_test, bc_type=bc_type
        )

        h = (b - a) / (n - 1)
        hs.append(h)

        # Function errors
        max_errors_f.append(max_err)
        rms_errors_f.append(rms_err)

        # First derivative
        f1 = elementwise_grad(lambda x: anp.array(f(x)))  # ensure autograd output is array
        spline1 = spline.derivative()
        err1 = np.abs(np.array(f1(x_test)) - spline1(x_test))
        max_errors_f1.append(np.max(err1))
        rms_errors_f1.append(np.sqrt(np.mean(err1**2)))

        # Second derivative
        f2_func = elementwise_grad(f1)
        spline2 = spline1.derivative()
        err2 = np.abs(np.array(f2_func(x_test)) - spline2(x_test))
        max_errors_f2.append(np.max(err2))
        rms_errors_f2.append(np.sqrt(np.mean(err2**2)))

        # Store for subplot fits
        p_vals_all.append(p_vals)
        f_vals_all.append(f_vals)
        n_nodes_all.append(n)

    # Convert to numpy arrays
    hs = np.array(hs)
    max_errors_f = np.array(max_errors_f)
    rms_errors_f = np.array(rms_errors_f)
    max_errors_f1 = np.array(max_errors_f1)
    rms_errors_f1 = np.array(rms_errors_f1)
    max_errors_f2 = np.array(max_errors_f2)
    rms_errors_f2 = np.array(rms_errors_f2)

    # -----------------------------
    # 1) Log-log plots: f, f', f'' errors vs h (max & RMS)
    # -----------------------------
    plt.figure(figsize=(14,6))
    ax1 = plt.subplot(1,3,1)
    ax1.loglog(hs, max_errors_f, 'o-', label='Max f error', color='C0')
    ax1.loglog(hs, rms_errors_f, 'o--', label='RMS f error', color='C0')
    ax1.loglog(hs, max_errors_f[-1]*(hs/hs[-1])**4, 'k--', label='slope=4 ref')
    ax1.set_xlabel('h'); ax1.set_ylabel('Error'); ax1.set_title(f'f error ({bc_type})'); ax1.grid(True); ax1.legend()

    ax2 = plt.subplot(1,3,2)
    ax2.loglog(hs, max_errors_f1, 's-', label="Max f' error", color='C1')
    ax2.loglog(hs, rms_errors_f1, 's--', label="RMS f' error", color='C1')
    ax2.loglog(hs, max_errors_f1[-1]*(hs/hs[-1])**3, 'k--', label='slope=3 ref')
    ax2.set_xlabel('h'); ax2.set_ylabel('Error'); ax2.set_title(f"f' error ({bc_type})"); ax2.grid(True); ax2.legend()

    ax3 = plt.subplot(1,3,3)
    ax3.loglog(hs, max_errors_f2, '^-', label="Max f'' error", color='C2')
    ax3.loglog(hs, rms_errors_f2, '^--', label="RMS f'' error", color='C2')
    ax3.loglog(hs, max_errors_f2[-1]*(hs/hs[-1])**2, 'k--', label='slope=2 ref')
    ax3.set_xlabel('h'); ax3.set_ylabel('Error'); ax3.set_title(f"f'' error ({bc_type})"); ax3.grid(True); ax3.legend()

    plt.suptitle(f'Cubic Spline ({bc_type}): Errors vs h, for {func_name}')
    plt.show()

    # -----------------------------
    # 2) Ratio plot: scaled errors
    # -----------------------------
    plt.figure(figsize=(12,4))
    plt.plot(hs, max_errors_f / hs**4, 'o-', label='Max f / h^4', color='C0')
    plt.plot(hs, rms_errors_f / hs**4, 'o--', label='RMS f / h^4', color='C0')
    plt.xlabel('h'); plt.ylabel('Scaled error')
    plt.title(f'Cubic Spline ({bc_type}): Scaled Errors, for {func_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # -----------------------------
    # 3) Reference line vs actual error (log-log, h^4)
    # -----------------------------
    C_ref = max_errors_f[-1] / hs[-1]**4
    ref_curve = C_ref * hs**4

    plt.figure(figsize=(8,6))
    plt.loglog(hs, max_errors_f, 'o-', label='Max error', color='C0')
    plt.loglog(hs, rms_errors_f, 'o--', label='RMS error', color='C0')
    plt.loglog(hs, ref_curve, '--', label='C h^4 reference', color='k')
    plt.xlabel('h'); plt.ylabel('Error')
    plt.title(f'Cubic Spline ({bc_type}): Error vs h^4, for {func_name}')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    # -----------------------------
    # 4) Interpolated fits for each n (subplot figure)
    # -----------------------------
    plt.figure(figsize=(16,16))
    ncols = 5
    nrows = int(np.ceil(len(n_nodes_all) / ncols))
    for i, n in enumerate(n_nodes_all):
        plt.subplot(nrows, ncols, i+1)
        plt.plot(x_test, p_vals_all[i], 'k--', label='Spline')
        plt.plot(x_test, f_vals_all[i], 'b', alpha=0.3, label='Actual')
        plt.scatter(utils.generate_nodes(a, b, n),
                    f(utils.generate_nodes(a, b, n)),
                    c='r', s=10)
        plt.xlabel('x'); plt.ylabel('f(x)')
        plt.title(f'n={n}')
        plt.grid(True)
    plt.suptitle(f'Cubic Spline ({bc_type}): Interpolation vs Actual, for {func_name}')
    plt.legend()
    plt.show()
