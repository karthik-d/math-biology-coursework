import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pkg.ivp.solver import f_linear, solve_pc


def analyze_pc_order(t0=0.0, T=0.1, hs=None, A=None, y0=None,
                     plot_filename=None, discard_first=0):
    """
    System:
        dy1/dt = -5 y1 +  3 y2
        dy2/dt = 100 y1 - 301 y2
        y1(0)  = 52.29, y2(0) = 83.82

    Parameters
    ----------
    t0 : float, optional
        Initial time (default 0.0).
    T : float, optional
        Final time (default 0.1).
    hs : array_like, optional
        Array of step sizes to test. If None, a default geometric sequence
        is used, dense over a wide range.
    A : ndarray, optional
        2×2 system matrix. If None, the matrix from the assignment is used.
    y0 : array_like, optional
        Initial condition. If None, the given initial condition is used.
    plot_filename : str or None, optional
        If given, save the log–log error vs h plot to this filename.
    discard_first : int, optional
        Number of largest step sizes to discard (for stability).

    Returns
    -------
    hs : ndarray
        Step sizes used.
    errors1 : ndarray
        Absolute errors |e1(T)| for y1.
    errors2 : ndarray
        Absolute errors |e2(T)| for y2.
    """
    # Default system
    if A is None:
        A = np.array([[-5.0,   3.0],
                      [100.0, -301.0]])
    if y0 is None:
        y0 = np.array([52.29, 83.82], dtype=float)

    # Dense geometric hs over a broader range
    if hs is None:
        exponents = np.linspace(-1, -10, 15)
        hs = 2.0**exponents

    # High-accuracy reference integrator
    def rhs(t, y):
        return f_linear(t, y, A)

    errors1 = np.zeros_like(hs, dtype=float)
    errors2 = np.zeros_like(hs, dtype=float)

    for i, h in enumerate(hs):
        N = int(np.round((T - t0) / h))
        if N < 1:
            errors1[i] = np.nan
            errors2[i] = np.nan
            continue
        T_eff = t0 + N * h

        # Numerical solution with predictor–corrector
        t_num, Y_num = solve_pc(f_linear, (t0, T_eff), y0, h, A)

        # Reference at T_eff
        sol_ref_h = solve_ivp(rhs, (t0, T_eff), y0, method="Radau",
                              rtol=1e-12, atol=1e-14)
        y_ref_h = sol_ref_h.y[:, -1]

        # Component-wise absolute errors
        errors1[i] = np.abs(Y_num[-1, 0] - y_ref_h[0])
        errors2[i] = np.abs(Y_num[-1, 1] - y_ref_h[1])

    # Remove NaNs and optionally discard largest hs
    mask = np.isfinite(errors1) & np.isfinite(errors2)
    hs_used = hs[mask]
    errors1_used = errors1[mask]
    errors2_used = errors2[mask]

    if discard_first > 0 and discard_first < len(hs_used):
        hs_used = hs_used[discard_first:]
        errors1_used = errors1_used[discard_first:]
        errors2_used = errors2_used[discard_first:]

    # Plot: component-wise errors vs h (log–log) with O(h) reference
    plt.figure(figsize=(8, 6))
    plt.loglog(hs_used, errors1_used, "o-", label="|e₁(T)|")
    plt.loglog(hs_used, errors2_used, "s-", label="|e₂(T)|")

    # Reference O(h^1) line (first-order global expected)
    hs_ref = np.logspace(np.log10(hs_used.min()),
                         np.log10(hs_used.max()), 200)
    plt.loglog(hs_ref, 1e-1 * hs_ref**1.0, "k--", linewidth=2,
               label="O(h¹) reference (expected first-order)")

    plt.gca().invert_xaxis()  # decreasing h left to right
    plt.xlabel("Step size h")
    plt.ylabel("Absolute error |eᵢ(T)|")
    plt.title("Predictor–Corrector: Component-wise global error vs step size h")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename, dpi=300)
    plt.show()

    return hs_used, errors1_used, errors2_used


def analyze_true_lte(t0=0.0, hs=None, A=None, y0=None,
                     plot_filename=None, fit_order=True):
    """
    Compute and plot the true LOCAL truncation error |τ_{n+1}| vs h for one
    step of the predictor-corrector scheme at t_n = t0.

    LTE = y(t_{n+1}) - Φ(t_n, y(t_n), y(t_{n-1}), h)
    where Φ is the full predictor-corrector step, using exact inputs.

    System:
        dy1/dt = -5 y1 +  3 y2
        dy2/dt = 100 y1 - 301 y2
        y1(t0) = 52.29, y2(t0) = 83.82

    Parameters
    ----------
    t0 : float, optional
        Time t_n where LTE is evaluated (default 0.0).
    hs : array_like, optional
        Step sizes. If None, a dense geometric sequence is used.
    A : ndarray, optional
        2×2 system matrix. If None, the assignment matrix is used.
    y0 : array_like, optional
        y(t0). If None, the given initial condition is used.
    plot_filename : str or None, optional
        Save plot filename.
    fit_order : bool, optional
        Fit and display the O(h^p) slope.

    Returns
    -------
    hs : ndarray
        Step sizes.
    ltes1 : ndarray
        |τ1| component errors.
    ltes2 : ndarray
        |τ2| component errors.
    p1_fit : float
        Fitted order for |τ1|.
    p2_fit : float
        Fitted order for |τ2|.
    """
    # Default system
    if A is None:
        A = np.array([[-5.0,   3.0],
                      [100.0, -301.0]])
    if y0 is None:
        y0 = np.array([52.29, 83.82], dtype=float)

    # Dense geometric hs
    if hs is None:
        hs = np.logspace(-4, 0, 25)  # h from 1e-4 to 1.0
        
    ltes1 = np.zeros_like(hs)
    ltes2 = np.zeros_like(hs)

    for i, h in enumerate(hs):
        t_nm1 = t0 - h
        t_n   = t0
        t_np1 = t0 + h

        # Exact solution y(t_k) = exp(A t_k) y0 at grid points
        y_nm1 = expm(A * t_nm1) @ y0
        y_n   = expm(A * t_n)   @ y0
        y_np1 = expm(A * t_np1) @ y0

        # f(t_k, y(t_k)) = A y(t_k)
        f_nm1 = A @ y_nm1
        f_n   = A @ y_n

        # Predictor: y^*_{n+1} = y_n + h/2 (3 f_n - f_{n-1})
        y_star = y_n + 0.5 * h * (3 * f_n - f_nm1)

        # Corrector: y_{n+1} = y_n + h/2 (3 f_n + f(y^*_{n+1}))
        f_star = A @ y_star
        y_num = y_n + 0.5 * h * (3 * f_n + f_star)

        # LTE = y(t_{n+1}) - numerical approximation
        tau = y_np1 - y_num
        ltes1[i] = np.abs(tau[0])
        ltes2[i] = np.abs(tau[1])

    # Fit orders for each component
    if fit_order:
        log_h = np.log(hs)
        log_tau1 = np.log(ltes1)
        p1_fit, c1_fit = np.polyfit(log_h, log_tau1, 1)

        log_tau2 = np.log(ltes2)
        p2_fit, c2_fit = np.polyfit(log_h, log_tau2, 1)
    else:
        p1_fit = p2_fit = np.nan

    # Plot component-wise LTE vs h (log–log)
    plt.figure(figsize=(8, 6))
    plt.loglog(hs, ltes1, "o-", label=f"|τ₁|, fit: h^{p1_fit:.2f}")
    plt.loglog(hs, ltes2, "s-", label=f"|τ₂|, fit: h^{p2_fit:.2f}")

    if fit_order:
        hs_fit = np.logspace(np.log10(hs.min()), np.log10(hs.max()), 200)
        plt.loglog(hs_fit, np.exp(c1_fit) * hs_fit**p1_fit, "b--", alpha=0.7)
        plt.loglog(hs_fit, np.exp(c2_fit) * hs_fit**p2_fit, "r--", alpha=0.7)
        plt.loglog(hs_fit, 1e-2 * hs_fit**2, "k:", label="O(h²) reference")

    plt.gca().invert_xaxis()  # decreasing h left to right
    plt.xlabel("Step size h")
    plt.ylabel("Local truncation error |τᵢ|")
    plt.title(f"Predictor–Corrector: True LTE vs h (one step at tₙ={t0})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename, dpi=300)
    plt.show()

    return hs, ltes1, ltes2, p1_fit, p2_fit
