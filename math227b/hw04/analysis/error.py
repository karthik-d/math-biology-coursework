import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pkg.ivp.solver import predictor_corrector_step


def exact_solution_linear_system(t, A, y0):
    """Compute exact solution y(t) = exp(At)y0."""
    return expm(A * t) @ y0


def analyze_errors(f, solver_fn, t_span, y0, A, h_values, t_lte=0.5):
    """
    Compute Global Truncation Error (GTE) and Local Truncation Error (LTE)
    correctly. LTE is computed as single-step error starting from exact previous values.
    """
    
    t0, T = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    gte = []
    lte = []
    
    for h in h_values:
        # -------------------------
        # 1. Global Truncation Error
        # -------------------------
        t_num, Y_num = solver_fn(f, t_span, y0, h, A)
        Y_exact_final = exact_solution_linear_system(t_num[-1], A, y0)
        gte.append(np.linalg.norm(Y_num[-1] - Y_exact_final))
        
        # -------------------------
        # 2. Local Truncation Error
        # -------------------------
        t_n  = t_lte
        t_nm1 = t_n - h
        t_np1 = t_n + h
        
        # Exact previous values
        y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
        y_n   = exact_solution_linear_system(t_n, A, y0)
        y_exact_np1 = exact_solution_linear_system(t_np1, A, y0)
        
        if "predictor_corrector" in solver_fn.__name__.lower():
            # PECE: single predictor-corrector step from exact y_n, y_nm1
            f_n = f(t_n, y_n, A)
            f_nm1 = f(t_nm1, y_nm1, A)
            y_num_np1, _, _ = predictor_corrector_step(y_n, f_n, f_nm1, h, f, t_np1, A)
        else:
            # Predictor-only (explicit) single step using exact previous values
            # For AB2: y_{n+1} = y_n + h/2 * (3 f_n - f_{n-1})
            f_n = f(t_n, y_n, A)
            f_nm1 = f(t_nm1, y_nm1, A)
            y_num_np1 = y_n + (h/2)*(3*f_n - f_nm1)
        
        lte.append(np.linalg.norm(y_num_np1 - y_exact_np1))
    
    gte = np.array(gte)
    lte = np.array(lte)
    
    # Reference slopes
    gte_ref = gte[-1]*(h_values/h_values[-1])**2          # AB2 GTE ~ O(h^2)
    lte_ref = lte[-1]*(h_values/h_values[-1])**3          # PECE LTE ~ O(h^3)
    
    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.loglog(h_values, gte, 'bo-', alpha=0.6, label='Global Error (GTE)')
    plt.loglog(h_values, lte, 'ro-', alpha=0.6, label='Local Error (LTE)')
    plt.loglog(h_values, gte_ref, 'b--', alpha=1, label='O(h^2) ref')
    plt.loglog(h_values, lte_ref, 'r--', alpha=1, label='O(h^3) ref')
    
    plt.xlabel("Step size h")
    plt.ylabel("Error")
    plt.title(f"{solver_fn.__name__}: Local and Global Error (correct LTE)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.show()
    
    return gte, lte


def compare_global_errors(f, solver_fn1, solver_fn2, t_span, y0, A, h_values):
    """
    Compare global truncation error of two solver functions.
    """
    
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    gte1 = []
    gte2 = []
    
    # Compute GTE for first solver
    for h in h_values:
        t_num, Y_num = solver_fn1(f, t_span, y0, h, A)
        Y_exact_final = exact_solution_linear_system(t_num[-1], A, y0)
        gte1.append(np.linalg.norm(Y_num[-1] - Y_exact_final))
    
    # Compute GTE for second solver
    for h in h_values:
        t_num, Y_num = solver_fn2(f, t_span, y0, h, A)
        Y_exact_final = exact_solution_linear_system(t_num[-1], A, y0)
        gte2.append(np.linalg.norm(Y_num[-1] - Y_exact_final))
    
    gte1 = np.array(gte1)
    gte2 = np.array(gte2)
    
    # Reference O(h^2) line (optional, based on smaller solver)
    gte_ref = gte1[-1]*(h_values/h_values[-1])**2
    
    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.loglog(h_values, gte1, color='#FF7F0E', marker='o', lw=1.8, alpha=0.8, label=solver_fn1.__name__)
    plt.loglog(h_values, gte2, color='#2CA02C', marker='s', lw=1.8, alpha=0.8, label=solver_fn2.__name__)
    plt.loglog(h_values, gte_ref, 'k--', alpha=0.6, lw=1.5, label='O(h^2) ref')
    
    plt.xlabel("Step size h")
    plt.ylabel("Global Error (GTE)")
    plt.title("Global Error Comparison")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.show()
    
    return gte1, gte2


def compare_local_errors(f, solver_fn1, solver_fn2, t_span, y0, A, h_values, t_lte=0.5):
    """
    Compare local truncation errors (LTE) for two solver functions.
    Computes LTE as the error of the **last step before t_lte**.
    """
    
    t0, _ = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    lte1 = []
    lte2 = []

    for h in h_values:
        # --- Solver 1 ---
        # Take enough steps from t0 to reach t_lte
        n_steps = int(np.ceil((t_lte - t0)/h))
        t_end = t0 + n_steps*h
        t_num, Y_num = solver_fn1(f, (t0, t_end), y0, h, A)
        
        # Last step LTE: compare last numerical point to exact at that time
        y_exact = exact_solution_linear_system(t_num[-1], A, y0)
        lte1.append(np.linalg.norm(Y_num[-1] - y_exact))
        
        # --- Solver 2 ---
        t_num, Y_num = solver_fn2(f, (t0, t_end), y0, h, A)
        y_exact = exact_solution_linear_system(t_num[-1], A, y0)
        lte2.append(np.linalg.norm(Y_num[-1] - y_exact))
    
    lte1 = np.array(lte1)
    lte2 = np.array(lte2)
    
    # Reference O(h^3) line (for typical predictor-corrector LTE)
    lte_ref = lte1[-1]*(h_values/h_values[-1])**3
    
    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.loglog(h_values, lte1, color='#FF7F0E', marker='o', lw=1.8, alpha=0.8, label=solver_fn1.__name__)
    plt.loglog(h_values, lte2, color='#2CA02C', marker='s', lw=1.8, alpha=0.8, label=solver_fn2.__name__)
    plt.loglog(h_values, lte_ref, 'k--', lw=1.5, alpha=0.6, label='O(h^3) ref')
    
    plt.xlabel("Step size h")
    plt.ylabel(f"Local Truncation Error (LTE) at t≈{t_lte}")
    plt.title("Local Error Comparison")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.show()
    
    return lte1, lte2

def compute_single_step_LTE(f, solver_fn, y_nm1, y_n, t_nm1, t_n, h, A):
    """
    Compute the LTE for a single step from t_n to t_n + h using exact previous values.
    
    Parameters
    ----------
    f : callable
        Derivative function f(t, y, A)
    solver_fn : callable
        Solver function (used to determine method type)
    y_nm1 : array_like
        Exact solution at previous step
    y_n : array_like
        Exact solution at current step
    t_nm1 : float
        Previous step time
    t_n : float
        Current step time
    h : float
        Step size
    A : array_like
        System matrix (linear system)
    
    Returns
    -------
    LTE : float
        Norm of single-step local truncation error
    """
    t_np1 = t_n + h
    y_exact_np1 = exact_solution_linear_system(t_np1, A, y0=y_nm1)  # exact at next step

    if "predictor_corrector" in solver_fn.__name__.lower():
        # PECE: predictor-corrector single step
        f_n = f(t_n, y_n, A)
        f_nm1 = f(t_nm1, y_nm1, A)
        y_num_np1, _, _ = predictor_corrector_step(y_n, f_n, f_nm1, h, f, t_np1, A)
    else:
        # AB2: explicit predictor step
        f_n = f(t_n, y_n, A)
        f_nm1 = f(t_nm1, y_nm1, A)
        y_num_np1 = y_n + (h/2)*(3*f_n - f_nm1)

    return np.linalg.norm(y_num_np1 - y_exact_np1)


def local_error_heatmap(f, solver_fn, y0, A, h_values, t_values):
    """
    Compute LTE heatmap using the exact previous values at each step.
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    LTE_matrix = np.zeros((len(h_values), len(t_values)))

    for i, h in enumerate(h_values):
        for j, t_end in enumerate(t_values):
            # --- simulate steps from t0 to t_end ---
            t0 = t_values[0]
            n_steps = max(1, int(np.ceil((t_end - t0)/h)))
            t_grid = t0 + h * np.arange(n_steps + 1)

            LTE_steps = np.zeros(n_steps)
            for k in range(1, n_steps + 1):
                t_nm1 = t_grid[k-1]
                t_n = t_grid[k]

                y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
                y_n = exact_solution_linear_system(t_n, A, y0)

                LTE_steps[k-1] = compute_single_step_LTE(f, solver_fn, y_nm1, y_n, t_nm1, t_n, h, A)

            # Take LTE of last step to t_end
            LTE_matrix[i, j] = LTE_steps[-1]

    # --- Plot heatmap ---
    plt.figure(figsize=(10,6))
    im = plt.imshow(
        np.log(LTE_matrix.T),
        aspect='auto',
        origin='lower',
        extent=[t_values[0], t_values[-1], h_values[0], h_values[-1]],
        cmap='viridis'
    )
    plt.colorbar(im, label="log(local truncation error)")
    plt.xlabel("Time t")
    plt.ylabel("Step size h")
    plt.title(f"LTE Heatmap using {solver_fn.__name__} (single-step)")
    plt.yscale('log')
    plt.show()

    return LTE_matrix