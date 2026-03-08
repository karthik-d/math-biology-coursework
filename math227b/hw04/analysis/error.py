import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pkg.ivp.solver import predictor_corrector_step


def exact_solution_linear_system(t, A, y0):
    """Compute exact solution y(t) = exp(At)y0."""
    return expm(A * t) @ y0


def analyze_errors(f, solver_fn, t_span, y0, A, h_values, t_lte=0.5):
    
    t0, T = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    gte = []
    lte = []
    
    for h in h_values:
        # --- Global Error via full solver ---
        t_num, Y_num = solver_fn(f, t_span, y0, h, A)
        Y_exact_final = exact_solution_linear_system(t_num[-1], A, y0)
        gte.append(np.linalg.norm(Y_num[-1] - Y_exact_final))
        
        # --- Local Truncation Error (single step) ---
        t_nm1, t_n = t_lte - h, t_lte
        y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
        y_n = exact_solution_linear_system(t_n, A, y0)
        f_nm1 = f(t_nm1, y_nm1, A)
        f_n = f(t_n, y_n, A)
        
        # One predictor-corrector step to t_n + h
        y_num_np1, _, _ = predictor_corrector_step(y_n, f_n, f_nm1, h, f, t_n + h, A)
        y_exact_np1 = exact_solution_linear_system(t_n + h, A, y0)
        
        lte.append(np.linalg.norm(y_num_np1 - y_exact_np1))
    
    gte = np.array(gte)
    lte = np.array(lte)
    
    # Reference slopes
    gte_ref = gte[-1]*(h_values/h_values[-1])**2
    lte_ref = lte[-1]*(h_values/h_values[-1])**3
    
    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.loglog(h_values, gte, 'bo-', alpha=0.6, label='Global Error (GTE)')
    plt.loglog(h_values, lte, 'ro-', alpha=0.6, label='Local Error (LTE)')
    plt.loglog(h_values, gte_ref, 'b--', alpha=1, label='O(h^2) ref')
    plt.loglog(h_values, lte_ref, 'r--', alpha=1, label='O(h^3) ref')
    
    plt.xlabel("Step size h")
    plt.ylabel("Error")
    plt.title(f"{solver_fn.__name__}: Local and Global Error")
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
    
    Parameters
    ----------
    f : callable
        Derivative function f(t, y, A)
    solver_fn1, solver_fn2 : callable
        Solvers taking (f, t_span, y0, h, A) and returning (t_array, Y_array)
    t_span : tuple
        (t0, T)
    y0 : array_like
        Initial condition
    A : array_like
        System matrix (for linear systems)
    h_values : array_like
        Step sizes
    t_lte : float
        Time at which to evaluate LTE (last step before t_lte)
        
    Returns
    -------
    lte1, lte2 : np.ndarray
        LTE for solver_fn1 and solver_fn2
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


def local_error_heatmap(f, solver_fn, y0, A, h_values, t_values):
    """
    Compute and plot a heatmap of local truncation error (LTE) 
    with time on the x-axis and step size h on the y-axis, 
    using a provided solver function for single-step evaluation.
    """
    
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    LTE_matrix = np.zeros((len(h_values), len(t_values)))
    
    for i, h in enumerate(h_values):
        for j, t_n in enumerate(t_values):
            t_nm1, t_np1 = t_n - h, t_n + h
            # Exact previous value and current value
            y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
            y_n = exact_solution_linear_system(t_n, A, y0)
            y_exact_np1 = exact_solution_linear_system(t_np1, A, y0)
            
            # Use single-step method for LTE
            # For predictor-corrector, call predictor_corrector_step
            # For other solver_fn, assume it can integrate a single step from t_n to t_np1
            t_step, Y_step = solver_fn(f, (t_n, t_np1), y_n, h, A)
            y_num_np1 = Y_step[-1]
            
            LTE_matrix[i, j] = np.linalg.norm(y_num_np1 - y_exact_np1)
    
    # Plot heatmap
    plt.figure(figsize=(10,6))
    im = plt.imshow(
        np.log(LTE_matrix), 
        aspect='auto', 
        origin='lower', 
        extent=[t_values[0], t_values[-1], h_values[0], h_values[-1]],
        cmap='viridis'
    )
    plt.colorbar(im, label="log(local truncation error)")
    plt.xlabel("Time t")
    plt.ylabel("Step size h")
    plt.title(f"LTE Heatmap using {solver_fn.__name__}")
    plt.yscale('log')
    plt.show()
    
    return LTE_matrix