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


def local_error_heatmap(f, y0, A, h_values, t_values):
    """
    Compute and plot a heatmap of local truncation error (LTE) 
    with time on the x-axis and step size h on the y-axis.
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    LTE_matrix = np.zeros((len(h_values), len(t_values)))

    for i, h in enumerate(h_values):
        for j, t_n in enumerate(t_values):
            t_nm1, t_np1 = t_n - h, t_n + h
            y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
            y_n = exact_solution_linear_system(t_n, A, y0)
            y_exact_np1 = exact_solution_linear_system(t_np1, A, y0)
            f_n = f(t_n, y_n, A)
            f_nm1 = f(t_nm1, y_nm1, A)
            y_num_np1, _, _ = predictor_corrector_step(y_n, f_n, f_nm1, h, f, t_np1, A)
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
    plt.title("LTE Heatmap for Predictor-Corrector Method")
    plt.yscale('log') 
    plt.show()

    return LTE_matrix