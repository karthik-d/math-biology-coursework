import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pkg.ivp.solver import solve_predictor_corrector, predictor_corrector_step


def exact_solution_linear_system(t, A, y0):
    """Compute exact solution y(t) = exp(At)y0."""
    return expm(A * t) @ y0


def analyze_errors(f, t_span, y0, A, h_values):
    """
    Analyze both local and global errors for predictor-corrector method.
    Plots both on a single log-log figure.
    """
    t0, T = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    gte = []  # Global error at final time
    lte = []  # Local error for single step
    t_lte = 0.5  # Arbitrary mid-point for local error
    
    for h in h_values:
        # --- Global Error ---
        N = int((T - t0)/h)
        t_grid = t0 + h*np.arange(N+1)
        Y = np.zeros((N+1, y0.size))
        F = np.zeros((N+1, y0.size))
        
        Y[0] = y0
        F[0] = f(t0, Y[0], A)
        if N > 0:
            Y[1] = exact_solution_linear_system(t_grid[1], A, y0)
            F[1] = f(t_grid[1], Y[1], A)
        
        for n in range(1, N):
            Y[n+1], F[n+1], _ = predictor_corrector_step(
                Y[n], F[n], F[n-1], h, f, t_grid[n+1], A
            )
        y_exact_final = exact_solution_linear_system(t_grid[-1], A, y0)
        gte.append(np.linalg.norm(Y[-1] - y_exact_final))
        
        # --- Local Error (single step) ---
        t_nm1, t_n, t_np1 = t_lte - h, t_lte, t_lte + h
        y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
        y_n = exact_solution_linear_system(t_n, A, y0)
        y_exact_np1 = exact_solution_linear_system(t_np1, A, y0)
        f_n = f(t_n, y_n, A)
        f_nm1 = f(t_nm1, y_nm1, A)
        y_num_np1, _, _ = predictor_corrector_step(y_n, f_n, f_nm1, h, f, t_np1, A)
        lte.append(np.linalg.norm(y_num_np1 - y_exact_np1))
    
    gte = np.array(gte)
    lte = np.array(lte)
    
    # Reference slopes
    gte_ref = gte[-1]*(h_values/h_values[-1])**2
    lte_ref = lte[-1]*(h_values/h_values[-1])**3
    
    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.loglog(h_values, gte, 'bo-', label='Global Error (GTE)')
    plt.loglog(h_values, lte, 'ro-', label='Local Error (LTE)')
    plt.loglog(h_values, gte_ref, 'b--', alpha=0.5, label='O(h^2) ref')
    plt.loglog(h_values, lte_ref, 'r--', alpha=0.5, label='O(h^3) ref')
    
    plt.xlabel("Step size h")
    plt.ylabel("Error")
    plt.title("Predictor-Corrector: Local and Global Error")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.show()
    
    return gte, lte