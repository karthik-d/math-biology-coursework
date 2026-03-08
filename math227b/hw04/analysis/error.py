import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pkg.ivp.solver import solve_predictor_corrector, predictor_corrector_step


def exact_solution_linear_system(t, A, y0):
    """Computes exact solution y(t) = exp(At)y0."""
    return expm(A * t) @ y0


def analyze_global_error(f, t_span, y0, A, h_values):
    """
    Analyze global truncation error for the predictor–corrector method.

    Expected slope: 2 (O(h^2))
    """

    t0, T = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    errors = []

    # Exact terminal solution
    y_exact_T = exact_solution_linear_system(T, A, y0)

    for h in h_values:

        # Robust step count
        N = int((T - t0) / h)

        # Construct time grid
        t = t0 + h * np.arange(N + 1)

        Y = np.zeros((N + 1, y0.size))
        F = np.zeros((N + 1, y0.size))

        # Perfect start
        Y[0] = y0
        F[0] = f(t[0], Y[0], A)

        if N > 0:
            Y[1] = exact_solution_linear_system(t[1], A, y0)
            F[1] = f(t[1], Y[1], A)

        # Main PECE loop
        for n in range(1, N):

            Y[n+1], F[n+1], _ = predictor_corrector_step(
                Y[n],
                F[n],
                F[n-1],
                h,
                f,
                t[n+1],
                A
            )

        # Error at final time
        errors.append(np.linalg.norm(Y[-1] - y_exact_T))

    errors = np.array(errors)

    # Fit slope
    slope = np.polyfit(np.log(h_values), np.log(errors), 1)[0]

    # Pairwise observed orders (diagnostic)
    observed_orders = np.log(errors[:-1] / errors[1:]) / np.log(np.array(h_values[:-1]) / np.array(h_values[1:]))

    print("Errors:", errors)
    print("Observed pairwise orders:", observed_orders)
    print("Fitted slope:", slope)

    # Reference O(h^2) line
    ref = errors[-1] * (h_values / h_values[-1])**2

    plt.figure(figsize=(8,5))
    plt.loglog(h_values, errors, 'bo-', label='PC GTE')
    plt.loglog(h_values, ref, 'k--', alpha=0.6, label='Theoretical $O(h^2)$')

    plt.title("Global Error Convergence")
    plt.xlabel("Step size h")
    plt.ylabel("Error at terminal time T")

    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.show()

    return slope

def analyze_local_error(f, y0, A, h_values):
    """
    Analyzes Local Error: error of a single step.
    Expected slope: 3.0 (O(h^3)).
    """
    errors = []
    t_n = 0.5 
    
    for h in h_values:
        t_nm1 = t_n - h
        t_np1 = t_n + h
        
        # Exact values for the current and previous states
        y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
        y_n = exact_solution_linear_system(t_n, A, y0)
        y_exact_np1 = exact_solution_linear_system(t_np1, A, y0)
        
        # Exact derivatives for the current and previous states
        f_n = f(t_n, y_n, A)
        f_nm1 = f(t_nm1, y_nm1, A)
        
        # Take a single step using the corrected signature
        # Signature: y_n, f_n, f_nm1, h, f, t_np1, *f_args
        y_num_np1, _, _ = predictor_corrector_step(y_n, f_n, f_nm1, h, f, t_np1, A)
        
        errors.append(np.linalg.norm(y_num_np1 - y_exact_np1))

    errors = np.array(errors)
    slope = np.polyfit(np.log(h_values), np.log(errors), 1)[0]

    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, errors, 'ro-', label=f'PC LTE')
    plt.loglog(h_values, (errors[-1]/h_values[-1]**3) * h_values**3, 'k--', alpha=0.5, label='Theoretical $O(h^3)$')
    plt.title("Local Truncation Error Convergence (LTE)")
    plt.xlabel("Step size h")
    plt.ylabel("Single step error")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()
    
    return slope