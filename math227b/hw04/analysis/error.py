import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pkg.ivp.solver import solve_predictor_corrector, predictor_corrector_step


def exact_solution_linear_system(t, A, y0):
    """
    Computes exact solution y(t) = exp(At)y0.
    """
    return expm(A * t) @ y0

def analyze_global_error(f, t_span, y0, A, h_values):
    """
    Analyzes Global Error: error at final T vs step size h.
    Expected slope: 2.0 (O(h^2)) for AB2 and PC schemes.
    """
    t0, T = t_span
    errors = []
    y_exact_T = exact_solution_linear_system(T, A, y0)

    for h in h_values:
        # Using the Predictor-Corrector for the analysis
        t, Y = solve_predictor_corrector(f, (t0, T), y0, h, A)
        # Compute L2 norm of the difference at final time
        errors.append(np.linalg.norm(Y[-1] - y_exact_T))

    errors = np.array(errors)
    slope = np.polyfit(np.log(h_values), np.log(errors), 1)[0]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, errors, 'bo-', label=f'Observed (slope={slope:.2f})')
    # Reference line for O(h^2)
    plt.loglog(h_values, (errors[-1]/h_values[-1]**2) * h_values**2, 'k--', alpha=0.5, label='Reference $O(h^2)$')
    
    plt.title("Global Error Convergence (GTE)")
    plt.xlabel("Step size h")
    plt.ylabel("Error at terminal time T")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()
    return slope

def analyze_local_error(f, y0, A, h_values):
    """
    Analyzes Local Error: error of a single step vs h.
    Expected slope: 3.0 (O(h^3)) for AB2/Trapezoidal LTE.
    """
    errors = []
    # Arbitrary point in time to test a single step
    t_n = 0.5 
    
    for h in h_values:
        t_nm1 = t_n - h
        t_np1 = t_n + h
        
        # Assume perfect previous values from exact solution
        y_nm1 = exact_solution_linear_system(t_nm1, A, y0)
        y_n = exact_solution_linear_system(t_n, A, y0)
        y_exact_np1 = exact_solution_linear_system(t_np1, A, y0)
        
        # Take a single predictor-corrector step
        y_num_np1, _ = predictor_corrector_step(y_nm1, y_n, h, f, t_nm1, t_n, A)
        
        errors.append(np.linalg.norm(y_num_np1 - y_exact_np1))

    errors = np.array(errors)
    slope = np.polyfit(np.log(h_values), np.log(errors), 1)[0]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, errors, 'ro-', label=f'Observed (slope={slope:.2f})')
    # Reference line for O(h^3)
    plt.loglog(h_values, (errors[-1]/h_values[-1]**3) * h_values**3, 'k--', alpha=0.5, label='Reference $O(h^3)$')
    
    plt.title("Local Truncation Error Convergence (LTE)")
    plt.xlabel("Step size h")
    plt.ylabel("Single step error")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()
    return slope