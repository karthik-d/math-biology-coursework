import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from pkg.ivp.solver import (
    f_linear, 
    solve_adams_bashforth_predictor, 
    solve_predictor_corrector,
    adams_bashforth_predictor_step,
    predictor_corrector_step
)


def compute_exact_final(A, t_final, y0):
    """Exact y(T) = exp(A T) y0."""
    return expm(A * t_final) @ y0


def analyze_global_error(f, t_span, y0, h_values, A, atol=1e-12):
    """
    Global error ||y(T)-y_num(T)|| vs h (log-log). Expects linear f(t,y,A).

    Plots + prints observed order via polyfit.
    """
    t0, T = t_span
    y0 = np.asarray(y0)
    errors_ab, errors_pc = [], []

    fig, ax = plt.subplots(figsize=(8, 6))
    
    for h in h_values:
        # AB predictor
        t_ab, Y_ab = solve_adams_bashforth_predictor(f, t_span, y0, h, A)
        y_ab_final = Y_ab[-1]
        
        # PC
        t_pc, Y_pc = solve_predictor_corrector(f, t_span, y0, h, A)
        y_pc_final = Y_pc[-1]
        
        # Exact at T
        y_exact_final = compute_exact_final(A, T, y0)
        
        errors_ab.append(np.linalg.norm(y_ab_final - y_exact_final))
        errors_pc.append(np.linalg.norm(y_pc_final - y_exact_final))

    errors_ab = np.array(errors_ab)
    errors_pc = np.array(errors_pc)

    # Plot
    ax.loglog(h_values, errors_ab, 'go-', linewidth=2, markersize=8, label='AB Predictor')
    ax.loglog(h_values, errors_pc, 'bs-', linewidth=2, markersize=8, label='PC (AB2-AM1)')
    
    # O(h^2) reference (scale to match first point)
    h_ref = np.geomspace(h_values.min(), h_values.max(), 100)
    scale_ab = errors_ab[0] / (h_values[0]**2)
    ax.loglog(h_ref, scale_ab * h_ref**2, 'k--', linewidth=2, label=r'$\mathcal{O}(h^2)$')

    ax.set_xlabel('Step size h')
    ax.set_ylabel('Global Error $\|y(T)-y_\text{num}(T)\|_2$')
    ax.set_title('Global Error Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Observed orders
    order_ab = np.polyfit(np.log(h_values), np.log(errors_ab), 1)[0]
    order_pc = np.polyfit(np.log(h_values), np.log(errors_pc), 1)[0]
    print(f"AB order: {order_ab:.2f}  |  PC order: {order_pc:.2f}")


def analyze_local_truncation_error(f, t_span, y0, h_values, A, n_step=5):
    """
    LTE analysis: exact y_{n-1}, y_n → one step → ||τ_{n+1}|| vs h.

    Plots + prints orders (expect ~3.0).
    """
    t0 = t_span[0]
    lte_ab, lte_pc = [], []

    fig, ax = plt.subplots(figsize=(8, 6))
    
    for h in h_values:
        # Exact points for LTE test
        t_nm1 = t0 + (n_step-1) * h
        t_n = t_nm1 + h
        t_np1 = t_n + h
        
        y_nm1 = compute_exact_final(A, t_nm1, y0)
        y_n = compute_exact_final(A, t_n, y0)
        y_np1_exact = compute_exact_final(A, t_np1, y0)
        
        # AB predictor LTE
        y_star = adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, A)
        lte_ab.append(np.linalg.norm(y_star - y_np1_exact))
        
        # PC LTE
        y_pc, _ = predictor_corrector_step(y_nm1, y_n, h, f, t_nm1, t_n, A)
        lte_pc.append(np.linalg.norm(y_pc - y_np1_exact))

    lte_ab = np.array(lte_ab)
    lte_pc = np.array(lte_pc)

    # Plot
    ax.loglog(h_values, lte_ab, 'go-', linewidth=2, markersize=8, label='AB Predictor')
    ax.loglog(h_values, lte_pc, 'bs-', linewidth=2, markersize=8, label='PC (AB2-AM1)')
    
    # O(h^3) reference
    h_ref = np.geomspace(h_values.min(), h_values.max(), 100)
    scale_pc = lte_pc[0] / (h_values[0]**3)
    ax.loglog(h_ref, scale_pc * h_ref**3, 'k--', linewidth=2, label=r'$\mathcal{O}(h^3)$')

    ax.set_xlabel('Step size h')
    ax.set_ylabel('Local Truncation Error $\|τ_{n+1}\|_2$')
    ax.set_title(f'L TE at Step {n_step} (Exact Input)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    order_lte_ab = np.polyfit(np.log(h_values), np.log(lte_ab), 1)[0]
    order_lte_pc = np.polyfit(np.log(h_values), np.log(lte_pc), 1)[0]
    print(f"LTE AB order: {order_lte_ab:.2f}  |  LTE PC order: {order_lte_pc:.2f}")


# Driver usage:
if __name__ == "__main__":
    A = np.array([[-1, 0], [0, -2]])  # Non-stiff
    y0 = [1.0, 1.0]
    h_vals = np.logspace(-4, -1, 10)
    
    analyze_global_error(f_linear, (0, 1), y0, h_vals, A)
    analyze_local_truncation_error(f_linear, (0, 1), y0, h_vals, A)
