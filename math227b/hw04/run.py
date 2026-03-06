import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# Import from your module (adjust path as needed)
from pkg.ivp.solver import (
    f_linear, 
    solve_adams_bashforth_predictor, 
    solve_predictor_corrector
)
from analysis.error import analyze_global_error, analyze_local_truncation_error


def exact_solution_linear_system(t, A, y0):
    """
    Exact solution y(t) = exp(A t) y0 for y' = A y.
    """
    At = A * t
    return expm(At) @ y0


def solve_example_system_comparison():
    """
    Solve stiff linear system with both methods, plot each variable in subplots.
    """
    A = np.array([[-5.0,   3.0],
                  [100.0, -301.0]])
    y0 = np.array([52.29, 83.82])
    
    t0, T = 0.0, 0.1
    h = 1e-3
    
    # Adams-Bashforth predictor-only
    t_ab, Y_ab = solve_adams_bashforth_predictor(f_linear, (t0, T), y0, h, A)
    
    # Predictor-corrector (full scheme)
    t_pc, Y_pc = solve_predictor_corrector(f_linear, (t0, T), y0, h, A)
    
    # Exact reference
    Y_ref = np.zeros((len(t_ab), 2))
    for k, tk in enumerate(t_ab):
        Y_ref[k] = exact_solution_linear_system(tk, A, y0)
    
    # SciPy RK45 reference (added back for completeness)
    def f_ivp(t, y):
        return f_linear(t, y, A)
    sol = solve_ivp(f_ivp, (t0, T), y0, method="RK45", t_eval=t_ab,
                    rtol=1e-9, atol=1e-12)
    Y_rk45 = sol.y.T
    
    # Create subplots: one for each ODE variable
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # y1 subplot
    ax1.plot(t_ab, Y_ab[:, 0], "g-", linewidth=2, label="AB predictor")
    ax1.plot(t_pc, Y_pc[:, 0], "b-", linewidth=2, label="PC full")
    ax1.plot(t_ab, Y_ref[:, 0], "k:", linewidth=2.5, label="Exact")
    ax1.plot(t_ab, Y_rk45[:, 0], "m-.", linewidth=2, label="RK45")
    ax1.set_ylabel("y₁(t)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Stiff Linear System: y' = A y")
    
    # y2 subplot
    ax2.plot(t_ab, Y_ab[:, 1], "g-", linewidth=2, label="AB predictor")
    ax2.plot(t_pc, Y_pc[:, 1], "b-", linewidth=2, label="PC full")
    ax2.plot(t_ab, Y_ref[:, 1], "k:", linewidth=2.5, label="Exact")
    ax2.plot(t_ab, Y_rk45[:, 1], "m-.", linewidth=2, label="RK45")
    ax2.set_ylabel("y₂(t)")
    ax2.set_xlabel("t")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
	solve_example_system_comparison()

	# B. Error analysis
	A = np.array([[-1, 0], [0, -2]])  # Non-stiff test
	y0 = [1.0, 1.0]
	h_vals = np.logspace(-5, 0, 20)

	analyze_global_error(f_linear, (0, 1), y0, h_vals, A)
	analyze_local_truncation_error(f_linear, (0, 1), y0, h_vals, A, n_step=5)
