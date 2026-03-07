import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Ensure these match your solver file naming/locations
from pkg.ivp.solver import (
    f_linear, 
    solve_adams_bashforth_predictor, 
    solve_predictor_corrector
)
from analysis.error import analyze_global_error, analyze_local_error

def exact_solution_linear_system(t, A, y0):
    """Exact solution y(t) = exp(A t) y0."""
    return expm(A * t) @ y0

def detect_instability(t, Y, y_ref, threshold=3.0):
    """Identify the point where the numerical solution diverges significantly."""
    norms_Y = np.linalg.norm(Y, axis=1)
    norms_ref = np.linalg.norm(y_ref, axis=1)
    return norms_Y <= threshold * norms_ref

def run_single_h(A, y0, h, t0=0.0, T=0.1):
    """Compute full solutions for a specific h value."""
    t_ab_full, Y_ab_full = solve_adams_bashforth_predictor(f_linear, (t0, T), y0, h, A)
    t_pc_full, Y_pc_full = solve_predictor_corrector(f_linear, (t0, T), y0, h, A)
    
    Y_ref_at_ab = np.array([exact_solution_linear_system(tk, A, y0) for tk in t_ab_full])
    Y_ref_at_pc = np.array([exact_solution_linear_system(tk, A, y0) for tk in t_pc_full])
    
    stable_ab_mask = detect_instability(t_ab_full, Y_ab_full, Y_ref_at_ab)
    is_ab_stable = np.all(stable_ab_mask)
    ab_onset = t_ab_full[~stable_ab_mask][0] if not is_ab_stable else None
    
    stable_pc_mask = detect_instability(t_pc_full, Y_pc_full, Y_ref_at_pc)
    is_pc_stable = np.all(stable_pc_mask)
    pc_onset = t_pc_full[~stable_pc_mask][0] if not is_pc_stable else None
    
    t_ref_smooth = np.linspace(t0, T, 500)
    Y_ref_smooth = np.array([exact_solution_linear_system(tk, A, y0) for tk in t_ref_smooth])
    
    return {
        'h': h,
        't_ab_full': t_ab_full, 'Y_ab_full': Y_ab_full, 'ab_onset': ab_onset, 'is_ab_stable': is_ab_stable,
        't_pc_full': t_pc_full, 'Y_pc_full': Y_pc_full, 'pc_onset': pc_onset, 'is_pc_stable': is_pc_stable,
        't_ref': t_ref_smooth, 'Y_ref': Y_ref_smooth
    }

def plot_single_h_2x2(res):
    """Generate 2x2: [y1 Pred, y1 PC], [y2 Pred, y2 PC] vs Exact."""
    fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    plot_map = [
        (0, 0, 0, 'ab', 'Predictor (AB2)'),
        (0, 1, 0, 'pc', 'Pred-Corr (PECE)'),
        (1, 0, 1, 'ab', 'Predictor (AB2)'),
        (1, 1, 1, 'pc', 'Pred-Corr (PECE)')
    ]
    
    colors = {'ab': '#228B22', 'pc': '#4169E1', 'exact': '#000000'}
    lw = 1.8  # Standard line width
    alpha = 0.7
    
    for row, col, comp, meth, meth_name in plot_map:
        ax = axes_grid[row, col]
        
        # Determine stability status for title
        is_stable = res[f'is_{meth}_stable']
        status_str = "[STABLE]" if is_stable else "[UNSTABLE]"
        
        # 1. Plot Exact (Dashed)
        ax.plot(res['t_ref'], res['Y_ref'][:, comp], color=colors['exact'], 
               ls='--', lw=lw, alpha=alpha, zorder=1, label="Exact")
        
        # 2. Plot Full Numerical Trajectory (Solid)
        t_full = res[f't_{meth}_full']
        Y_full = res[f'Y_{meth}_full']
        onset = res[f'{meth}_onset']
        
        # Clipping for plot readability
        Y_plot = np.clip(Y_full[:, comp], -1e5, 1e5)
        
        ax.plot(t_full, Y_plot, color=colors[meth], 
               ls='-', lw=lw, alpha=alpha, zorder=5, label=f"{meth_name}")
        
        # 3. Mark Onset of Instability (Extra Thick)
        if onset is not None:
            ax.axvline(onset, color='red', ls=':', lw=3.5, alpha=0.9, label="Divergence Onset")
        
        # Aesthetics
        var_name = f"y_{comp+1}"
        ax.set_title(f"{status_str} ${var_name}(t)$: {meth_name}", 
                     fontsize=13, fontweight='bold', color='red' if not is_stable else 'black')
        ax.set_ylabel(f"Value of ${var_name}$")
        if row == 1: ax.set_xlabel("Time $t$")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)
        
        # Y-limit scaling based on Exact reference
        y_min, y_max = np.min(res['Y_ref'][:, comp]), np.max(res['Y_ref'][:, comp])
        buffer = 0.5 * (y_max - y_min) if y_max != y_min else 1.0
        ax.set_ylim(y_min - buffer, y_max + buffer)

    fig.suptitle(f'Numerical Stability Comparison (h = {res["h"]})', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

if __name__ == "__main__":
	# System Definition
	A = np.array([[-5.0, 3.0], [100.0, -301.0]])
	y0 = np.array([52.29, 83.82])
	t_span = (0.0, 3)

	# --- Part 1: Stability Visualization (Original) ---
	# hs_cases = [0.001, 0.004, 0.01] 
	# for h in hs_cases:
	#     print(f"Analyzing stability for h={h}...")
	#     results = run_single_h(A, y0, h, T=t_span[1]) 
	#     fig = plot_single_h_2x2(results)
	#     plt.show()

	# System Definition
	A = np.array([[-5.0, 3.0], [100.0, -301.0]])
	y0 = np.array([52.29, 83.82])
	t_span = (0.0, 1)

	results = run_single_h(A, y0, h=0.001, T=1) 
	fig = plot_single_h_2x2(results)
	plt.show()

	# --- Part 2: Convergence/Error Analysis ---
	print("\nRunning Convergence Analysis...")

	# We choose h values within the stable region for the PC scheme
	# to accurately measure the convergence slope.
	h_convergence = np.logspace(-4.5, -2.5, 50) 

	# 1. Global Error Analysis (Expected Slope: 2)
	analyze_global_error(f_linear, t_span, y0, A, h_convergence)

	# 2. Local Truncation Error Analysis (Expected Slope: 3)
	analyze_local_error(f_linear, y0, A, h_convergence)
     

		# =============================================================================
	# 1. SCALAR EXPONENTIAL (Decay)
	# y' = -k*y  => Exact: y(t) = y0 * exp(-k*t)
	# =============================================================================
	def test_scalar_exponential():
		k = 2.0
		A_scalar = np.array([[-k]])
		y0_scalar = np.array([1.0])
		t_span = (0.0, 1.0)
		h_convergence = np.logspace(-4.5, -2.5, 50)
		
		print("--- Running Test: Scalar Exponential ---")
		analyze_global_error(f_linear, t_span, y0_scalar, A_scalar, h_convergence)
		analyze_local_error(f_linear, y0_scalar, A_scalar, h_convergence)


	# =============================================================================
	# 2. SIMPLE HARMONIC OSCILLATOR (Sinusoidal)
	# y1' = y2, y2' = -y1 => Exact: y1 = cos(t), y2 = -sin(t)
	# =============================================================================
	def test_harmonic_oscillator():
		# Matrix A for [y1' = y2; y2' = -y1]
		A_osc = np.array([[0.0, 1.0], 
						[-1.0, 0.0]])
		y0_osc = np.array([1.0, 0.0]) # cos(0)=1, sin(0)=0
		t_span = (0.0, 2.0 * np.pi)   # One full period
		h_convergence = np.logspace(-4.5, -2.5, 50)
		
		print("--- Running Test: Simple Harmonic Oscillator ---")
		analyze_global_error(f_linear, t_span, y0_osc, A_osc, h_convergence)
		analyze_local_error(f_linear, y0_osc, A_osc, h_convergence)


	# =============================================================================
	# 3. COUPLED DECAY (Moderate Eigenvalues)
	# =============================================================================
	def test_coupled_decay():
		# Non-stiff matrix with real, negative eigenvalues
		A_coupled = np.array([[-2.0, 0.5], 
							[0.1, -1.5]])
		y0_coupled = np.array([10.0, 5.0])
		t_span = (0.0, 0.5)
		h_convergence = np.logspace(-4.5, -2.5, 50)
		
		print("--- Running Test: Coupled Decay ---")
		analyze_global_error(f_linear, t_span, y0_coupled, A_coupled, h_convergence)
		analyze_local_error(f_linear, y0_coupled, A_coupled, h_convergence)

	def test_circular_orbit():
		# y1' = -y2, y2' = y1  => Solution: y1 = cos(t), y2 = sin(t)
		# This is a Hamiltonian system (conserves y1^2 + y2^2 = 1)
		A_orbit = np.array([[0.0, -1.0], 
							[1.0,  0.0]])
		y0_orbit = np.array([1.0, 0.0])
		t_span = (0.0, 10.0) # Integrate for several orbits
		h_convergence = np.logspace(-4.5, -2.5, 50)

		print("\n--- Running Test: Circular Orbit (Oscillatory) ---")
		analyze_global_error(f_linear, t_span, y0_orbit, A_orbit, h_convergence)
		analyze_local_error(f_linear, y0_orbit, A_orbit, h_convergence)
            
	def test_sequential_decay():
		# y1' = -k1*y1
		# y2' = k1*y1 - k2*y2
		# Matrix A for k1=2, k2=1
		A_chain = np.array([[-2.0,  0.0], 
							[ 2.0, -1.0]])
		y0_chain = np.array([1.0, 0.0])
		t_span = (0.0, 2.0)
		h_convergence = np.logspace(-4.5, -2.5, 50)
		
		print("\n--- Running Test: Sequential Decay (Chain) ---")
		analyze_global_error(f_linear, t_span, y0_chain, A_chain, h_convergence)
		analyze_local_error(f_linear, y0_chain, A_chain, h_convergence)
            

	if __name__ == "__main__":
		# You can uncomment these one by one to verify your solver's accuracy
		test_scalar_exponential()
		test_harmonic_oscillator()
		test_coupled_decay()
		test_circular_orbit()
		test_sequential_decay()