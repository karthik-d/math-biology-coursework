import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Ensure these match your solver file naming/locations
from pkg.ivp.solver import (
    f_linear, 
    solve_adams_bashforth_predictor, 
    solve_predictor_corrector
)
from analysis.error import analyze_errors, local_error_heatmap

def exact_solution_linear_system(t, A, y0):
    """Exact solution y(t) = exp(A t) y0."""
    return expm(A * t) @ y0

def detect_instability(t, Y, y_ref, threshold=3.0):
    """Identify the point where the numerical solution diverges significantly."""
    norms_Y = np.linalg.norm(Y, axis=1)
    norms_ref = np.linalg.norm(y_ref, axis=1)
    return norms_Y <= threshold * norms_ref

def solve_and_plot_trajectories(A, y0, system_name="System", t_span=(0.0, 1), f=f_linear):
	"""
	Solve the system using predictor-only (AB2) and predictor-corrector (PECE) for multiple h values.
	Generates two figures: Predictor-only and Predictor-Corrector.
	Each figure: 2 rows (system components) x 5 columns (different h values)
	Exact solution overlaid. Stable/unstable status in figure title.
	"""

	h_values = np.logspace(-0.5, -2, 5)  # 5 h values
	n_comp = len(y0)
	results = {'ab': [], 'pc': []}

	# --- Solve for all h values ---
	for h in h_values:
		t0, T = t_span
		
		# Predictor-only (AB2)
		t_ab, Y_ab = solve_adams_bashforth_predictor(f, t_span, y0, h, A)
		Y_ref_ab = np.array([exact_solution_linear_system(tk, A, y0) for tk in t_ab])
		stable_mask_ab = detect_instability(t_ab, Y_ab, Y_ref_ab)
		ab_onset = t_ab[~stable_mask_ab][0] if not np.all(stable_mask_ab) else None
		results['ab'].append({'h': h, 't': t_ab, 'Y': Y_ab, 'Y_ref': Y_ref_ab, 'onset': ab_onset, 'stable': np.all(stable_mask_ab)})
		
		# Predictor-Corrector (PECE)
		t_pc, Y_pc = solve_predictor_corrector(f, t_span, y0, h, A)
		Y_ref_pc = np.array([exact_solution_linear_system(tk, A, y0) for tk in t_pc])
		stable_mask_pc = detect_instability(t_pc, Y_pc, Y_ref_pc)
		pc_onset = t_pc[~stable_mask_pc][0] if not np.all(stable_mask_pc) else None
		results['pc'].append({'h': h, 't': t_pc, 'Y': Y_pc, 'Y_ref': Y_ref_pc, 'onset': pc_onset, 'stable': np.all(stable_mask_pc)})

	def plot_grid(res_list, method_name, color):
		fig, axes = plt.subplots(n_comp, len(h_values), figsize=(4*len(h_values), 3*n_comp), sharex=False, sharey=False)
		if n_comp == 1: axes = [axes]
		if len(h_values) == 1: axes = np.array([axes]).T
		
		for i in range(n_comp):
			for j, res in enumerate(res_list):
				ax = axes[i, j]
				h = res['h']
				t, Y, Y_ref, onset, stable = res['t'], res['Y'], res['Y_ref'], res['onset'], res['stable']
				
				# Exact solution
				ax.plot(t, Y_ref[:, i], 'k--', lw=1.5, label='Exact')
				# Numerical trajectory as scatter
				ax.scatter(t, Y[:, i], color=color, s=10, alpha=1, label=f"Numerical")
				# Instability onset
				if onset is not None:
					ax.axvline(onset, color='red', ls=':', lw=2, alpha=0.8, label='Divergence')
				
				ax.set_title(f"h={h:.0e}", fontsize=10)
				if i == n_comp-1: ax.set_xlabel("t")
				if j == 0: ax.set_ylabel(f"y_{i+1}")
				ax.grid(True, alpha=0.2)
				if i == 0 and j == 0: ax.legend(fontsize=8)
		
		# Overall figure title inside figure
		status_str = "[STABLE]" if all(r['stable'] for r in res_list) else "[UNSTABLE]"
		fig.suptitle(f"{method_name} Trajectories for {system_name} {status_str}", fontsize=12, fontweight='bold')
		
		# Reduce whitespace to show title
		fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at top for suptitle
		return fig

	# --- Generate figures ---
	fig_ab = plot_grid(results['ab'], "Predictor Only (AB2)", color='#228B22')
	fig_pc = plot_grid(results['pc'], "Predictor-Corrector (PECE)", color='#4169E1')

	plt.show()
	return fig_ab, fig_pc


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
	
	solve_and_plot_trajectories(A_osc, y0_osc, t_span=t_span, f=f_linear, system_name="Harmonic Oscillator")
	print("--- Running Test: Simple Harmonic Oscillator ---")
	analyze_errors(f_linear, t_span, y0_osc, A_osc, h_convergence)
	t_vals = np.linspace(0.1, 1.0, 50)  # time points for LTE evaluation
	h_vals = np.logspace(-3, -1, 10)     # step sizes
	LTE_mat = local_error_heatmap(f_linear, y0_osc, A_osc, h_vals, t_vals)


# =============================================================================
# 3. COUPLED DECAY (Moderate Eigenvalues)
# =============================================================================
def test_coupled_decay():
	# Non-stiff matrix with real, negative eigenvalues
	A_coupled = np.array([[-2.0, 0.5], 
						[0.1, -1.5]])
	y0_coupled = np.array([10.0, 5.0])
	t_span = (0.0, 1.0)
	h_convergence = np.logspace(-4.5, -2.5, 50)

	solve_and_plot_trajectories(A_coupled, y0_coupled, t_span=t_span, f=f_linear, system_name="Coupled Decay")
	print("--- Running Test: Coupled Decay ---")
	analyze_errors(f_linear, t_span, y0_coupled, A_coupled, h_convergence)
	t_vals = np.linspace(0.1, 1.0, 50)  # time points for LTE evaluation
	h_vals = np.logspace(-3, -1, 10)     # step sizes
	LTE_mat = local_error_heatmap(f_linear, y0_coupled, A_coupled, h_vals, t_vals)


def test_circular_orbit():
	# y1' = -y2, y2' = y1  => Solution: y1 = cos(t), y2 = sin(t)
	# This is a Hamiltonian system (conserves y1^2 + y2^2 = 1)
	A_orbit = np.array([[0.0, -1.0], 
						[1.0,  0.0]])
	y0_orbit = np.array([1.0, 0.0])
	t_span = (0.0, 10.0) # Integrate for several orbits
	h_convergence = np.logspace(-4.5, -2.5, 50)

	solve_and_plot_trajectories(A_orbit, y0_orbit, t_span, f_linear)
	print("\n--- Running Test: Circular Orbit (Oscillatory) ---")
	analyze_errors(f_linear, t_span, y0_orbit, A_orbit, h_convergence)
	t_vals = np.linspace(0.1, 1.0, 50)  # time points for LTE evaluation
	h_vals = np.logspace(-3, -1, 10)     # step sizes
	LTE_mat = local_error_heatmap(f_linear, y0, A_orbit, h_vals, t_vals)
      

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
	# A = np.array([[-5.0, 3.0], [100.0, -301.0]])
	# y0 = np.array([52.29, 83.82])
	# t_span = (0.0, 1)

	# results = run_single_h(A, y0, h=0.001, T=1) 
	# fig = plot_single_h_2x2(results)
	# plt.show()

	# # --- Part 2: Convergence/Error Analysis ---
	# print("\nRunning Convergence Analysis...")

	# # We choose h values within the stable region for the PC scheme
	# # to accurately measure the convergence slope.
	# h_convergence = np.logspace(-4.5, -2.5, 50) 

	# ## Error Analysis (Expected Slope: 2)
	# t_vals = np.linspace(0.1, 1.0, 50)  # time points for LTE evaluation
	# h_vals = np.logspace(-3, -1, 10)     # step sizes
	# LTE_mat = local_error_heatmap(f_linear, y0, A, h_vals, t_vals)
	# analyze_errors(f_linear, t_span, y0, A, h_convergence)
			
	## ADDITIONAL TESTS.
	test_harmonic_oscillator()
	test_coupled_decay()
	test_circular_orbit()