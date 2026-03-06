import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from pkg.ivp.solver import f_linear, solve_pc


def analyze_pc_order(t0=0.0, T=1.0, hs=None, A=None, y0=None, plot_filename=None):
	"""Global error vs h. Expected O(h^1). Wider h range."""
	if A is None:
		A = np.array([[-5.0, 3.0], [100.0, -301.0]])
	if y0 is None:
		y0 = np.array([52.29, 83.82], dtype=float)

	m = len(y0)
	if hs is None:
		hs = np.logspace(-5, -1, 20)  # h: 3e-6 to 0.1

	def rhs(t, y):
		return f_linear(t, y, A)

	errors = [np.zeros_like(hs) for _ in range(m)]

	for i, h in enumerate(hs):
		N = int(np.round((T - t0) / h))
		if N < 2:  # need at least 2 steps for 2-step method
			continue
		T_eff = t0 + N * h
		
		t_num, Y_num = solve_pc(f_linear, (t0, T_eff), y0, h, A)
		
		sol_ref = solve_ivp(rhs, (t0, T_eff), y0, method="Radau",
							rtol=1e-13, atol=1e-15)
		y_ref = sol_ref.y[:, -1]
		
		for j in range(m):
			errors[j][i] = np.abs(Y_num[-1, j] - y_ref[j])

	mask = np.isfinite(errors[0])
	hs_used = hs[mask]
	errors_used = [e[mask] for e in errors]

	plt.figure(figsize=(8, 6))
	colors, markers = ['blue', 'red'], ['o', 's']
	for j in range(m):
		plt.loglog(hs_used, errors_used[j], f"{markers[j]}-",
					color=colors[j], label=f"|e{j+1}(T)|")

	hs_ref = np.logspace(-5, -1, 200)
	plt.loglog(hs_ref, 1e-2 * hs_ref**1.0, "k--", lw=2, label="O(h¹)")

	plt.gca().invert_xaxis()
	plt.xlabel("h")
	plt.ylabel("|eᵢ(T)|")
	plt.title("Global Error vs h")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	if plot_filename:
		plt.savefig(plot_filename, dpi=300)
	plt.show()

	if m>1:
		return hs_used, errors_used[0], *[errors_used[1]] 
	else:
		return None


def analyze_true_lte(t0=0.0, hs=None, A=None, y0=None, plot_filename=None):
	"""True LTE vs h (1 step). Expected O(h^2)."""
	if A is None:
		A = np.array([[-5.0, 3.0], [100.0, -301.0]])
	if y0 is None:
		y0 = np.array([52.29, 83.82], dtype=float)

	m = len(y0)
	if hs is None:
		hs = np.logspace(-6, 0, 25)  # finer range

	ltes = [np.zeros_like(hs) for _ in range(m)]

	for i, h in enumerate(hs):
		t_nm1, t_n, t_np1 = t0 - h, t0, t0 + h
		
		y_nm1 = expm(A * t_nm1) @ y0
		y_n   = expm(A * t_n)   @ y0
		y_np1 = expm(A * t_np1) @ y0
		
		f_nm1 = A @ y_nm1
		f_n   = A @ y_n
		
		y_star = y_n + 0.5 * h * (3 * f_n - f_nm1)
		f_star = A @ y_star
		y_num  = y_n + 0.5 * h * (3 * f_n + f_star)
		
		tau = y_np1 - y_num
		for j in range(m):
			ltes[j][i] = np.abs(tau[j])

	plt.figure(figsize=(8, 6))
	colors, markers = ['blue', 'red'], ['o', 's']
	for j in range(m):
		plt.loglog(hs, ltes[j], f"{markers[j]}-",
					color=colors[j], label=f"|τ_{j+1}|")

	hs_ref = np.logspace(-6, 0, 200)
	plt.loglog(hs_ref, 1e-3 * hs_ref**2.0, "k--", lw=2, label="O(h²)")

	plt.gca().invert_xaxis()
	plt.xlabel("h")
	plt.ylabel("|τᵢ|")
	plt.title(f"LTE vs h (tₙ={t0})")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	if plot_filename:
		plt.savefig(plot_filename, dpi=300)
	plt.show()

	if m>1:
		return hs, ltes[0], *[ltes[1]]
	else:
		return None
