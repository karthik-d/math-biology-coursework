import matplotlib.pyplot as plt
import numpy as np


def given_pde_system(N=101, L=1.0, D=0.01, c=0.1, v0=1.0):
    """
    Define the PDE system:
        du/dt = D d2u/dx2 + v(x) - c u
    on domain x in (0,1)
    Boundary conditions:
        du/dx = 0 at x=0 (Neumann)
        u = 0 at x=1 (Dirichlet)
    Initial condition: u(x,0) = 0
    Source: v(x) = v0 for x in (0,0.1), 0 elsewhere
    
    Returns:
        x: spatial grid
        u0: initial condition array
        dx: spatial step
        v: source term array
        D: diffusion coefficient
        c: decay coefficient
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    
    # Initial condition
    u0 = np.zeros(N)
    
    # Source term
    v = np.zeros_like(x)
    v[x < 0.1] = v0
    
    return x, u0, dx, v, D, c


def plot_pde_solution(x, times, usol, overlay_times=None, overlay_x=None, cmap='viridis'):
	"""
	Plot PDE solution in three subplots:
	1. Heatmap of u(x,t)
	2. Spatial profiles at selected times
	3. Temporal trajectories at selected spatial locations

	Parameters
	----------
	x : array_like
		Spatial grid.
	times : array_like
		Time points.
	usol : 2D array
		Solution array of shape (nt, nx).
	overlay_times : list or array, optional
		Times at which to plot spatial profiles.
	overlay_x : list or array, optional
		Spatial locations at which to plot time trajectories.
	cmap : str
		Colormap for heatmap.
	"""
	fig, axes = plt.subplots(2, 2)

	# --------------------------------------
	# 1. Heatmap of u(x,t)
	# --------------------------------------
	ax = axes[0][0]
	extent = [x[0], x[-1], times[0], times[-1]]
	im = ax.imshow(usol, aspect='auto', origin='lower', extent=extent, cmap=cmap)
	fig.colorbar(im, ax=ax, label='u(x,t)')
	ax.set_xlabel('x')
	ax.set_ylabel('t')
	ax.set_title('Heatmap of u(x,t)')

	# --------------------------------------
	# 2. Spatial profiles at selected times
	# --------------------------------------
	ax = axes[0][1]
	if overlay_times is not None:
		for t_overlay in overlay_times:
			idx = np.argmin(np.abs(times - t_overlay))
			ax.plot(x, usol[idx], label=f't={times[idx]:.2f}', linewidth=2, alpha=0.5)
	ax.set_xlabel('x')
	ax.set_ylabel('u(x)')
	ax.set_title('Spatial profiles at select times')
	ax.grid(True)
	ax.legend()

	# --------------------------------------
	# 3. Temporal trajectories at selected spatial locations
	# --------------------------------------
	ax = axes[1][0]
	if overlay_x is not None:
		for x_val in overlay_x:
			idx = np.argmin(np.abs(x - x_val))
			ax.plot(times, usol[:, idx], label=f'x={x[idx]:.2f}', linewidth=2, alpha=0.8)
	ax.set_xlabel('t')
	ax.set_ylabel('u(t)')
	ax.set_title('Temporal trajectories at select x')
	ax.grid(True)
	ax.legend()

	# --------------------------------------
	# 4. Total mass over time
	# --------------------------------------
	mass = np.trapz(usol, x=x, axis=1)
	ax = axes[1][1]
	ax.plot(times, mass, linewidth=2)
	ax.set_xlabel('t')
	ax.set_ylabel('Total mass')
	ax.set_title('Total mass over time')
	ax.grid(True)

	plt.tight_layout()
	plt.show()


def spatial_convergence(solve_func, ref_usol, ref_x, N_values, L=1.0, D=0.01, c=0.1, v0=1.0, dt=1e-4, T=1.0):
    """
    Compute spatial convergence of MOL solver.
    
    Parameters
    ----------
    solve_func : callable
        Solver function: solve_func(N, L, D, c, v0, dt, T) -> x, usol, times
    ref_usol : array_like
        Reference solution evaluated at ref_x (highly resolved solution)
    ref_x : array_like
        Reference grid points
    N_values : list of int
        List of spatial resolutions to test
    dt : float
        Time step (small enough to neglect temporal error)
    T : float
        Final time
    """
    errors = []
    dx_vals = []
    
    for N in N_values:
        x, usol, times = solve_func(N=N, L=L, D=D, c=c, v0=v0, dt=dt, T=T)
        # Interpolate numerical solution to reference grid
        u_interp = np.interp(ref_x, x, usol[-1])
        error = np.linalg.norm(u_interp - ref_usol, ord=2) * np.sqrt(ref_x[1]-ref_x[0])
        errors.append(error)
        dx_vals.append(L / (N-1))
    
    # Log-log plot
    plt.figure(figsize=(6,4))
    plt.loglog(dx_vals, errors, 'o-', label='Numerical error')
    # slope=2 reference
    C = errors[0] / dx_vals[0]**2
    plt.loglog(dx_vals, C*np.array(dx_vals)**2, '--', label='slope=2')
    plt.xlabel('dx')
    plt.ylabel('L2 error')
    plt.title('Spatial convergence')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    

def temporal_convergence(solve_func, ref_usol, ref_x, dt_values, N=101, L=1.0, D=0.01, c=0.1, v0=1.0, T=1.0, ref_dt=1e-5):
    """
    Compute temporal convergence of MOL solver and plot local truncation error (LTE) at single step.
    
    LTE is computed by comparing a single step of size dt to a highly resolved reference step of size ref_dt.
    """
    global_errors = []
    lte_errors = []

    # Compute single-step reference solution (very small dt)
    x_ref, usol_ref, _ = solve_func(N=N, L=L, D=D, c=c, v0=v0, dt=ref_dt, T=dt_values[-1])
    u_ref_interp = np.interp(ref_x, x_ref, usol_ref[0])  # initial condition

    for dt in dt_values:
        # --- Global error over full simulation ---
        x_num, usol_num, _ = solve_func(N=N, L=L, D=D, c=c, v0=v0, dt=dt, T=T)
        u_num = np.interp(ref_x, x_num, usol_num[-1])
        error = np.linalg.norm(u_num - ref_usol, ord=2) * np.sqrt(ref_x[1]-ref_x[0])
        global_errors.append(error)

        # --- Single-step LTE estimate ---
        # Take a single step of size dt from the initial condition
        x_step, usol_step, _ = solve_func(N=N, L=L, D=D, c=c, v0=v0, dt=dt, T=dt)
        u_num_step = np.interp(ref_x, x_step, usol_step[-1])

        # Reference solution: integrate over the same interval using tiny dt (ref_dt)
        num_ref_steps = int(np.ceil(dt / ref_dt))
        x_ref_step, u_ref_step, _ = solve_func(N=N, L=L, D=D, c=c, v0=v0, dt=dt/num_ref_steps, T=dt)
        u_ref_step_interp = np.interp(ref_x, x_ref_step, u_ref_step[-1])

        # LTE: difference between coarse single step and reference single step
        lte = np.linalg.norm(u_num_step - u_ref_step_interp, ord=2) * np.sqrt(ref_x[1]-ref_x[0])
        lte_errors.append(lte)

    # Plot global error and LTE
    plt.figure(figsize=(6,4))
    plt.loglog(dt_values, global_errors, 'o-', label='Global error (final time)')
    plt.loglog(dt_values, lte_errors, 's-', label='Local truncation error (single step)')

    # Reference slopes
    C_global = global_errors[0] / dt_values[0]**2
    plt.loglog(dt_values, C_global*np.array(dt_values)**2, '--', label='slope=2 (global)')

    C_lte = lte_errors[0] / dt_values[0]**3
    plt.loglog(dt_values, C_lte*np.array(dt_values)**3, '--', label='slope=3 (LTE)')

    plt.xlabel('dt')
    plt.ylabel('L2 error')
    plt.title('Temporal convergence: Global vs LTE')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()