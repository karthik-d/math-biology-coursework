import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from pkg.mol.solver import solve_pde_system
from pkg.mol.utils import given_pde_system, plot_pde_solution, temporal_convergence, spatial_convergence


if __name__ == "__main__":

	# # Solve the PDE
	# T = 100.0  # final time
	# x, usol, times = solve_pde_system(
	# 	pde_func=given_pde_system,
	# 	N=101, L=1.0, D=0.01, c=0.1, v0=1.0, dt=0.001, T=T
	# )
	# # Assume x, usol, times are obtained from solver
	# overlay_times = np.linspace(0, T, 9)
	# overlay_x = [0.0, 0.25, 0.5, 0.75, 1.0]
	# plot_pde_solution(x, times, usol, overlay_times=overlay_times, overlay_x=overlay_x, title='Solution with Given IC')

	# # Case 1: Zero initial condition everywhere.
	# T = 100.0  # final time
	# x, usol, times = solve_pde_system(
	# 	pde_func=given_pde_system,
	# 	N=101, L=1.0, D=0.01, c=0.1, v0=1.0, dt=0.001, T=T,
	# 	u0_func=lambda x: np.zeros_like(x)  # IC: zero everywhere
	# )
	# overlay_times = np.linspace(0, T, 9)
	# overlay_x = [0.0, 0.25, 0.5, 0.75, 1.0]
	# plot_pde_solution(x, times, usol, overlay_times=overlay_times, overlay_x=overlay_x, title='Solution with Zero IC')

	# # Case 2: Linear initial condition.
	# T = 100.0  # final time
	# x, usol, times = solve_pde_system(
	# 	pde_func=given_pde_system,
	# 	N=101, L=1.0, D=0.01, c=0.1, v0=1.0, dt=0.001, T=T,
	# 	u0_func=lambda x: 1.0 - x  # IC: linear from 1 at x=0 to 0 at x=1
	# )
	# overlay_times = np.linspace(0, T, 9)
	# overlay_x = [0.0, 0.25, 0.5, 0.75, 1.0]
	# plot_pde_solution(x, times, usol, overlay_times=overlay_times, overlay_x=overlay_x, title='Solution with Linear IC')

	# # Case 3: Localized Gaussian bump.
	# T = 100.0  # final time
	# x, usol, times = solve_pde_system(
	# 	pde_func=given_pde_system,
	# 	N=101, L=1.0, D=0.01, c=0.1, v0=1.0, dt=0.001, T=T,
	# 	u0_func=lambda x: np.exp(-100*(x-0.5)**2)  # IC: localized bump at center
	# )
	# overlay_times = np.linspace(0, T, 9)
	# overlay_x = [0.0, 0.25, 0.5, 0.75, 1.0]
	# plot_pde_solution(x, times, usol, overlay_times=overlay_times, overlay_x=overlay_x, title='Solution with Localized Gaussian Bump')


	# # # Reference solution
	# N_ref = 1001      # fine spatial resolution
	# dt_ref = 1e-5    # very small dt for stability
	# T_ref = 0.2      # final time for reference solution
	# x_ref, usol_ref, times_ref = solve_pde_system(
	# 	pde_func=given_pde_system,
	# 	N=N_ref, L=1.0, D=0.01, c=0.1, v0=1.0, dt=dt_ref, T=T_ref
	# )
	# ref_x = x_ref
	# ref_usol = usol_ref[-1]  # final time solution as reference

	# # Spatial convergence test
	# print("running spatial convergence test...")
	# ref_x = np.linspace(0, 1, 501)  # grid to interpolate errors onto
	# dt_spatial = 1e-5  # small enough for largest N
	# N_values = np.linspace(201, 501, 20, dtype=int)  # increasing spatial resolution
	# T_spatial = T_ref
	# spatial_convergence(
    # lambda N: solve_pde_system(
    #     pde_func=given_pde_system,
    #     N=N,
    #     L=1.0,
    #     D=0.01,
    #     c=0.1,
    #     v0=1.0,
    #     dt=dt_spatial,
    #     T=T_spatial
    # ),
    # ref_x,
    # N_values
	# )

	# # Temporal convergence test
	# print("running temporal convergence test...")
	# N_temporal = 601        # fine enough spatial grid
	# T_temporal = T_ref
	# # dt_values = [7e-5, 5e-5, 2.5e-5, 1.25e-5]  # decreasing, all below dt_max ≈ 3.1e-4
	# dt_values = np.linspace(7e-5, 1.25e-4, 50)  # 4 values from 7e-5 to 1.25e-5
	# temporal_convergence(
	# 	lambda dt, **kwargs: solve_pde_system(
	# 		pde_func=given_pde_system, dt=dt, **kwargs
	# 	),
	# 	ref_usol,      # reference solution (final time)
	# 	ref_x,
	# 	dt_values,
	# 	N=N_temporal,
	# 	L=1.0, D=0.01, c=0.1, v0=1.0,
	# 	T=T_temporal
	# )

	# Define IC functions
	ic_zero = lambda x: np.zeros_like(x)
	ic_linear = lambda x: 1.0 - x
	ic_bump = lambda x: np.exp(-100*(x-0.5)**2)

	IC_cases = [
		("Zero IC", ic_zero),
		("Linear IC", ic_linear),
		("Localized Gaussian Bump", ic_bump)
	]

	T = 1.0  # final time for simulations
	for ic_name, u0_func in IC_cases:
		print(f"\n=== Running simulation for {ic_name} ===")
		
		# Solve PDE
		# x, usol, times = solve_pde_system(
		# 	pde_func=given_pde_system,
		# 	N=101, L=1.0, D=0.01, c=0.1, v0=1.0, dt=0.001, T=T,
		# 	u0_func=u0_func
		# )
		
		# overlay_times = np.linspace(0, T, 9)
		# overlay_x = [0.0, 0.25, 0.5, 0.75, 1.0]
		# plot_pde_solution(
		# 	x, times, usol,
		# 	overlay_times=overlay_times,
		# 	overlay_x=overlay_x,
		# 	title=f'Solution with {ic_name}'
		# )
		
		# -----------------------
		# Reference solution for convergence tests
		# -----------------------
		N_ref = 1001
		dt_ref = 1e-5
		T_ref = 0.2  # shorter time for convergence study
		x_ref, usol_ref, times_ref = solve_pde_system(
			pde_func=given_pde_system,
			N=N_ref, L=1.0, D=0.01, c=0.1, v0=1.0, dt=dt_ref, T=T_ref,
			u0_func=u0_func
		)
		ref_x = x_ref
		ref_usol = usol_ref[-1]  # final time solution as reference

		# -----------------------
		# Spatial convergence test
		# -----------------------
		print(f"Running spatial convergence test for {ic_name}...")
		ref_x_interp = np.linspace(0, 1, 501)  # grid to interpolate errors onto
		dt_spatial = 1e-5  # small enough to minimize temporal error
		N_values = np.linspace(201, 501, 20, dtype=int)  # increasing spatial resolution
		T_spatial = T_ref
		spatial_convergence(
			lambda N: solve_pde_system(
				pde_func=given_pde_system,
				N=N,
				L=1.0,
				D=0.01,
				c=0.1,
				v0=1.0,
				dt=dt_spatial,
				T=T_spatial,
				u0_func=u0_func
			),
			ref_x_interp,
			N_values,
			T=T_spatial, 
			title=ic_name
		)

		# -----------------------
		# Temporal convergence test
		# -----------------------
		print(f"Running temporal convergence test for {ic_name}...")
		N_temporal = 601  # fine enough spatial grid
		T_temporal = T_ref
		dt_values = np.linspace(7e-5, 1.25e-4, 50)
		temporal_convergence(
			lambda dt: solve_pde_system(
				pde_func=given_pde_system, dt=dt,
				N=N_temporal, L=1.0, D=0.01, c=0.1, v0=1.0,
				T=T_temporal, u0_func=u0_func  # bind IC here
			),
			ref_usol,
			ref_x,
			dt_values, 
			title=ic_name
		)