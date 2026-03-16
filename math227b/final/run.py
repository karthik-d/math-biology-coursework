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
	# plot_pde_solution(x, times, usol, overlay_times=overlay_times, overlay_x=overlay_x)

	# Reference solution: use very fine dx and dt
	N_ref = 801      # fine spatial resolution
	dt_ref = 1e-5    # very small dt for stability
	T_ref = 1.0      # final time for reference solution
	x_ref, usol_ref, times_ref = solve_pde_system(
		pde_func=given_pde_system,
		N=N_ref, L=1.0, D=0.01, c=0.1, v0=1.0, dt=dt_ref, T=T_ref
	)
	ref_x = x_ref
	ref_usol = usol_ref[-1]  # final time solution as reference

	# Spatial convergence test
	# print("running spatial convergence test...")
	# dt_spatial = 1e-4  # small enough for largest N
	# N_values = np.linspace(51, 501, 20, dtype=int)  # increasing spatial resolution
	# T_spatial = 1.0
	# spatial_convergence(lambda N, **kwargs: solve_pde_system(pde_func=given_pde_system, N=N, **kwargs),
	# 					ref_usol, ref_x, N_values, L=1.0, D=0.01, c=0.1, v0=1.0, dt=dt_spatial, T=T_spatial)

	# Temporal convergence test
	print("running temporal convergence test...")
	N_temporal = 801        # fine enough spatial grid
	T_temporal = 1.0
	dt_values = [7e-5, 5e-5, 2.5e-5, 1.25e-5]  # decreasing, all below dt_max ≈ 3.1e-4
	dt_values = np.linspace(7e-5, 1.25e-5, 50)  # 4 values from 7e-5 to 1.25e-5
	temporal_convergence(
		lambda dt, **kwargs: solve_pde_system(
			pde_func=given_pde_system, dt=dt, **kwargs
		),
		ref_usol,      # reference solution (final time)
		ref_x,
		dt_values,
		N=N_temporal,
		L=1.0, D=0.01, c=0.1, v0=1.0,
		T=T_temporal
	)