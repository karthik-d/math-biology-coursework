import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from pkg.mol.solver import solve_pde_system
from pkg.mol.utils import given_pde_system, plot_pde_solution


if __name__ == "__main__":

	# Solve the PDE
	T = 100.0  # final time
	x, usol, times = solve_pde_system(
		pde_func=given_pde_system,
		N=101, L=1.0, D=0.01, c=0.1, v0=1.0, dt=0.001, T=T
	)

	# Assume x, usol, times are obtained from your solver
	overlay_times = np.linspace(0, T, 9)			# times at which to overlay u(x)
	overlay_x = [0.0, 0.25, 0.5, 0.75, 1.0]			# spatial locations to plot time trajectories
	plot_pde_solution(x, times, usol, overlay_times=overlay_times, overlay_x=overlay_x)