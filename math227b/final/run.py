# main.py
import matplotlib.pyplot as plt
from pkg.mol.solver import solve_pde_system
from pkg.mol.utils import given_pde_system


if __name__ == "__main__":
	N = 101
	L = 1.0
	D = 0.01
	c = 0.1
	v0 = 1.0
	dt = 0.01
	T = 1.0

	# Solve the PDE
	x, usol, times = solve_pde_system(
		pde_func=given_pde_system,
		N=N, L=L, D=D, c=c, v0=v0, dt=dt, T=T
	)

	# Plot the final solution
	plt.plot(x, usol[-1], label=f't={T}')
	plt.xlabel('x')
	plt.ylabel('u(x,t)')
	plt.title('Method of Lines solution with RK2')
	plt.legend()
	plt.grid(True)
	plt.show()