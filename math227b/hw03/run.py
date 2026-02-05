import numpy as np

from pkg.newton.newton import newton_system


# ---- Run analysis for each function ----
if __name__ == "__main__":
	def F(v):
		x, y = v
		return np.array([
			x**2 + y**2 - 4,
			x - y
		])

	def J(v):
		x, y = v
		return np.array([
			[2*x, 2*y],
			[1, -1]
		])

	x0 = [1, 1]
	sol, info = newton_system(F, x0, J)
	print("solution:", sol)
	print(info)
