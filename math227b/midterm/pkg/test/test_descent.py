import unittest
import numpy as np

from descent.steepest_descent import steepest_descent


class TestSteepestDescent(unittest.TestCase):

	def assertConverges(self, f, grad, x0, x_star, tol=1e-4):
		xmin, path = steepest_descent(f, grad, x0, tol=tol, maxiter=10000)
		print(xmin, x_star)
		self.assertTrue(np.linalg.norm(xmin - x_star) < tol,
						msg=f"Did not converge to expected minimizer from {x0}")
		self.assertTrue(np.linalg.norm(grad(xmin)) < tol,
						msg="Gradient not small at solution")
		self.assertTrue(len(path) > 1,
						msg="Path too short (no iterations?)")

	# ------------------------
	# 1. Convex quadratic
	# ------------------------
	def test_shifted_quadratic(self):
		"""Convex quadratic with unique minimizer (shifted, anisotropic)"""

		def f(x): 
			return (x[0] - 1)**2 + 5*(x[1] + 2)**2

		def grad(x):
			return np.array([2*(x[0] - 1), 10*(x[1] + 2)])

		x_star = np.array([1.0, -2.0])

		self.assertConverges(f, grad, np.array([3.0, 0.0]), x_star)
		self.assertConverges(f, grad, np.array([-2.0, 4.0]), x_star)

	# ------------------------
	# 2. Ill-conditioned quadratic
	# ------------------------
	def test_ill_conditioned_quadratic(self):
		"""Tests zig-zag behavior"""

		def f(x): return 100*x[0]**2 + x[1]**2
		def grad(x): return np.array([200*x[0], 2*x[1]])

		x_star = np.array([0.0, 0.0])

		self.assertConverges(f, grad, np.array([1.0, 1.0]), x_star)
		self.assertConverges(f, grad, np.array([-1.0, 2.0]), x_star)

	# ------------------------
	# 3. Flat function (edge case)
	# ------------------------
	def test_flat_function(self):
		"""Gradient is zero everywhere"""

		def f(x): return 1.0
		def grad(x): return np.array([0.0, 0.0])

		x0 = np.array([3.0, 4.0])
		xmin, path = steepest_descent(f, grad, x0)

		self.assertTrue(np.allclose(xmin, x0),
						msg="Should not move on flat function")
		self.assertEqual(len(path), 1,
							msg="Should stop immediately on zero gradient")

	# ------------------------
	# 4. Linear function (no minimizer)
	# ------------------------
	def test_linear_function(self):
		"""Tests behavior when no minimum exists"""

		def f(x): return x[0] + x[1]
		def grad(x): return np.array([1.0, 1.0])

		x0 = np.array([0.0, 0.0])
		xmin, path = steepest_descent(f, grad, x0, maxiter=50)

		# Should keep moving downhill
		self.assertTrue(f(xmin) < f(x0),
						msg="Function should decrease for linear case")
		
	# ------------------------------------------------
	# 5. Exponential bowl
	# ------------------------------------------------
	def test_exponential_bowl(self):
		"""Tests rapidly varying gradient and line search stability"""

		def f(x):
			return np.exp(x[0]**2 + x[1]**2) - 1.0

		def grad(x):
			factor = 2*np.exp(x[0]**2 + x[1]**2)
			return np.array([
				factor*x[0],
				factor*x[1]
			])

		x_star = np.array([0.0, 0.0])

		self.assertConverges(f, grad, np.array([0.5, 0.5]), x_star)
		self.assertConverges(f, grad, np.array([-1.0, 1.0]), x_star)

	# ------------------------------------------------
    # 6. Quartic convex function
    # ------------------------------------------------
	def test_quartic_convex(self):
		"""Tests nonlinear but convex objective"""

		def f(x):
			return x[0]**4 + x[1]**4 + x[0]**2 + x[1]**2

		def grad(x):
			return np.array([
				4*x[0]**3 + 2*x[0],
				4*x[1]**3 + 2*x[1]
			])

		x_star = np.array([0.0, 0.0])

		self.assertConverges(f, grad, np.array([1.0, 1.0]), x_star)
		self.assertConverges(f, grad, np.array([-1.5, 0.5]), x_star)

	# ------------------------------------------------
    # 7. Rotated quadratic
    # ------------------------------------------------
	def test_rotated_quadratic(self):
		"""Tests invariance under coordinate rotation"""

		def f(x):
			u = (x[0] + x[1]) / np.sqrt(2)
			v = (x[0] - x[1]) / np.sqrt(2)
			return 3*u**2 + v**2

		def grad(x):
			u = (x[0] + x[1]) / np.sqrt(2)
			v = (x[0] - x[1]) / np.sqrt(2)
			return np.array([
				3*u + v,
				3*u - v
			])

		x_star = np.array([0.0, 0.0])

		self.assertConverges(f, grad, np.array([2.0, 1.0]), x_star)
		self.assertConverges(f, grad, np.array([-1.0, 2.0]), x_star)

	# ------------------------------------------------
    # 8. Mixed polynomial-exponential
    # ------------------------------------------------
	def test_mixed_poly_exp(self):
		"""Tests mixed polynomial–exponential scaling"""

		def f(x):
			return x[0]**2 + x[1]**2 + np.exp(x[0]**2)

		def grad(x):
			return np.array([
				2*x[0] + 2*x[0]*np.exp(x[0]**2),
				2*x[1]
			])

		x_star = np.array([0.0, 0.0])

		self.assertConverges(f, grad, np.array([1.0, 1.0]), x_star)
		self.assertConverges(f, grad, np.array([-1.0, 2.0]), x_star)



if __name__ == "__main__":
    unittest.main()