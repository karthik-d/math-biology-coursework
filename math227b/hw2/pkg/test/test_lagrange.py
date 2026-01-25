import unittest
import math

from lagrange.lagrange import interpolate, divided_difference as divided_difference, compute_polynomial
            

class TestDividedDifference(unittest.TestCase):

	def test_constant(self):
		x = [-1.0, 0.0, 2.0]
		f = [5.0, 5.0, 5.0]
		a = divided_difference(x, f)

		self.assertAlmostEqual(a[0], 5.0)
		self.assertAlmostEqual(a[1], 0.0)
		self.assertAlmostEqual(a[2], 0.0)

	def test_linear(self):
		x = [-1.0, 0.0, 1.0]
		f = [2*xi + 1 for xi in x]
		a = divided_difference(x, f)

		self.assertAlmostEqual(a[0], f[0])
		self.assertAlmostEqual(a[1], 2.0)
		self.assertAlmostEqual(a[2], 0.0)

	def test_quadratic(self):
		x = [0.0, 1.0, 2.0]
		f = [xi**2 for xi in x]
		a = divided_difference(x, f)

		self.assertAlmostEqual(a[0], 0.0)
		self.assertAlmostEqual(a[1], 1.0)
		self.assertAlmostEqual(a[2], 1.0)

	def test_cubic(self):
		x = [0.0, 1.0, 2.0, 3.0]
		f = [xi**3 for xi in x]
		a = divided_difference(x, f)

		self.assertAlmostEqual(a[0], 0.0)
		self.assertAlmostEqual(a[1], 1.0)
		self.assertAlmostEqual(a[2], 3.0)
		self.assertAlmostEqual(a[3], 1.0)
		
	def test_nonmonotone_nodes(self):
		x = [1.0, -1.0, 2.0]
		f = [xi**2 for xi in x]
		a = divided_difference(x, f)
		
		# Check first three divided differences
		# exact values can be computed manually or via formula
		self.assertAlmostEqual(a[0], 1.0)
			
	def test_random_quadratic(self):
		# P(x) = 3 - 2x + x^2
		x = [0.0, 1.0, 2.0]
		f = [3 - 2*xi + xi**2 for xi in x]
		a = divided_difference(x, f)
		
		# Manual computation:
		# f[x0] = 3, f[x0,x1] = (-1)/1 = -1, f[x0,x1,x2] = (1 - -1)/(2-0) = 1.0
		self.assertAlmostEqual(a[0], 3.0)
		self.assertAlmostEqual(a[1], -1.0)
		self.assertAlmostEqual(a[2], 1.0)


class TestNewtonEval(unittest.TestCase):

	def test_constant(self):
		a = [5.0]
		x_nodes = []

		for x in [-2.0, 0.0, 3.0]:
			y = compute_polynomial(a, x_nodes, x)
			self.assertAlmostEqual(y, 5.0)

	def test_linear(self):
		# P(x) = 2 + 3(x - 1)
		a = [2.0, 3.0]
		x_nodes = [1.0]

		tests = {0.0: -1.0, 1.0: 2.0, 2.0: 5.0}
		for x, expected in tests.items():
			y = compute_polynomial(a, x_nodes, x)
			self.assertAlmostEqual(y, expected)

	def test_quadratic_newton_form(self):
		# P(x) = 1 + 2(x-0) + 3(x-0)(x-1)
		a = [1.0, 2.0, 3.0]
		x_nodes = [0.0, 1.0]

		test_points = [0.0, 0.5, 2.0]
		for x in test_points:
			expected = 1 + 2*(x-0) + 3*(x-0)*(x-1)
			y = compute_polynomial(a, x_nodes, x)
			self.assertAlmostEqual(y, expected)
			
	def test_cubic_newton_form(self):
		# P(x) = 1 + 2(x-0) + 3(x-0)(x-1) + 4(x-0)(x-1)(x-2)
		a = [1.0, 2.0, 3.0, 4.0]
		x_nodes = [0.0, 1.0, 2.0]

		test_points = [-1.0, 0.0, 0.5, 1.5, 3.0]
		for x in test_points:
			expected = 1 + 2*(x-0) + 3*(x-0)*(x-1) + 4*(x-0)*(x-1)*(x-2)
			y = compute_polynomial(a, x_nodes, x)
			self.assertAlmostEqual(y, expected)
               
	def test_random_newton_polynomial(self):
		a = [1.0, -2.0, 0.5, 3.0]  # arbitrary coefficients
		x_nodes = [0.0, 1.0, -1.0]

		# evaluate at several points
		test_points = [-2.0, -0.5, 0.5, 2.0]
		for x in test_points:
			# manual computation of Newton form
			expected = a[0] + a[1]*(x-x_nodes[0]) + a[2]*(x-x_nodes[0])*(x-x_nodes[1]) \
					+ a[3]*(x-x_nodes[0])*(x-x_nodes[1])*(x-x_nodes[2])
			y = compute_polynomial(a, x_nodes, x)
			self.assertAlmostEqual(y, expected)
	



class TestInterpolate(unittest.TestCase):

	def setUp(self):
		self.n = 6  # fixed reasonable number of nodes

	def nodes(self, a, b):
		return [a + i*(b-a)/(self.n-1) for i in range(self.n)]

	def test_exact_quadratic(self):
		def f(x):
			return 3*x**2 + 2*x + 1

		a, b = -1.0, 1.0
		x_nodes = self.nodes(a, b)

		test_points = [-1.0, -0.5, 0.0, 0.7, 1.0]
		for x in test_points:
			p = interpolate(x_nodes, f, x)
			self.assertAlmostEqual(p, f(x), places=10)

	def test_exact_cubic(self):
		def f(x):
			return x**3

		a, b = -1.0, 2.0
		x_nodes = self.nodes(a, b)

		test_points = [-1.0, 0.0, 1.0, 1.5]
		for x in test_points:
			p = interpolate(x_nodes, f, x)
			self.assertAlmostEqual(p, f(x), places=10)

	def test_interpolates_data_points(self):
		x_nodes = [0.0, 1.0, 2.0]
		f = [1.0, 2.0, 5.0]

		a = divided_difference(x_nodes, f)
		for xi, fi in zip(x_nodes, f):
			y = compute_polynomial(a, x_nodes[:-1], xi)
			self.assertAlmostEqual(y, fi)


	def test_known_function(self):
		x_nodes = [-1.0, 0.0, 1.0]
		f = [xi**3 for xi in x_nodes]

		a = divided_difference(x_nodes, f)

		for x in [-0.5, 0.5, 2.0]:
			y = compute_polynomial(a, x_nodes[:-1], x)
		self.assertAlmostEqual(y, x**3)


	def test_random_polynomial_reconstruction(self):
		coeffs = [2.0, -1.0, 0.5]
		x_nodes = [0.0, 1.0, 2.0]

		f = [coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 for x in x_nodes]
		a = divided_difference(x_nodes, f)

		for x in [-1.0, 0.5, 3.0]:
			y = compute_polynomial(a, x_nodes[:-1], x)
			expected = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2
			self.assertAlmostEqual(y, expected)


	def test_symmetry_even_function(self):
		def f(x):
			return x**2

		a, b = -1.0, 1.0
		x_nodes = self.nodes(a, b)

		test_points = [0.2, 0.5, 0.8]
		for x in test_points:
			p_pos = interpolate(x_nodes, f, x)
			p_neg = interpolate(x_nodes, f, -x)
			self.assertAlmostEqual(p_pos, p_neg, places=10)


if __name__ == "__main__":
    unittest.main()
