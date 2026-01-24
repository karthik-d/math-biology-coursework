import unittest

from lagrange import compute_polynomial


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


if __name__ == "__main__":
    unittest.main()
