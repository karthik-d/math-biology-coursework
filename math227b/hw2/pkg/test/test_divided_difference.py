import unittest

from lagrange import divided_difference


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
        self.assertAlmostEqual(a[2], 0.5)

    def test_cubic(self):
        x = [0.0, 1.0, 2.0, 3.0]
        f = [xi**3 for xi in x]
        a = divided_difference(x, f)

        self.assertAlmostEqual(a[0], 0.0)
        self.assertAlmostEqual(a[1], 1.0)
        self.assertAlmostEqual(a[2], 1.5)
        self.assertAlmostEqual(a[3], 1.0)


if __name__ == "__main__":
    unittest.main()
