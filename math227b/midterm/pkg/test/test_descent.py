import unittest
import numpy as np

from descent.steepest_descent import steepest_descent


class TestSteepestDescent(unittest.TestCase):

    def assertConverges(self, f, grad, x0, x_star, tol=1e-4):
        xmin, path = steepest_descent(f, grad, x0, tol=tol, maxiter=10000)
        self.assertTrue(np.linalg.norm(xmin - x_star) < tol,
                        msg=f"Did not converge to expected minimizer from {x0}")
        self.assertTrue(np.linalg.norm(grad(xmin)) < tol,
                        msg="Gradient not small at solution")
        self.assertTrue(len(path) > 1,
                        msg="Path too short (no iterations?)")

    # ------------------------
    # 1. Convex quadratic
    # ------------------------
    def test_simple_quadratic(self):
        """Convex quadratic: unique minimizer"""

        def f(x): return x[0]**2 + x[1]**2
        def grad(x): return np.array([2*x[0], 2*x[1]])

        x_star = np.array([0.0, 0.0])

        self.assertConverges(f, grad, np.array([1.0, 1.0]), x_star)
        self.assertConverges(f, grad, np.array([-2.0, 3.0]), x_star)

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
    # 3. Nonconvex (double well)
    # ------------------------
    def test_double_well(self):
        """Tests convergence to nearest local minimum"""

        def f(x): return (x[0]**2 - 1)**2 + x[1]**2
        def grad(x):
            return np.array([4*x[0]*(x[0]**2 - 1), 2*x[1]])

        # Two local minima: (±1,0)
        self.assertConverges(f, grad, np.array([2.0, 1.0]), np.array([1.0, 0.0]))
        self.assertConverges(f, grad, np.array([-2.0, -1.0]), np.array([-1.0, 0.0]))

    # ------------------------
    # 4. Flat function (edge case)
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
    # 5. Linear function (no minimizer)
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

    # ------------------------
    # 7. Badly scaled nonlinear
    # ------------------------
    def test_badly_scaled(self):
        """Tests line search robustness"""

        def f(x): return 1e-3*x[0]**2 + 1e3*x[1]**2
        def grad(x): return np.array([2e-3*x[0], 2e3*x[1]])

        x_star = np.array([0.0, 0.0])

        self.assertConverges(f, grad, np.array([10.0, 1.0]), x_star)
        self.assertConverges(f, grad, np.array([-5.0, -2.0]), x_star)


if __name__ == "__main__":
    unittest.main()