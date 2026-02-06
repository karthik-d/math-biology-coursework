import unittest
import numpy as np

from pkg.newton.newton import newton_system

class TestNewtonSystem(unittest.TestCase):
    """
    Unit tests for Newton's method for systems.
    """

    # ----------------------------
    # 1. Corner cases / edge cases
    # ----------------------------
    def test_zero_initial_guess(self):
        """Test F(x)=0 with zero initial guess."""
        F = lambda x: np.array([0.0, 0.0])
        J = lambda x: np.eye(2)
        x0 = np.array([0.0, 0.0])
        x, info = newton_system(F, x0, J)
        np.testing.assert_allclose(x, [0.0, 0.0])
        self.assertTrue(info["converged"])
        self.assertLessEqual(info["residual"], 1e-15)

    def test_singular_jacobian(self):
        """Test detection of singular Jacobian."""
        F = lambda x: np.array([x[0]**2, x[1]**2])
        J = lambda x: np.zeros((2,2))  # singular
        x0 = np.array([1.0, 1.0])
        x, info = newton_system(F, x0, J)
        self.assertFalse(info["converged"])
        self.assertEqual(info["reason"], "Jacobian singular")

    def test_nan_input(self):
        """Test behavior with NaN in initial guess."""
        F = lambda x: x
        J = lambda x: np.eye(2)
        x0 = np.array([np.nan, 0.0])
        x, info = newton_system(F, x0, J)
        self.assertTrue(np.isnan(x[0]))
        self.assertFalse(info["converged"])

    def test_empty_input(self):
        """Test empty input arrays."""
        F = lambda x: np.array([])
        J = lambda x: np.array([[]])
        x0 = np.array([])
        x, info = newton_system(F, x0, J)
        self.assertEqual(x.size, 0)
        self.assertTrue(info["converged"])

    # ----------------------------
    # 2. Linear systems
    # ----------------------------
    def test_linear_system(self):
        """Solve simple 2x2 linear system."""
        A = np.array([[2, 1],
                      [1, 3]])
        b = np.array([1, 2])
        F = lambda x: A @ x - b
        J = lambda x: A
        x0 = np.array([0.0, 0.0])
        x, info = newton_system(F, x0, J)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-12)
        self.assertTrue(info["converged"])

    def test_linear_system_nonzero_initial(self):
        """Linear system with nonzero initial guess."""
        A = np.array([[4, 2], [1, 3]])
        b = np.array([6, 5])
        F = lambda x: A @ x - b
        J = lambda x: A
        x0 = np.array([1.0, 1.0])
        x, info = newton_system(F, x0, J)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-12)
        self.assertTrue(info["converged"])

    # ----------------------------
    # 3. Non-linear systems
    # ----------------------------
    def test_simple_nonlinear_system(self):
        """Simple non-linear system: x^2 + y^2 = 1, x - y = 0"""
        F = lambda v: np.array([v[0]**2 + v[1]**2 - 1, v[0] - v[1]])
        J = lambda v: np.array([[2*v[0], 2*v[1]], [1, -1]])
        # Two solutions: (sqrt(0.5), sqrt(0.5)) and (-sqrt(0.5), -sqrt(0.5))
        sol1 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
        sol2 = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        # initial guess near first solution
        x, info = newton_system(F, x0=[0.7,0.7], J=J)
        np.testing.assert_allclose(x, sol1, rtol=1e-12)
        self.assertTrue(info["converged"])
        # initial guess near second solution
        x, info = newton_system(F, x0=[-0.7,-0.7], J=J)
        np.testing.assert_allclose(x, sol2, rtol=1e-12)
        self.assertTrue(info["converged"])

    def test_multiple_solutions_non_linear(self):
        """Non-linear system with multiple solutions: x^2 - 1 = 0, y^2 - 4 = 0"""
        F = lambda v: np.array([v[0]**2 - 1, v[1]**2 - 4])
        J = lambda v: np.array([[2*v[0], 0], [0, 2*v[1]]])
        sols = [np.array([1, 2]), np.array([1, -2]), np.array([-1, 2]), np.array([-1, -2])]
        for guess, sol_true in zip([[0.5,1],[0.5,-1],[-0.5,1],[-0.5,-1]], sols):
            x, info = newton_system(F, x0=guess, J=J)
            np.testing.assert_allclose(x, sol_true, rtol=1e-12)
            self.assertTrue(info["converged"])


if __name__ == "__main__":
    unittest.main()
