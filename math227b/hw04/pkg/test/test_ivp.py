import unittest
import numpy as np
from scipy.linalg import expm

from ivp.solver import solve_predictor_corrector, f_linear


# ------------------------------------------------------------
# Exact solution helper
# ------------------------------------------------------------
def exact_solution_linear_system(t, A, y0):
    return expm(A * t) @ y0


# ------------------------------------------------------------
# Helper for trajectory accuracy checks
# ------------------------------------------------------------
def check_solution_accuracy(testcase, f, t_span, y0, A, h, tol):

    t, Y = solve_predictor_corrector(f, t_span, y0, h, A)

    for i, ti in enumerate(t):

        y_exact = exact_solution_linear_system(ti, A, y0)
        err = np.linalg.norm(Y[i] - y_exact)

        testcase.assertLess(
            err,
            tol,
            msg=f"Error too large at step {i}, t={ti}, error={err}"
        )


class TestPredictorCorrectorSolver(unittest.TestCase):

    TOL = 5e-4
    H = 1e-3

    # ============================================================
    # 1. SCALAR EXPONENTIAL DECAY
    # ============================================================
    def test_scalar_exponential(self):

        k = 2.0
        A_scalar = np.array([[-k]])
        y0_scalar = np.array([1.0])
        t_span = (0.0, 1.0)

        print("\n--- Running Test: Scalar Exponential ---")

        check_solution_accuracy(
            self,
            f_linear,
            t_span,
            y0_scalar,
            A_scalar,
            self.H,
            self.TOL
        )


    # ============================================================
    # 2. SIMPLE HARMONIC OSCILLATOR
    # ============================================================
    def test_harmonic_oscillator(self):

        A_osc = np.array([[0.0, 1.0],
                          [-1.0, 0.0]])

        y0_osc = np.array([1.0, 0.0])
        t_span = (0.0, 2*np.pi)

        print("\n--- Running Test: Harmonic Oscillator ---")

        check_solution_accuracy(
            self,
            f_linear,
            t_span,
            y0_osc,
            A_osc,
            self.H,
            self.TOL
        )


    # ============================================================
    # 3. COUPLED DECAY
    # ============================================================
    def test_coupled_decay(self):

        A_coupled = np.array([[-2.0, 0.5],
                              [0.1, -1.5]])

        y0 = np.array([10.0, 5.0])
        t_span = (0.0, 0.5)

        print("\n--- Running Test: Coupled Decay ---")

        check_solution_accuracy(
            self,
            f_linear,
            t_span,
            y0,
            A_coupled,
            self.H,
            self.TOL
        )


    # ============================================================
    # 4. CIRCULAR ORBIT
    # ============================================================
    def test_circular_orbit(self):

        A_orbit = np.array([[0.0, -1.0],
                            [1.0,  0.0]])

        y0 = np.array([1.0, 0.0])
        t_span = (0.0, 10.0)

        print("\n--- Running Test: Circular Orbit ---")

        check_solution_accuracy(
            self,
            f_linear,
            t_span,
            y0,
            A_orbit,
            self.H,
            self.TOL
        )


    # ============================================================
    # 5. SEQUENTIAL DECAY CHAIN
    # ============================================================
    def test_sequential_decay(self):

        A_chain = np.array([[-2.0, 0.0],
                            [ 2.0, -1.0]])

        y0 = np.array([1.0, 0.0])
        t_span = (0.0, 2.0)

        print("\n--- Running Test: Sequential Decay ---")

        check_solution_accuracy(
            self,
            f_linear,
            t_span,
            y0,
            A_chain,
            self.H,
            self.TOL
        )


    # ============================================================
    # 6. STABLE SPIRAL
    # ============================================================
    def test_stable_spiral(self):

        A_spiral = np.array([[-1.0, -5.0],
                             [ 5.0, -1.0]])

        y0 = np.array([1.0, 0.0])
        t_span = (0.0, 3.0)

        print("\n--- Running Test: Stable Spiral ---")

        check_solution_accuracy(
            self,
            f_linear,
            t_span,
            y0,
            A_spiral,
            self.H,
            self.TOL
        )


    # ============================================================
    # 7. STIFF-LIKE SYSTEM
    # ============================================================
    def test_stiff_like_system(self):

        A_stiff = np.array([[-50.0, 0.0],
                            [  0.0, -1.0]])

        y0 = np.array([1.0, 1.0])
        t_span = (0.0, 0.5)

        print("\n--- Running Test: Stiff-like System ---")

        check_solution_accuracy(
            self,
            f_linear,
            t_span,
            y0,
            A_stiff,
            self.H,
            self.TOL
        )


if __name__ == "__main__":
    unittest.main()