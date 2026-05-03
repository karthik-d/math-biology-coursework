import unittest
import numpy as np

from rk4.rk4 import rk4_solver

class TestRK4Solver(unittest.TestCase):

    # ------------------ Basic tests ------------------
    def test_linear_growth(self):
        f = lambda t, y: y
        y_exact = lambda t: np.exp(t)
        t0, y0, tf = 0.0, 1.0, 1.0
        h = 0.1
        t, y = rk4_solver(f, t0, y0, h, tf)
        self.assertAlmostEqual(y[-1], y_exact(tf), delta=1e-5)

    def test_exponential_decay(self):
        f = lambda t, y: -2*y
        y_exact = lambda t: np.exp(-2*t)
        t0, y0, tf = 0.0, 1.0, 1.0
        h = 0.1
        t, y = rk4_solver(f, t0, y0, h, tf)
        self.assertAlmostEqual(y[-1], y_exact(tf), delta=1e-5)

    def test_quadratic(self):
        f = lambda t, y: 2*t
        y_exact = lambda t: t**2
        t0, y0, tf = 0.0, 0.0, 1.0
        h = 0.1
        t, y = rk4_solver(f, t0, y0, h, tf)
        self.assertAlmostEqual(y[-1], y_exact(tf), delta=1e-6)

    def test_sinusoidal(self):
        f = lambda t, y: np.cos(t)
        y_exact = lambda t: np.sin(t)
        t0, y0, tf = 0.0, 0.0, np.pi/2
        h = 0.01
        t, y = rk4_solver(f, t0, y0, h, tf)
        self.assertAlmostEqual(y[-1], y_exact(tf), delta=1e-4)

    def test_logistic(self):
        f = lambda t, y: y*(1-y)
        y_exact = lambda t: 1/(1 + 9*np.exp(-t))
        t0, y0, tf = 0.0, 0.1, 2.0
        h = 0.01
        t, y = rk4_solver(f, t0, y0, h, tf)
        self.assertAlmostEqual(y[-1], y_exact(tf), delta=1e-4)

    def test_trig_combination(self):
        """y' = y cos(t) + sin(t), y(0)=0, exact y = tan(t) - sin(t)? (integral form)"""
        # Use small t interval to avoid complexity
        f = lambda t, y: y*np.cos(t) + np.sin(t)
        # Exact solution using integrating factor: y = (1/2) (e^{sin(t)} - cos(t) -1)? 
        # We'll test only over small interval t in [0,0.5] numerically
        t0, y0, tf = 0.0, 0.0, 0.5
        h = 0.001
        t, y = rk4_solver(f, t0, y0, h, tf)
        # For test, compare with very fine RK4 solution as "exact"
        t_fine, y_fine = rk4_solver(f, t0, y0, 1e-4, tf)
        self.assertAlmostEqual(y[-1], y_fine[-1], delta=1e-4)

    def test_nonlinear_polynomial(self):
        """y' = y^2 + 2t, y(0)=0, small t for comparison"""
        f = lambda t, y: y**2 + 2*t
        t0, y0, tf = 0.0, 0.0, 0.1  # small t to avoid blow-up
        h = 0.001
        t, y = rk4_solver(f, t0, y0, h, tf)
        # Compare with very fine RK4 solution as reference
        t_fine, y_fine = rk4_solver(f, t0, y0, 1e-4, tf)
        self.assertAlmostEqual(y[-1], y_fine[-1], delta=1e-4)


if __name__ == "__main__":
    unittest.main()