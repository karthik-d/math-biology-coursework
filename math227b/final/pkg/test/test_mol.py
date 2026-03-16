# test_mol_solver.py
import unittest
import numpy as np
from mol_solver import solve_pde_system, rk2_step

# ---------------------------
# Define PDE systems for testing
# ---------------------------

def pde_zero_source(N=11, L=1.0, D=0.01, c=0.0, v0=0.0):
    dx = L / (N-1)
    x = np.linspace(0, L, N)
    u0 = np.zeros(N)
    v = np.zeros_like(x)
    return x, u0, dx, v, D, c

def pde_constant_solution(N=11, L=1.0, D=0.01, c=0.0, v0=0.0):
    dx = L / (N-1)
    x = np.linspace(0, L, N)
    u0 = np.ones(N)  # constant solution
    v = np.zeros_like(x)
    return x, u0, dx, v, D, c

def pde_decay_only(N=11, L=1.0, D=0.0, c=1.0, v0=0.0):
    dx = L / (N-1)
    x = np.linspace(0, L, N)
    u0 = np.ones(N)
    v = np.zeros_like(x)
    return x, u0, dx, v, D, c

def pde_diffusion_only(N=11, L=1.0, D=0.1, c=0.0, v0=0.0):
    dx = L / (N-1)
    x = np.linspace(0, L, N)
    u0 = np.zeros(N)
    u0[5] = 1.0  # spike in the middle
    v = np.zeros_like(x)
    return x, u0, dx, v, D, c

def pde_localized_source(N=11, L=1.0, D=0.01, c=0.1, v0=1.0):
    dx = L / (N-1)
    x = np.linspace(0, L, N)
    u0 = np.zeros(N)
    v = np.zeros_like(x)
    v[x < 0.2] = v0
    return x, u0, dx, v, D, c

# ---------------------------
# Unit test class
# ---------------------------

class TestMOLSolver(unittest.TestCase):

    def test_zero_source(self):
        """Solution with zero source and zero initial condition should remain zero."""
        x, usol, times = solve_pde_system(pde_zero_source, dt=0.01, T=0.1)
        self.assertTrue(np.allclose(usol, 0, atol=1e-12))

    def test_constant_solution(self):
        """Constant initial solution with zero source and zero decay should remain constant."""
        x, usol, times = solve_pde_system(pde_constant_solution, dt=0.01, T=0.1)
        self.assertTrue(np.allclose(usol, 1, atol=1e-12))

    def test_decay_only(self):
        """With decay only, solution should decrease exponentially."""
        x, usol, times = solve_pde_system(pde_decay_only, dt=0.01, T=0.1)
        expected = np.exp(-1.0 * times)[:, np.newaxis] * np.ones(len(x))
        self.assertTrue(np.allclose(usol, expected, rtol=1e-2))

    def test_diffusion_only(self):
        """With diffusion only, mass should spread symmetrically."""
        x, usol, times = solve_pde_system(pde_diffusion_only, dt=0.001, T=0.01)
        # Check that solution remains symmetric around spike
        self.assertTrue(np.allclose(usol[-1], usol[-1][::-1], atol=1e-12))

    def test_localized_source(self):
        """With localized source and decay, solution should grow near source and decay elsewhere."""
        x, usol, times = solve_pde_system(pde_localized_source, dt=0.01, T=0.5)
        # Check that maximum is in the source region
        max_idx = np.argmax(usol[-1])
        self.assertTrue(x[max_idx] < 0.2)
        # Check that final solution is nonzero
        self.assertTrue(np.any(usol[-1] > 0))

    def test_rk2_step_stability(self):
        """Verify single RK2 step does not blow up for small dt."""
        x, u0, dx, v, D, c = pde_diffusion_only()
        u1 = rk2_step(u0, dt=0.001, dx=dx, D=D, c=c, v=v)
        self.assertTrue(np.all(np.isfinite(u1)))

# ---------------------------
# Run the tests
# ---------------------------

if __name__ == "__main__":
    unittest.main()