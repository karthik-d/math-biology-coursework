import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def predictor_roots(Z):
    """
    Compute max |xi| for the predictor polynomial on a grid Z.

    Predictor polynomial:
        xi^2 - (1 + 1.5 z) xi + 0.5 z = 0.
    """
    a = 1.0
    b = -(1.0 + 1.5 * Z)
    c = 0.5 * Z
    disc = b**2 - 4 * a * c
    sqrt_disc = np.sqrt(disc)
    xi1 = (-b + sqrt_disc) / (2 * a)
    xi2 = (-b - sqrt_disc) / (2 * a)
    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    return rho


def pc_roots(Z):
    """
    Compute max |xi| for the predictor-corrector polynomial on a grid Z.

    Predictor-corrector polynomial:
        xi^2 - (1 + 2 z + 0.75 z^2) xi + 0.25 z^2 = 0.
    """
    a = 1.0
    b = -(1.0 + 2.0 * Z + 0.75 * Z**2)
    c = 0.25 * Z**2
    disc = b**2 - 4 * a * c
    sqrt_disc = np.sqrt(disc)
    xi1 = (-b + sqrt_disc) / (2 * a)
    xi2 = (-b - sqrt_disc) / (2 * a)
    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    return rho


def corrector_amplification(Z):
    """
    Amplification factor for a pure AM2-style corrector:
        R(z) = (1 + 1.5 z) / (1 - 0.5 z).
    """
    return (1.0 + 1.5 * Z) / (1.0 - 0.5 * Z)
