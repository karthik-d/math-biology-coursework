import numpy as np
from scipy.interpolate import CubicSpline


def build_cubic_spline(x, y, bc_type="natural"):
    """
    Construct a cubic spline interpolant.

    Parameters
    ----------
    x : array_like
        Grid points (must be strictly increasing).
    y : array_like
        Function values at grid points.
    bc_type : str
        Boundary condition: "natural" or "not-a-knot".

    Returns
    -------
    spline : CubicSpline
        SciPy CubicSpline object.
    """
    if bc_type not in ["natural", "not-a-knot"]:
        raise ValueError("bc_type must be 'natural' or 'not-a-knot'")

    spline = CubicSpline(x, y, bc_type=bc_type)
    return spline


def evaluate_spline(spline, x_eval):
    """
    Evaluate spline at new points.

    Parameters
    ----------
    spline : CubicSpline
        Cubic spline object.
    x_eval : array_like
        Points to evaluate spline.

    Returns
    -------
    y_eval : ndarray
        Spline values.
    """
    return spline(x_eval)


def spline_derivative(spline, order=1):
    """
    Return derivative spline.

    Parameters
    ----------
    spline : CubicSpline
        Original spline.
    order : int
        Derivative order.

    Returns
    -------
    dspline : CubicSpline
        Derivative spline.
    """
    return spline.derivative(order)
