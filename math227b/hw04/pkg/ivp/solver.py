import numpy as np


def f_linear(t, y, A):
    """
    Right-hand side for the linear system y' = A y.

    Parameters
    ----------
    t : float
        Time (not used for autonomous system, but kept for generality).
    y : array_like, shape (m,)
        State vector.
    A : array_like, shape (m, m)
        System matrix.

    Returns
    -------
    dy : ndarray, shape (m,)
        Time derivative A y.
    """
    return A @ y


def adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """
    One step of the 2-step Adams-Bashforth predictor:
        y_{n+1}^* = y_n + h/2 (3 f(t_n, y_n) - f(t_{n-1}, y_{n-1})).

    Parameters
    ----------
    y_nm1 : ndarray
        y_{n-1}.
    y_n : ndarray
        y_n.
    h : float
        Time step.
    f : callable
        RHS function f(t, y, *f_args).
    t_nm1 : float
        t_{n-1}.
    t_n : float
        t_n.
    f_args : tuple
        Extra arguments passed to f.

    Returns
    -------
    y_star : ndarray
        Predicted value y_{n+1}^*.
    """
    f_n = f(t_n, y_n, *f_args)
    f_nm1 = f(t_nm1, y_nm1, *f_args)
    return y_n + 0.5 * h * (3.0 * f_n - f_nm1)


def modified_trapezoidal_corrector_step(y_n, y_star, h, f, t_n, t_np1, *f_args):
    """
    One step of the corrector using predicted value:
        y_{n+1} = y_n + h/2 (3 f(t_n, y_n) + f(t_{n+1}, y_{n+1}^*)).

    Parameters
    ----------
    y_n : ndarray
        y_n.
    y_star : ndarray
        y_{n+1}^* (predictor output).
    h : float
        Time step.
    f : callable
        RHS function f(t, y, *f_args).
    t_n : float
        t_n.
    t_np1 : float
        t_{n+1}.
    f_args : tuple
        Extra arguments passed to f.

    Returns
    -------
    y_np1 : ndarray
        Corrected value y_{n+1}.
    """
    f_n = f(t_n, y_n, *f_args)
    f_star = f(t_np1, y_star, *f_args)
    return y_n + 0.5 * h * (3.0 * f_n + f_star)


def predictor_corrector_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """
    Full predictor-corrector step (second-order accurate):
        1. Predictor:   y_{n+1}^* ← Adams-Bashforth 2-step
        2. Corrector:   y_{n+1} ← modified trapezoidal using y_{n+1}^*.

    Parameters
    ----------
    y_nm1 : ndarray, y_n : ndarray, h : float, etc.
        See sub-steps.

    Returns
    -------
    y_np1 : ndarray
        y_{n+1}.
    y_star : ndarray
        y_{n+1}^* (predictor value).
    """
    t_np1 = t_n + h
    y_star = adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
    y_np1 = modified_trapezoidal_corrector_step(y_n, y_star, h, f, t_n, t_np1, *f_args)
    return y_np1, y_star


def adams_bashforth_predictor_only_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """
    Predictor-only step (use as explicit 2nd-order method):
        y_{n+1} = y_n + h/2 (3 f(t_n, y_n) - f(t_{n-1}, y_{n-1})).

    Parameters
    ----------
    Same as predictor step.

    Returns
    -------
    y_np1 : ndarray
        y_{n+1} = y_{n+1}^*.
    """
    return adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)


def initialize_two_step(y0, t0, h, f, *f_args):
    """
    Initialize y_{n-1}, y_n for two-step methods using Forward Euler for y1.

    Returns
    -------
    y_nm1, y_n, t_nm1, t_n
    """
    y1 = y0 + h * f(t0, y0, *f_args)
    return y0, y1, t0, t0 + h


def solve_adams_bashforth_predictor(f, t_span, y0, h, *f_args):
    """
    Solve ODE using Adams-Bashforth predictor-only (2nd-order explicit).

    Parameters
    ----------
    Same as solve_predictor_corrector.

    Returns
    -------
    t : ndarray
    Y : ndarray, shape (N+1, m)
    """
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    m = y0.size
    N = int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, m))
    Y[0] = y0

    y_nm1, y_n, t_nm1, t_n = initialize_two_step(y0, t0, h, f, *f_args)

    for n in range(N):
        y_np1 = adams_bashforth_predictor_only_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
        Y[n + 1] = y_np1
        y_nm1, y_n = y_n, y_np1
        t_nm1, t_n = t_n, t[n + 1]

    return t, Y


def solve_predictor_corrector(f, t_span, y0, h, *f_args, store_predictor=False):
    """
    Solve ODE using full predictor-corrector (2nd-order).

    Parameters
    ----------
    Same as above, plus store_predictor for y^*.

    Returns
    -------
    t, Y[, Y_star]
    """
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    m = y0.size
    N = int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, m))
    Y[0] = y0

    if store_predictor:
        Y_star = np.zeros_like(Y)

    y_nm1, y_n, t_nm1, t_n = initialize_two_step(y0, t0, h, f, *f_args)

    for n in range(N):
        y_np1, y_star = predictor_corrector_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
        Y[n + 1] = y_np1
        if store_predictor:
            Y_star[n + 1] = y_star
        y_nm1, y_n = y_n, y_np1
        t_nm1, t_n = t_n, t[n + 1]

    if store_predictor:
        return t, Y, Y_star
    return t, Y
