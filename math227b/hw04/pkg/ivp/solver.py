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


def step_predictor(y_nm1, y_n, h, f, t_n, t_nm1, *f_args):
    """
    One step of the explicit predictor:
        y_{n+1}^* = y_n + h/2 (3 f(y_n) - f(y_{n-1})).

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
    t_n : float
        t_n.
    t_nm1 : float
        t_{n-1}.
    f_args : tuple
        Extra arguments passed to f.

    Returns
    -------
    y_star : ndarray
        Predicted value y_{n+1}^*.
    """
    f_n = f(t_n,   y_n,   *f_args)
    f_nm1 = f(t_nm1, y_nm1, *f_args)
    return y_n + 0.5 * h * (3.0 * f_n - f_nm1)


def step_corrector_from_star(y_n, y_star, h, f, t_n, t_np1, *f_args):
    """
    One step of the corrector using the predicted value y_{n+1}^*:
        y_{n+1} = y_n + h/2 (3 f(y_n) + f(y_{n+1}^*)).

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
    f_n = f(t_n,   y_n,   *f_args)
    f_star = f(t_np1, y_star, *f_args)
    return y_n + 0.5 * h * (3.0 * f_n + f_star)


def pc_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """
    One full predictor-corrector step:
      1. Predictor:   y_{n+1}^* = y_n + h/2 (3 f_n - f_{n-1})
      2. Corrector:   y_{n+1}   = y_n + h/2 (3 f_n + f(y_{n+1}^*))

    Parameters
    ----------
    y_nm1 : ndarray
        y_{n-1}.
    y_n : ndarray
        y_n.
    h : float
        Step size.
    f : callable
        RHS function f(t, y, *f_args).
    t_nm1 : float
        t_{n-1}.
    t_n : float
        t_n.
    f_args : tuple
        Extra arguments for f.

    Returns
    -------
    y_np1 : ndarray
        y_{n+1}.
    y_star : ndarray
        y_{n+1}^* (in case you want to analyze it).
    """
    t_np1 = t_n + h
    y_star = step_predictor(y_nm1, y_n, h, f, t_n, t_nm1, *f_args)
    y_np1 = step_corrector_from_star(y_n, y_star, h, f, t_n, t_np1, *f_args)
    return y_np1, y_star


def initialize_two_step(y0, t0, h, f, *f_args):
    """
    Initialize y_{n-1} and y_n for a two-step method.

    We compute y1 using a one-step method (Forward Euler):
        y1 = y0 + h f(t0, y0)

    Then the two-step method starts with
        (y_{n-1}, y_n) = (y0, y1).

    Parameters
    ----------
    y0 : ndarray
        Initial condition at t0.
    t0 : float
        Initial time.
    h : float
        Time step.
    f : callable
        RHS function f(t, y, *f_args).
    f_args : tuple
        Extra arguments passed to f.

    Returns
    -------
    y_nm1 : ndarray
        y_{n-1} = y0.
    y_n : ndarray
        y_n = y1.
    t_nm1 : float
        t_{n-1} = t0.
    t_n : float
        t_n = t0 + h.
    """

    # compute y1 with forward Euler
    y1 = y0 + h * f(t0, y0, *f_args)

    y_nm1 = y0
    y_n = y1

    t_nm1 = t0
    t_n = t0 + h

    return y_nm1, y_n, t_nm1, t_n


def solve_pc(f, t_span, y0, h, *f_args, store_predictor=False):
    """
    Solve an ODE system using the predictor-corrector scheme.

    Parameters
    ----------
    f : callable
        RHS function f(t, y, *f_args).
    t_span : tuple
        (t0, T) time interval.
    y0 : array_like
        Initial condition at t0.
    h : float
        Time step.
    f_args : tuple
        Extra arguments for f (e.g. matrix A).
    store_predictor : bool, optional
        If True, also return the predictor values y^*_{n+1}.

    Returns
    -------
    t : ndarray
        Time grid.
    Y : ndarray, shape (N, m)
        Solution at each time step.
    Y_star : ndarray, shape (N, m), optional
        Predictor values (if store_predictor=True, otherwise not returned).
    """
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    m = y0.size
    N = int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)

    Y = np.zeros((N + 1, m))
    Y[0] = y0

    # Initialize two-step history
    y_nm1, y_n, t_nm1, t_n = initialize_two_step(y0, t0, h, f, *f_args)

    if store_predictor:
        Y_star = np.zeros_like(Y)
        Y_star[0] = y0

    for n in range(N):
        y_np1, y_star = pc_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
        Y[n + 1] = y_np1
        if store_predictor:
            Y_star[n + 1] = y_star

        # shift history
        y_nm1, y_n = y_n, y_np1
        t_nm1, t_n = t_n, t[n + 1]

    if store_predictor:
        return t, Y, Y_star
    else:
        return t, Y
