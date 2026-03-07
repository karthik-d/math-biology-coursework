import numpy as np

def f_linear(t, y, A):
    """Linear RHS: y' = A y."""
    return A @ y

def adams_bashforth_predictor_step(y_n, f_n, f_nm1, h):
    """
    AB2 Predictor using pre-computed derivatives:
    y_star = y_n + (h/2) * (3*f_n - f_nm1)
    """
    return y_n + 0.5 * h * (3.0 * f_n - f_nm1)

def predictor_corrector_step(y_n, f_n, f_nm1, h, f, t_np1, *f_args):
    """
    Full PECE Cycle:
    1. Predict: y_star (AB2)
    2. Evaluate: f_star = f(t_np1, y_star)
    3. Correct: y_np1 (Trapezoidal)
    4. Evaluate: f_np1 = f(t_np1, y_np1)  <-- Crucial for O(h^2) Global Error
    """
    # P: Predict
    y_star = adams_bashforth_predictor_step(y_n, f_n, f_nm1, h)
    
    # E: Evaluate
    f_star = f(t_np1, y_star, *f_args)
    
    # C: Correct
    y_np1 = y_n + 0.5 * h * (f_n + f_star)
    
    # E: Final Evaluate (Ensures derivatives are O(h^2) accurate for next step)
    f_np1 = f(t_np1, y_np1, *f_args)
    
    return y_np1, f_np1, y_star

def initialize_two_step_rk4(y0, t0, h, f, *f_args):
    """Standard RK4 startup to generate y1."""
    k1 = f(t0, y0, *f_args)
    k2 = f(t0 + 0.5*h, y0 + 0.5*h*k1, *f_args)
    k3 = f(t0 + 0.5*h, y0 + 0.5*h*k2, *f_args)
    k4 = f(t0 + h, y0 + h*k3, *f_args)
    return y0 + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def solve_adams_bashforth_predictor(f, t_span, y0, h, *f_args):
    """Corrected AB2 predictor only."""
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    N = int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, y0.size))
    F = np.zeros((N + 1, y0.size)) # Store derivatives to avoid re-calculation
    
    Y[0] = y0
    F[0] = f(t[0], Y[0], *f_args)
    
    if N > 0:
        Y[1] = initialize_two_step_rk4(Y[0], t[0], h, f, *f_args)
        F[1] = f(t[1], Y[1], *f_args)
    
    for n in range(1, N):
        Y[n+1] = adams_bashforth_predictor_step(Y[n], F[n], F[n-1], h)
        F[n+1] = f(t[n+1], Y[n+1], *f_args)
        
    return t, Y

def solve_predictor_corrector(f, t_span, y0, h, *f_args):
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    N = int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    
    Y = np.zeros((N + 1, y0.size))
    F = np.zeros((N + 1, y0.size)) # Store derivatives

    # 1. Initialize Step 0
    Y[0] = y0
    F[0] = f(t[0], Y[0], *f_args)
    
    # 2. Initialize Step 1 (Must be at least O(h^2) accurate)
    if N > 0:
        Y[1] = initialize_two_step_rk4(Y[0], t[0], h, f, *f_args)
        # CRITICAL: Evaluate derivative at the new state
        F[1] = f(t[1], Y[1], *f_args) 
    
    # 3. Main Loop
    for n in range(1, N):
        # Predict y_{n+1} using AB2 (Uses F[n] and F[n-1])
        y_star = Y[n] + 0.5 * h * (3.0 * F[n] - F[n-1])
        
        # Evaluate derivative at predicted state
        f_star = f(t[n+1], y_star, *f_args)
        
        # Correct y_{n+1} using Trapezoidal rule
        Y[n+1] = Y[n] + 0.5 * h * (F[n] + f_star)
        
        # CRITICAL "E": Re-evaluate derivative at the CORRECTED state
        # Without this line, the next step uses a derivative that is only O(h) accurate
        F[n+1] = f(t[n+1], Y[n+1], *f_args)
            
    return t, Y