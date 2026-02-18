import matplotlib.pyplot as plt
import numpy as np


def plot_solutions(t_rk, y_rk, t_ex, y_ex):
    
    plt.figure(figsize=(8,5))
    plt.plot(t_ex, y_ex, 'k-', label="Analytical", alpha=0.5)
    plt.plot(t_rk, y_rk, 'bo', label="RK4", alpha=0.5)

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Numerical solution of y' = y - t^2 + 1")
    plt.legend()
    plt.grid(True)
    plt.show()
    

