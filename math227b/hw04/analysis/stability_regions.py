import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_stability_predictor_corrector():

    x = np.linspace(-5, 2, 600)
    y = np.linspace(-5, 5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    rho = np.zeros_like(Z, dtype=float)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]
            # coefficients of stability polynomial
            a = 1
            b = -(1 + 2*z + 0.75*z**2)
            c = 0.25*z**2
            roots = np.roots([a, b, c])
            rho[i, j] = np.max(np.abs(roots))

    stable = rho <= 1

    plt.figure(figsize=(7,6))
    colors = ["lightcoral", "lightblue"]  # unstable, stable
    plt.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=colors)
    plt.contour(X, Y, rho, levels=[1], colors="black", linewidths=1)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Predictor-Corrector: Region of Absolute Stability")

    legend_elements = [
        Patch(facecolor="lightblue", label="Stable (max |ξ| ≤ 1)"),
        Patch(facecolor="lightcoral", label="Unstable (max |ξ| > 1)")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()
    

def plot_stability_predictor():

    x = np.linspace(-5, 2, 600)
    y = np.linspace(-5, 5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    rho = np.zeros_like(Z, dtype=float)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]

            # predictor stability polynomial
            a = 1
            b = -(1 + 1.5*z)
            c = 0.5*z

            roots = np.roots([a, b, c])
            rho[i, j] = np.max(np.abs(roots))

    stable = rho <= 1

    plt.figure(figsize=(7,6))
    colors = ["lightcoral", "lightblue"]  # unstable, stable
    plt.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=colors)
    plt.contour(X, Y, rho, levels=[1], colors="black", linewidths=1)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Predictor Stability Region")

    legend_elements = [
        Patch(facecolor="lightblue", label="Stable (max |ξ| ≤ 1)"),
        Patch(facecolor="lightcoral", label="Unstable (max |ξ| > 1)")
    ]

    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()
    

def plot_stability_corrector():

    x = np.linspace(-5, 2, 600)
    y = np.linspace(-5, 5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    rho = np.zeros_like(Z, dtype=float)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]

            # amplification factor
            R = (1 + 1.5*z) / (1 - 0.5*z)

            rho[i, j] = np.abs(R)

    stable = rho <= 1

    plt.figure(figsize=(7,6))
    colors = ["lightcoral", "lightblue"]  # unstable, stable
    plt.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=colors)
    plt.contour(X, Y, rho, levels=[1], colors="black", linewidths=1)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Corrector Stability Region")

    legend_elements = [
        Patch(facecolor="lightblue", label="Stable (|R(z)| ≤ 1)"),
        Patch(facecolor="lightcoral", label="Unstable (|R(z)| > 1)")
    ]

    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()
    

if __name__ == "__main__":
    plot_stability_predictor_corrector()
    plot_stability_predictor()
    plot_stability_corrector()