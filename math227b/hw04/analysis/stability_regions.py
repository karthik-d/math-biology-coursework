import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_stability_predictor_corrector():
    # Use a wider range to see the structure
    x = np.linspace(-3, 1, 600)
    y = np.linspace(-2, 2, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # PC Polynomial (PECE with AB2-AM2):
    # xi^2 - (1 + z + 0.75 z^2) xi + 0.25 z^2 = 0
    a = 1.0
    b = -(1.0 + Z + 0.75 * Z**2)
    c = 0.25 * Z**2

    disc = b**2 - 4 * a * c
    sqrt_disc = np.sqrt(disc + 0j)
    xi1 = (-b + sqrt_disc) / (2 * a)
    xi2 = (-b - sqrt_disc) / (2 * a)

    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    stable = rho <= 1.000001 # Tolerance for numerical precision

    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=["lightcoral", "lightblue"])
    plt.contour(X, Y, rho, levels=[1], colors="black", linewidths=1.5)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Predictor-Corrector (AB2-AM2) Stability Region")
    plt.show()

def plot_stability_predictor():
    x = np.linspace(-2.5, 1, 600)
    y = np.linspace(-1.5, 1.5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # AB2 Predictor polynomial:
    # xi^2 - (1 + 1.5 z) xi + 0.5 z = 0
    a = 1.0
    b = -(1.0 + 1.5 * Z)
    c = 0.5 * Z

    disc = b**2 - 4 * a * c
    sqrt_disc = np.sqrt(disc + 0j)
    xi1 = (-b + sqrt_disc) / (2 * a)
    xi2 = (-b - sqrt_disc) / (2 * a)

    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    stable = rho <= 1.000001

    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=["lightcoral", "gold"])
    plt.contour(X, Y, rho, levels=[1], colors="black", linewidths=1.5)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Predictor (AB2) Stability Region")
    plt.show()

def plot_stability_corrector():
    x = np.linspace(-5, 5, 600)
    y = np.linspace(-5, 5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Correct AM2 (Trapezoidal) amplification factor:
    # R(z) = (1 + z/2) / (1 - z/2)
    R = (1.0 + 0.5 * Z) / (1.0 - 0.5 * Z)
    rho = np.abs(R)
    stable = rho <= 1.000001

    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=["lightcoral", "lightskyblue"])
    plt.contour(X, Y, rho, levels=[1], colors="black", linewidths=1.5)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Corrector (AM2/Trapezoidal) Stability Region")
    plt.show()

def plot_stability_overlay():
    x = np.linspace(-3, 1, 800)
    y = np.linspace(-2.5, 2.5, 800)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # 1. Predictor (AB2)
    rho_p = np.maximum(np.abs((- (-(1.0 + 1.5 * Z)) + np.sqrt((-(1.0 + 1.5 * Z))**2 - 2*Z + 0j)) / 2),
                       np.abs((- (-(1.0 + 1.5 * Z)) - np.sqrt((-(1.0 + 1.5 * Z))**2 - 2*Z + 0j)) / 2))
    
    # 2. Corrector (AM2)
    rho_c = np.abs((1.0 + 0.5 * Z) / (1.0 - 0.5 * Z))
    
    # 3. PC Scheme
    b_pc = -(1.0 + Z + 0.75 * Z**2)
    rho_pc = np.maximum(np.abs((-b_pc + np.sqrt(b_pc**2 - Z**2 + 0j)) / 2),
                        np.abs((-b_pc - np.sqrt(b_pc**2 - Z**2 + 0j)) / 2))

    plt.figure(figsize=(9, 7))
    plt.contourf(X, Y, np.zeros_like(X), levels=[-0.5, 0.5], colors=["lightcoral"]) # Unstable background

    # Overlay stability regions
    plt.contourf(X, Y, (rho_c <= 1.001), levels=[0.5, 1.5], colors=["lightskyblue"], alpha=0.3)
    plt.contourf(X, Y, (rho_pc <= 1.001), levels=[0.5, 1.5], colors=["mediumseagreen"], alpha=0.5)
    plt.contourf(X, Y, (rho_p <= 1.001), levels=[0.5, 1.5], colors=["gold"], alpha=0.7)

    # Clean boundaries
    plt.contour(X, Y, rho_c, levels=[1], colors="blue", linewidths=1)
    plt.contour(X, Y, rho_pc, levels=[1], colors="darkgreen", linewidths=1)
    plt.contour(X, Y, rho_p, levels=[1], colors="orange", linewidths=1)

    plt.axhline(0, color="black", lw=1)
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Comparison of Absolute Stability Regions")

    legend_elements = [
        Patch(facecolor="gold", label="Predictor (AB2)"),
        Patch(facecolor="mediumseagreen", label="PC Scheme (PECE)"),
        Patch(facecolor="lightskyblue", label="Corrector (AM2 - A-Stable)"),
        Patch(facecolor="lightcoral", label="Unstable")
    ]
    plt.legend(handles=legend_elements, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()

if __name__ == "__main__":
	plot_stability_predictor()
	plot_stability_corrector()
	plot_stability_predictor_corrector()
	plot_stability_overlay()