import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Global plotting settings for consistency
STABLE_COLOR = "lightblue"
UNSTABLE_COLOR = "lightcoral"
PREDICTOR_COLOR = "gold"
CORRECTOR_COLOR = "lightskyblue"
PC_COLOR = "mediumseagreen"

def add_plot_details(ax, title, custom_legend_elements=None):
    """Refines plot aesthetics for publication quality."""
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel(r"Re($z$)", fontsize=12)
    ax.set_ylabel(r"Im($z$)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6)
    
    if custom_legend_elements:
        ax.legend(handles=custom_legend_elements, loc="upper left", framealpha=0.9)
    else:
        legend_elements = [
            Patch(facecolor=STABLE_COLOR, label=r"Stable ($|\xi| \leq 1$)"),
            Patch(facecolor=UNSTABLE_COLOR, label=r"Unstable ($|\xi| > 1$)")
        ]
        ax.legend(handles=legend_elements, loc="upper left")

def plot_stability_predictor():
    x = np.linspace(-2.5, 1, 600)
    y = np.linspace(-2, 2, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # AB2 Predictor polynomial: xi^2 - (1 + 1.5 z) xi + 0.5 z = 0
    b = -(1.0 + 1.5 * Z)
    c = 0.5 * Z
    disc = b**2 - 4 * c
    sqrt_disc = np.sqrt(disc + 0j)
    xi1 = (-b + sqrt_disc) / 2
    xi2 = (-b - sqrt_disc) / 2
    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    stable = rho <= 1.000001

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=[UNSTABLE_COLOR, PREDICTOR_COLOR])
    ax.contour(X, Y, rho, levels=[1], colors="black", linewidths=1.5)
    
    legend = [Patch(facecolor=PREDICTOR_COLOR, label=r"Stable (AB2)"),
              Patch(facecolor=UNSTABLE_COLOR, label=r"Unstable")]
    add_plot_details(ax, "Stability Region: 2nd-Order Adams-Bashforth (Predictor)", legend)
    plt.show()

def plot_stability_corrector():
    # Wider range to demonstrate A-stability
    x = np.linspace(-5, 2, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # AM2 (Trapezoidal) amplification factor: R(z) = (1 + z/2) / (1 - z/2)
    R = (1.0 + 0.5 * Z) / (1.0 - 0.5 * Z)
    rho = np.abs(R)
    stable = rho <= 1.000001

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=[UNSTABLE_COLOR, CORRECTOR_COLOR])
    ax.contour(X, Y, rho, levels=[1], colors="black", linewidths=1.5)
    
    legend = [Patch(facecolor=CORRECTOR_COLOR, label=r"Stable (AM2/Trapezoidal)"),
              Patch(facecolor=UNSTABLE_COLOR, label=r"Unstable")]
    add_plot_details(ax, "Stability Region: Trapezoidal Rule (Corrector)", legend)
    plt.show()

def plot_stability_predictor_corrector():
    x = np.linspace(-3, 1, 600)
    y = np.linspace(-2.5, 2.5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # PC Polynomial: xi^2 - (1 + z + 0.75 z^2) xi + 0.25 z^2 = 0
    b = -(1.0 + Z + 0.75 * Z**2)
    c = 0.25 * Z**2
    disc = b**2 - 4 * c
    sqrt_disc = np.sqrt(disc + 0j)
    xi1 = (-b + sqrt_disc) / 2
    xi2 = (-b - sqrt_disc) / 2
    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    stable = rho <= 1.000001

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(X, Y, stable, levels=[-0.5, 0.5, 1.5], colors=[UNSTABLE_COLOR, PC_COLOR])
    ax.contour(X, Y, rho, levels=[1], colors="black", linewidths=1.5)
    
    legend = [Patch(facecolor=PC_COLOR, label=r"Stable (Full PC Scheme)"),
              Patch(facecolor=UNSTABLE_COLOR, label=r"Unstable")]
    add_plot_details(ax, "Stability Region: Combined Predictor-Corrector", legend)
    plt.show()

def plot_stability_overlay():
    x = np.linspace(-3.5, 1.5, 800)
    y = np.linspace(-3, 3, 800)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # AB2 Predictor
    rho_p = np.maximum(np.abs(( (1.0 + 1.5*Z) + np.sqrt((1.0 + 1.5*Z)**2 - 2*Z + 0j)) / 2),
                       np.abs(( (1.0 + 1.5*Z) - np.sqrt((1.0 + 1.5*Z)**2 - 2*Z + 0j)) / 2))
    # AM2 Corrector
    rho_c = np.abs((1.0 + 0.5 * Z) / (1.0 - 0.5 * Z))
    # PC Scheme
    b_pc = -(1.0 + Z + 0.75 * Z**2)
    rho_pc = np.maximum(np.abs((-b_pc + np.sqrt(b_pc**2 - Z**2 + 0j)) / 2),
                        np.abs((-b_pc - np.sqrt(b_pc**2 - Z**2 + 0j)) / 2))

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Base unstable background
    ax.contourf(X, Y, np.zeros_like(X), levels=[-0.5, 0.5], colors=[UNSTABLE_COLOR])

    # Overlay stable sets
    ax.contourf(X, Y, (rho_c <= 1.001), levels=[0.5, 1.5], colors=[CORRECTOR_COLOR], alpha=0.3)
    ax.contourf(X, Y, (rho_pc <= 1.001), levels=[0.5, 1.5], colors=[PC_COLOR], alpha=0.5)
    ax.contourf(X, Y, (rho_p <= 1.001), levels=[0.5, 1.5], colors=[PREDICTOR_COLOR], alpha=0.7)

    # Boundaries
    ax.contour(X, Y, rho_c, levels=[1], colors="blue", linewidths=1, linestyles='--')
    ax.contour(X, Y, rho_pc, levels=[1], colors="darkgreen", linewidths=1.5)
    ax.contour(X, Y, rho_p, levels=[1], colors="darkorange", linewidths=1.5)

    legend_elements = [
        Patch(facecolor=PREDICTOR_COLOR, alpha=0.7, label="Predictor (AB2)"),
        Patch(facecolor=PC_COLOR, alpha=0.5, label="PC Scheme (PECE)"),
        Patch(facecolor=CORRECTOR_COLOR, alpha=0.3, label="Corrector (AM2 - A-Stable)"),
        Patch(facecolor=UNSTABLE_COLOR, label="Unstable Region")
    ]
    add_plot_details(ax, "Overlay of Absolute Stability Regions", legend_elements)
    plt.show()

if __name__ == "__main__":
    # Standard call sequence for all plots
    plot_stability_predictor()
    plot_stability_corrector()
    plot_stability_predictor_corrector()
    plot_stability_overlay()