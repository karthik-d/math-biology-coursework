import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_stability_predictor_corrector():
    x = np.linspace(-5, 2, 600)
    y = np.linspace(-5, 5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Predictor–corrector polynomial:
    # xi^2 - (1 + 2z + 0.75 z^2) xi + 0.25 z^2 = 0
    a = 1.0
    b = -(1.0 + 2.0 * Z + 0.75 * Z**2)
    c = 0.25 * Z**2

    disc = b**2 - 4 * a * c
    sqrt_disc = np.sqrt(disc)
    xi1 = (-b + sqrt_disc) / (2 * a)
    xi2 = (-b - sqrt_disc) / (2 * a)

    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    stable = rho <= 1

    plt.figure(figsize=(7, 6))
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
    Z = X + 1j * Y

    # Predictor polynomial:
    # xi^2 - (1 + 1.5 z) xi + 0.5 z = 0
    a = 1.0
    b = -(1.0 + 1.5 * Z)
    c = 0.5 * Z

    disc = b**2 - 4 * a * c
    sqrt_disc = np.sqrt(disc)
    xi1 = (-b + sqrt_disc) / (2 * a)
    xi2 = (-b - sqrt_disc) / (2 * a)

    rho = np.maximum(np.abs(xi1), np.abs(xi2))
    stable = rho <= 1

    plt.figure(figsize=(7, 6))
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
    Z = X + 1j * Y

    # Corrector amplification factor:
    # R(z) = (1 + 1.5 z) / (1 - 0.5 z)
    R = (1.0 + 1.5 * Z) / (1.0 - 0.5 * Z)
    rho = np.abs(R)
    stable = rho <= 1

    plt.figure(figsize=(7, 6))
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
    

def plot_stability_overlay():
    x = np.linspace(-5, 2, 600)
    y = np.linspace(-5, 5, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Predictor polynomial: xi^2 - (1 + 1.5 z) xi + 0.5 z = 0
    a_p = 1.0
    b_p = -(1.0 + 1.5 * Z)
    c_p = 0.5 * Z
    disc_p = b_p**2 - 4 * a_p * c_p
    sqrt_disc_p = np.sqrt(disc_p)
    xi1_p = (-b_p + sqrt_disc_p) / (2 * a_p)
    xi2_p = (-b_p - sqrt_disc_p) / (2 * a_p)
    rho_p = np.maximum(np.abs(xi1_p), np.abs(xi2_p))
    stable_p = rho_p <= 1

    # Corrector amplification factor (pure AM2-style corrector):
    # R(z) = (1 + 1.5 z) / (1 - 0.5 z)
    R_c = (1.0 + 1.5 * Z) / (1.0 - 0.5 * Z)
    rho_c = np.abs(R_c)
    stable_c = rho_c <= 1

    # Predictor–corrector polynomial:
    # xi^2 - (1 + 2 z + 0.75 z^2) xi + 0.25 z^2 = 0
    a_pc = 1.0
    b_pc = -(1.0 + 2.0 * Z + 0.75 * Z**2)
    c_pc = 0.25 * Z**2
    disc_pc = b_pc**2 - 4 * a_pc * c_pc
    sqrt_disc_pc = np.sqrt(disc_pc)
    xi1_pc = (-b_pc + sqrt_disc_pc) / (2 * a_pc)
    xi2_pc = (-b_pc - sqrt_disc_pc) / (2 * a_pc)
    rho_pc = np.maximum(np.abs(xi1_pc), np.abs(xi2_pc))
    stable_pc = rho_pc <= 1

    plt.figure(figsize=(7, 6))

    # Background: unstable region (lightcoral), overlay stable sets by contours
    # Use alpha so overlaps are visible
    plt.contourf(X, Y, np.zeros_like(X), levels=[-0.5, 0.5],
                 colors=["lightcoral"])  # base background

    # Filled regions (optional, semi-transparent) for each method
    plt.contourf(X, Y, stable_c, levels=[0.5, 1.5],
                 colors=["lightblue"], alpha=0.25)
    plt.contourf(X, Y, stable_pc, levels=[0.5, 1.5],
                 colors=["mediumseagreen"], alpha=0.4)
    plt.contourf(X, Y, stable_p, levels=[0.5, 1.5],
                 colors=["gold"], alpha=0.6)

    # Boundaries |·| = 1
    plt.contour(X, Y, rho_c, levels=[1], colors="blue", linewidths=1.2)
    plt.contour(X, Y, rho_pc, levels=[1], colors="green", linewidths=1.2)
    plt.contour(X, Y, rho_p, levels=[1], colors="orange", linewidths=1.2)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Overlay of Stability Regions: Predictor, Corrector, Predictor–Corrector")

    legend_elements = [
        Patch(facecolor="gold", alpha=0.6, label="Predictor stable"),
        Patch(facecolor="mediumseagreen", alpha=0.4, label="Predictor–Corrector stable"),
        Patch(facecolor="lightblue", alpha=0.25, label="Corrector stable"),
        Patch(facecolor="lightcoral", label="Unstable background")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()

    
# if __name__ == "__main__":
#     plot_stability_predictor_corrector()
#     plot_stability_predictor()
#     plot_stability_corrector()
#     plot_stability_overlay()