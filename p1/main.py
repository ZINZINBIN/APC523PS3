import numpy as np
import matplotlib.pyplot as plt

def implicit_RK(z):
    return (1 + 0.5 * z + z**2 / 12) / (1 - 0.5 * z + z**2 / 12)

def BDF_2nd(z):
    return (2 + z) + np.sqrt((2 + z) ** 2 - 1), (2 + z) - np.sqrt((2 + z) ** 2 - 1)

def BDF_3rd(z):
    rho = np.roots([1, (-1) * (18/11 + 6 * z), 9/11, (-1) * 2 / 11])
    return rho

if __name__ == "__main__":

    # Problem (a). Linear stability of the implicit Runge-Kutta method
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    RZ = implicit_RK(Z)

    # Plot the linear stability regime
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.cla()
    ax.contour(X, Y, np.abs(RZ), np.linspace(0, 1, 60))
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\Omega_r \Delta t$")
    ax.set_ylabel(r"$\Omega_i \Delta t$")
    ax.set_title("Stability Region")
    ax.grid(True)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("./p1_a.png")

    # Problem (b). Explicit BDF for second-order and third order method
    bdf_2nd_p, bdf_2nd_m = BDF_2nd(Z)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes = axes.ravel()

    axes[0].cla()
    axes[0].contour(X, Y, np.abs(bdf_2nd_p), np.linspace(0, 1, 60))
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].set_xlabel(r"$\Omega_r \Delta t$")
    axes[0].set_ylabel(r"$\Omega_i \Delta t$")
    axes[0].set_title(r"$\rho = (2+z) + \sqrt{(2+z)^2 -1}$")
    axes[0].grid(True)
    axes[0].set_aspect("equal")

    axes[1].contour(X, Y, np.abs(bdf_2nd_m), np.linspace(0, 1, 60))
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel(r"$\Omega_r \Delta t$")
    axes[1].set_ylabel(r"$\Omega_i \Delta t$")
    axes[1].set_title(r"$\rho = (2+z) - \sqrt{(2+z)^2 -1}$")
    axes[1].grid(True)
    axes[1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig("./p1_b_2nd.png")

    bdf_3rd_1 = np.empty_like(X, dtype = complex)
    bdf_3rd_2 = np.empty_like(X, dtype = complex)
    bdf_3rd_3 = np.empty_like(X, dtype = complex)

    for i in range(len(X)):
        for j in range(len(X)):
            rho_1, rho_2, rho_3 = BDF_3rd(Z[i,j])
            bdf_3rd_1[i,j] = rho_1
            bdf_3rd_2[i,j] = rho_2
            bdf_3rd_3[i,j] = rho_3

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.ravel()

    axes[0].cla()
    axes[0].contour(X, Y, np.abs(bdf_3rd_1), np.linspace(0, 1, 60))
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].set_xlabel(r"$\Omega_r \Delta t$")
    axes[0].set_ylabel(r"$\Omega_i \Delta t$")
    axes[0].set_title(r"$\rho_1(z)$")
    axes[0].grid(True)
    axes[0].set_aspect("equal")

    axes[1].cla()
    axes[1].contour(X, Y, np.abs(bdf_3rd_2), np.linspace(0, 1, 60))
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel(r"$\Omega_r \Delta t$")
    axes[1].set_ylabel(r"$\Omega_i \Delta t$")
    axes[1].set_title(r"$\rho_2(z)$")
    axes[1].grid(True)
    axes[1].set_aspect("equal")

    axes[2].cla()
    axes[2].contour(X, Y, np.abs(bdf_3rd_3), np.linspace(0, 1, 60))
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].axvline(0, color="k", lw=0.5)
    axes[2].set_xlabel(r"$\Omega_r \Delta t$")
    axes[2].set_ylabel(r"$\Omega_i \Delta t$")
    axes[2].set_title(r"\rho_3(z)")
    axes[2].grid(True)
    axes[2].set_aspect("equal")

    fig.tight_layout()
    fig.savefig("./p1_b_3rd.png")
