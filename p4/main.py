import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def compute_fu(u:np.array, k1:float, k2:float, k3:float):
    xn, xo, xn2, xo2, xno = u[0], u[1], u[2], u[3], u[4]
    
    dxn = k1 * xn2 * xo - k2 * xno * xn - k3 * xn * xo2
    dxo = (-1) * k1 * xn2 * xo + k2 * xno * xn + k3 * xn * xo2
    dxn2 = (-1) * k1 * xn2 * xo + k2 * xno * xn 
    dxo2 = (-1) * k3 * xn * xo2
    dxno = k1 * xn2 * xo - k2 * xno * xn + k3 * xn * xo2
    
    return np.array([dxn, dxo, dxn2, dxo2, dxno])

def compute_Ju(u:np.array, k1:float, k2:float, k3:float):
    
    xn, xo, xn2, xo2, xno = u[0], u[1], u[2], u[3], u[4]
    
    J = np.zeros((5,5))
    J[0,0] = (-1) * k2 * xno + (-1) * k3 * xo2
    J[0,1] = k1 * xn2
    J[0,2] = k1 * xo
    J[0,3] = (-1) * k3 * xn
    J[0,4] = (-1) * k2 * xn
    
    J[1,0] = k2 * xno + k3 * xo2
    J[1,1] = (-1) * k1 * xn
    J[1,2] = (-1) * k1 * xo
    J[1,3] = k3 * xn
    J[1,4] = k2 * xn
    
    J[2,0] = k2 * xno
    J[2,1] = (-1) * k1 * xn2
    J[2,2] = (-1) * k1 * xo
    J[2,3] = 0
    J[2,4] = k2 * xn
    
    J[3,0] = (-1) * k3 * xo2
    J[3,1] = 0
    J[3,2] = 0
    J[3,3] = (-1) * k3 * xn
    J[3,4] = 0
    
    J[4,0] = (-1) * k2 * xno + k3 * xo2
    J[4,1] = k1 * xn2
    J[4,2] = k1 * xo
    J[4,3] = k3 * xn
    J[4,4] = (-1) * k2 * xn
    
    return J

def RK4(u:np.ndarray, dt:float, func:Callable):
    k1 = func(u)
    k2 = func(u + 0.5 * dt * k1)
    k3 = func(u + 0.5 * dt * k2)
    k4 = func(u + dt * k3)
    uf = u + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return uf

def DIRK2(u: np.ndarray, dt: float, func: Callable):
    k1 = func(u)
    u1 = u + 0.5 * k1 * dt
    uf = u + dt * func(u1)
    return uf

def BDF2(u: np.ndarray, dt: float, func: Callable):
    k1 = func(u)
    k2 = func(u + 0.5 * dt * k1)
    k3 = func(u + 0.5 * dt * k2)
    k4 = func(u + dt * k3)
    uf = u + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return uf


if __name__ == "__main__":

    # Coefficient
    k1 = 2.0 * 10 ** 3
    k2 = 3.0 * 10 ** (-12)
    k3 = 2.0 * 10 ** 1

    # N,O,N2,O2,NO
    u_init = np.array([0.01, 0.01, 0.75, 0.23, 0.00])

    # Problem 1. Eigenvalues of the Jacobian at t = 0
    J_init = compute_Ju(u_init, k1, k2, k3)
    eigval, eigvec = np.linalg.eig(J_init)

    eig_max = max(np.abs(eigval))
    eig_min = min(np.abs(eigval))
    ratio = eig_max / eig_min

    print("ratio: ", ratio)

    # Problem 2. solve the evolution of the system
    dt = 0.001
    t = 0
    t_list = []
    tmax = 40.0

    u1 = np.copy(u_init)
    u2 = np.copy(u_init)
    u3 = np.copy(u_init)

    u1_list = []
    u2_list = []
    u3_list = []

    f = lambda x : compute_fu(x, k1, k2, k3)

    while t < tmax:
        t_list.append(t)
        t += dt

        u1 = RK4(u1, dt, f)
        u2 = DIRK2(u2, dt, f)
        u3 = BDF2(u3, dt, f)

        u1_list.append(u1.reshape(1,-1))
        u2_list.append(u2.reshape(1,-1))
        u3_list.append(u3.reshape(1,-1))

    tx = np.array(t_list)
    u1 = np.concatenate(u1_list, axis=0)
    u2 = np.concatenate(u2_list, axis=0)
    u3 = np.concatenate(u3_list, axis=0)

    # plot the graph
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.ravel()

    axes[0].plot(tx, u1[:, 0], "r", label=r"$X_{N}$")
    axes[0].plot(tx, u1[:, 1], "g", label=r"$X_{O}$")
    axes[0].plot(tx, u1[:, 2], "b", label=r"$X_{N_2}$")
    axes[0].plot(tx, u1[:, 3], "k", label=r"$X_{O_2}$")
    axes[0].plot(tx, u1[:, 4], "y", label=r"$X_{NO}$")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Mole fraction")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(loc = "lower left")
    axes[0].set_title(r"RK4")

    axes[1].plot(tx, u2[:, 0], "r", label=r"$X_{N}$")
    axes[1].plot(tx, u2[:, 1], "g", label=r"$X_{O}$")
    axes[1].plot(tx, u2[:, 2], "b", label=r"$X_{N_2}$")
    axes[1].plot(tx, u2[:, 3], "k", label=r"$X_{O_2}$")
    axes[1].plot(tx, u2[:, 4], "y", label=r"$X_{NO}$")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mole fraction")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend(loc="lower left")
    axes[1].set_title(r"DIRK2")

    axes[2].plot(tx, u3[:, 0], "r", label=r"$X_{N}$")
    axes[2].plot(tx, u3[:, 1], "g", label=r"$X_{O}$")
    axes[2].plot(tx, u3[:, 2], "b", label=r"$X_{N_2}$")
    axes[2].plot(tx, u3[:, 3], "k", label=r"$X_{O_2}$")
    axes[2].plot(tx, u3[:, 4], "y", label=r"$X_{NO}$")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Mole fraction")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].legend(loc="lower left")
    axes[2].set_title(r"BDF2")

    fig.tight_layout()
    fig.savefig("./p4_b.png")
