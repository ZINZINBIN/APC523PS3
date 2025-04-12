import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable
from functools import wraps

# Wrapper for measuring the execution time
def check_time(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        et = time.time()
        dt = et - st
        return result, dt

    return wrapper

def compute_dynamics(u:np.ndarray, t:float, w:float, wf:float, Fm:float):
    if u.ndim == 1:
        u = u.reshape(-1,1)

    fu = np.zeros_like(u)
    fu[0] = u[1]
    fu[1] = - w** 2 * u[0] + Fm * np.cos(wf * t)
    
    return fu

def compute_hamiltonian(u:np.ndarray, t:float, w:float, wf:float, Fm:float):

    u = u.ravel()
    x = u[0]
    v = u[1]

    KE = 0.5 * v ** 2
    PE = 0.5 * w ** 2 * x ** 2
    FE = Fm * np.cos(wf * t) * x * (-1)

    return KE + PE + FE

@check_time
def forward_euler(u:np.ndarray, dt:float, t:float, func:Callable):
    return u + dt * func(u,t)

@check_time
def symplectic_euler(u:np.ndarray, dt:float, t:float, func:Callable):
    uf = np.zeros_like(u, dtype=float)
    u_tmp = u + func(u,t) * dt
    uf[0] = u[0] + u_tmp[1] * dt
    uf[1] = u_tmp[1]
    return uf

@check_time
def RK4(u:np.ndarray, dt:float, t:float, func:Callable):
    k1 = func(u,t)
    k2 = func(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = func(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = func(u + dt * k3, t + dt)
    uf = u + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return uf

if __name__ == "__main__":

    # initial condition for system
    w = 5.0
    Fm = 1.0
    wf = 0.1
    u0 = np.array([0,0]).reshape(-1,1)

    ts = np.linspace(0,100,10000,endpoint=True)
    dt = ts[1] - ts[0]

    u_fe = np.copy(u0)
    u_se = np.copy(u0)
    u_rk = np.copy(u0)

    us_fe = [] # Forward euler
    us_se = [] # Symplectic euler
    us_rk = [] # RK4

    hs_fe = []
    hs_se = []
    hs_rk = []

    tc_fe_list = []
    tc_se_list = []
    tc_rk_list = []

    def fu(u:np.ndarray, t:float):
        out = compute_dynamics(u,t,w,wf,Fm)
        return out

    def analytic_solution(t:float):
        x = Fm / (w**2 - wf**2) * (np.cos(wf*t) - np.cos(w*t))
        v = Fm / (w**2 - wf**2) * ((-wf) * np.sin(wf*t) + w*np.sin(w*t))
        u = np.array([x,v]).reshape(-1,1)
        return u

    for t in ts:

        us_fe.append(np.copy(u_fe))
        us_se.append(np.copy(u_se))
        us_rk.append(np.copy(u_rk))

        u_fe, tc_fe = forward_euler(u_fe, dt, t, fu)
        u_se, tc_se = symplectic_euler(u_se, dt, t, fu)
        u_rk, tc_rk = RK4(u_rk, dt, t, fu)

        h_fe = compute_hamiltonian(u_fe, t, w, wf, Fm)
        h_se = compute_hamiltonian(u_se, t, w, wf, Fm)
        h_rk = compute_hamiltonian(u_rk, t, w, wf, Fm)

        hs_fe.append(h_fe)
        hs_se.append(h_se)
        hs_rk.append(h_rk)

        tc_fe_list.append(tc_fe)
        tc_se_list.append(tc_se)
        tc_rk_list.append(tc_rk)

    us_fe = np.concatenate(us_fe, axis = 1).T
    us_se = np.concatenate(us_se, axis = 1).T
    us_rk = np.concatenate(us_rk, axis = 1).T

    tc_fe = np.array(tc_fe_list)
    tc_se = np.array(tc_se_list)
    tc_rk = np.array(tc_rk_list)

    hs_fe = np.array(hs_fe)
    hs_se = np.array(hs_se)
    hs_rk = np.array(hs_rk)

    us_gt = np.concatenate([analytic_solution(t) for t in ts], axis = 1).T
    hs_gt = np.array([compute_hamiltonian(u_gt, t, w, wf, Fm) for u_gt, t in zip(us_gt, ts)])

    err_fe = np.sqrt((us_gt[:,0] - us_fe[:,0]) ** 2)
    err_se = np.sqrt((us_gt[:,0] - us_se[:,0]) ** 2)
    err_rk = np.sqrt((us_gt[:,0] - us_rk[:,0]) ** 2)

    err_h_fe = np.sqrt((hs_gt - hs_fe) ** 2)
    err_h_se = np.sqrt((hs_gt - hs_se) ** 2)
    err_h_rk = np.sqrt((hs_gt - hs_rk) ** 2)

    fig, axes = plt.subplots(1,4,figsize = (16,4))
    axes = axes.ravel()

    axes[0].plot(ts, us_fe[:, 0], "r", label="Forward Euler")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("$x(t)$")
    axes[0].set_title(r"Forward Euler")

    axes[1].plot(ts, us_se[:, 0], "g", label="Symplectic Euler")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("$x(t)$")
    axes[1].set_title(r"Symplectic Euler")

    axes[2].plot(ts, us_rk[:, 0], "b", label="RK4")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("$x(t)$")
    axes[2].set_title(r"RK4")

    axes[3].plot(ts, us_gt[:, 0], "k", label="Analytic solution")
    axes[3].set_xlabel("time")
    axes[3].set_ylabel("$x(t)$")
    axes[3].set_title(r"Analytic solution")

    fig.tight_layout()
    fig.savefig("./p2_response.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.ravel()

    axes[0].plot(us_fe[:, 0], ts, "r", label="Forward Euler")
    axes[0].set_ylabel("time")
    axes[0].set_xlabel("$x(t)$")
    axes[0].set_title(r"Forward Euler")

    axes[1].plot(us_se[:, 0],ts, "g", label="Symplectic Euler")
    axes[1].set_ylabel("time")
    axes[1].set_xlabel("$x(t)$")
    axes[1].set_title(r"Symplectic Euler")

    axes[2].plot(us_rk[:, 0], ts, "b", label="RK4")
    axes[2].set_ylabel("time")
    axes[2].set_xlabel("$x(t)$")
    axes[2].set_title(r"RK4")

    axes[3].plot(us_gt[:, 0], ts, "k", label="Analytic solution")
    axes[3].set_ylabel("time")
    axes[3].set_xlabel("$x(t)$")
    axes[3].set_title(r"Analytic solution")

    fig.tight_layout()
    fig.savefig("./p2_response_xt.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.ravel()

    axes[0].plot(us_fe[:, 1], us_fe[:,0], "r", label="Forward Euler")
    axes[0].set_xlabel("$v(t)$")
    axes[0].set_ylabel("$x(t)$")
    axes[0].set_title(r"Forward Euler")

    axes[1].plot(us_se[:,1], us_se[:, 0], "g", label="Symplectic Euler")
    axes[1].set_xlabel("$v(t)$")
    axes[1].set_ylabel("$x(t)$")
    axes[1].set_title(r"Symplectic Euler")

    axes[2].plot(us_rk[:,1], us_rk[:, 0], "b", label="RK4")
    axes[2].set_xlabel("$v(t)t$")
    axes[2].set_ylabel("$x(t)$")
    axes[2].set_title(r"RK4")

    axes[3].plot(us_gt[:,1], us_gt[:, 0], "k", label="Analytic solution")
    axes[3].set_xlabel("$v(t)$")
    axes[3].set_ylabel("$x(t)$")
    axes[3].set_title(r"Analytic solution")

    fig.tight_layout()
    fig.savefig("./p2_phase.png")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(ts, err_fe, "r", label="Forward Euler")
    ax.plot(ts, err_se, "g", label="Symplectic Euler")
    ax.plot(ts, err_rk, "b", label="RK4")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$e(t)$")
    ax.set_yscale("log")
    ax.set_title(r"$|e(t)|_{L_2}$ vs t")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("./p2_error.png")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.ravel()

    axes[0].plot(ts, hs_fe, "r", label="Forward Euler")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$H(t)$")
    axes[0].set_title(r"Forward Euler")

    axes[1].plot(ts, hs_se, "g", label="Symplectic Euler")
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("$H(t)$")
    axes[1].set_title(r"Symplectic Euler")

    axes[2].plot(ts, hs_rk, "b", label="RK4")
    axes[2].set_xlabel("$t$")
    axes[2].set_ylabel("$H(t)$")
    axes[2].set_title(r"RK4")

    axes[3].plot(ts, hs_gt, "k", label="Analytic")
    axes[3].set_xlabel("$t$")
    axes[3].set_ylabel("$H(t)$")
    axes[3].set_title(r"Analytic solution")

    fig.tight_layout()
    fig.savefig("./p2_energy.png")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(ts, err_h_fe, "r", label="Forward Euler")
    ax.plot(ts, err_h_se, "g", label="Symplectic Euler")
    ax.plot(ts, err_h_rk, "b", label="RK4")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$e(t)_{H}$")
    ax.set_yscale("log")
    ax.set_title(r"$|e(t)|_{H}$ vs t")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("./p2_energy_error.png")

    # Computaitonal cost for each time integration
    tc_fe_mean, tc_fe_dev = np.mean(tc_fe), np.std(tc_fe)
    tc_se_mean, tc_se_dev = np.mean(tc_se), np.std(tc_se)
    tc_rk_mean, tc_rk_dev = np.mean(tc_rk), np.std(tc_rk)

    print("Average computational time required for Forward Euler:{}".format(tc_fe_mean))
    print("Average computational time required for Symplectic Euler:{}".format(tc_se_mean))
    print("Average computational time required for RK4:{}".format(tc_rk_mean))
