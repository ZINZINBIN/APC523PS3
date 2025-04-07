import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def compute_dynamics(u:np.ndarray, t:float, w:float, wf:float, Fm:float):
    if u.ndim == 1:
        u = u.reshape(-1,1)

    fu = np.zeros_like(u)
    fu[0] = u[1]
    fu[1] = - w** 2 * u[0] + Fm * np.cos(wf * t)
    
    return fu

def forward_euler(u:np.ndarray, dt:float, t:float, func:Callable):
    return u + dt * func(u,t)

def symplectic_euler(u:np.ndarray, dt:float, t:float, func:Callable):
    
    uf = np.zeros_like(u)
    du1 = func(u,t)
    du2 = func(u + du1 * dt, t)
    
    uf[0] = u[0] + du2[0] * dt
    uf[1] = u[1] + du1[1] * dt
    
    return uf

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

    ts = np.linspace(0,100,1000,endpoint=True)
    dt = ts[1] - ts[0]

    u_fe = np.copy(u0)
    u_se = np.copy(u0)
    u_rk = np.copy(u0)

    us_fe = [] # Forward euler
    us_se = [] # Symplectic euler
    us_rk = [] # RK4

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

        u_fe = forward_euler(u_fe, dt, t, fu)
        u_se = symplectic_euler(u_se, dt, t, fu)
        u_rk = RK4(u_rk, dt, t, fu)

    us_fe = np.concatenate(us_fe, axis = 1).reshape(-1,2)
    us_se = np.concatenate(us_se, axis = 1).reshape(-1,2)
    us_rk = np.concatenate(us_rk, axis = 1).reshape(-1,2)

    us_gt = np.concatenate([analytic_solution(t) for t in ts], axis = 1).reshape(-1,2)

    err_fe = np.sqrt((us_gt[:,0] - us_fe[:,0]) ** 2)
    err_se = np.sqrt((us_gt[:,0] - us_se[:,0]) ** 2)
    err_rk = np.sqrt((us_gt[:,0] - us_rk[:,0]) ** 2)

    fig, ax = plt.subplots(1,1,figsize = (6,4))

    # ax.plot(ts, us_fe[:,0], 'r', label="Forward Euler")
    ax.plot(ts, us_se[:,0], "g", label="Symplectic Euler")
    ax.plot(ts, us_rk[:,0], "b", label="Fourth-order RK")
    ax.plot(ts, us_gt[:,0], "k", label="Analytic solution")
    ax.set_xlabel("time")
    ax.set_ylabel("$x(t)$")
    ax.set_title(r"$t$ vs $x(t)$")
    ax.legend(loc = "upper right")
    fig.tight_layout()
    fig.savefig("./p2_response.png")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # ax.plot(us_fe[:,1], us_fe[:, 0], "r", label="Forward Euler")
    ax.plot(us_se[:,1], us_se[:, 0], "g", label="Symplectic Euler")
    ax.plot(us_rk[:,1], us_rk[:, 0], "b", label="Fourth-order RK")
    ax.plot(us_gt[:,1], us_gt[:,0], "k", label="Analytic solution")
    ax.set_xlabel("$v(t)$")
    ax.set_ylabel("$x(t)$")
    ax.set_title(r"phase portrait")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("./p2_phase.png")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # ax.plot(ts, err_fe, "r", label="Forward Euler")
    ax.plot(ts, err_se, "g", label="Symplectic Euler")
    ax.plot(ts, err_rk, "b", label="Fourth-order RK")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$e(t)$")
    ax.set_yscale("log")
    ax.set_title(r"$|e(t)|_{L_2}$ vs t")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("./p2_error.png")
