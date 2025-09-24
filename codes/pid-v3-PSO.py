import numpy as np
from scipy.integrate import odeint
import random

a = 0.05
k = 2.0
a_err = 0.1
k_err = 0.3

tf = 2.0
ts_ms = 1
dt = ts_ms/1000.0

n_part = 10
max_iter = 20
lim = [(0.0,500.0),(0.0,500.0),(0.0,100.0)]

peso_inercia = 0.7
peso_local = 1.5
peso_global = 1.5
veloc_max = [(b[1]-b[0])*0.2 for b in lim]

def motor_model(x1, u, a, k, a_err=0.0, k_err=0.0):
    a_ef = a*(1+a_err)
    k_ef = k*(1+k_err)
    dx = -a_ef*k_ef*x1 + k_ef*u
    return dx

def motor_controller(e, e_acum, de, kp, ki, kd):
    u = kp*e + ki*e_acum + kd*de
    u = max(-1000.0, min(1000.0, u))
    return u

def connected_systems_model(states, t, kp, ki, kd, ref, e_acum, e_ant):
    x1 = states[0]
    e = ref - x1
    e_acum += e*dt
    de = (e - e_ant)/dt
    u = motor_controller(e, e_acum, de, kp, ki, kd)
    dx = motor_model(x1, u, a, k, a_err, k_err)
    return [dx]

def calcular_goodhart(kp, ki, kd):
    n = int(tf/dt)+1
    t_vec = np.linspace(0, tf, n)
    ref = np.sin(t_vec)
    states = np.zeros(n)
    e_acum = 0.0
    e_ant = 0.0
    for i in range(n-1):
        t_span = [t_vec[i], t_vec[i+1]]
        out_states = odeint(connected_systems_model, [states[i]], t_span,
                            args=(kp, ki, kd, ref[i], e_acum, e_ant))
        states[i+1] = out_states[-1,0]
        e_ant = ref[i] - states[i]
        e_acum += e_ant*dt
    erro = ref - states
    ita = np.trapz(np.abs(erro), t_vec)/tf
    ss = int(0.9*n)
    esa = np.mean(np.abs(erro[ss:]))
    u_eff = np.mean(np.abs(kp*erro + ki*e_acum + kd*np.gradient(erro, dt)))
    ov = max(0.0, np.max(states)-np.max(ref))
    j = 1.0*ita + 10.0*esa + 0.1*u_eff + 50.0*ov
    return j

part = []
vel = []
pbest = []
pbest_fit = []

for i in range(n_part):
    p = [random.uniform(lim[j][0],lim[j][1]) for j in range(3)]
    part.append(p)
    vel.append([random.uniform(-veloc_max[j],veloc_max[j]) for j in range(3)])
    pbest.append(p[:])
    pbest_fit.append(float('inf'))

gbest = part[0][:]
gbest_fit = calcular_goodhart(*gbest)
for i in range(1,n_part):
    f = calcular_goodhart(*part[i])
    if f < gbest_fit:
        gbest_fit = f
        gbest = part[i][:]

print("Busca Inicial:", gbest, gbest_fit)

for it in range(max_iter):
    for i in range(n_part):
        f = calcular_goodhart(*part[i])
        if f < pbest_fit[i]:
            pbest_fit[i] = f
            pbest[i] = part[i][:]
        if f < gbest_fit:
            gbest_fit = f
            gbest = part[i][:]
    for i in range(n_part):
        for d in range(3):
            r1 = random.random()
            r2 = random.random()
            vel[i][d] = (peso_inercia*vel[i][d] +
                         peso_local*r1*(pbest[i][d]-part[i][d]) +
                         peso_global*r2*(gbest[d]-part[i][d]))
            vel[i][d] = max(-veloc_max[d], min(veloc_max[d], vel[i][d]))
            part[i][d] += vel[i][d]
            part[i][d] = max(lim[d][0], min(lim[d][1], part[i][d]))
    if it%5==0:
        print(f"Iter {it}: {gbest_fit:.6f}")

print("Final:", gbest, gbest_fit)