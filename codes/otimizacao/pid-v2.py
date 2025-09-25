"""
Sistema de Controle PID com Otimização para k_p, k_i e k_d por Enxame de Partículas (PSO)

O código simula um modelo do motor, com controlador e simulações para avaliar a diferença 
entre a saída do sistema e a referência desejada.

Nota: A função objetivo atualmente tá implementada com a saída ITA (Integral Time Absolute Error) 
e ITE (Integral Time Error), não usei Goodhart porque já tava com essas métricas prontas na função. 
TO DO: Precisa trocar ITA e ITE para aplicar o goodhart aqui.
"""

import numpy as np
from scipy.integrate import odeint
import random

a = 0.05
k = 2.0
a_model_error = 0.1
k_model_error = 0.3

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

def motor_model(x1_m, u, a, k):
    dx1_m = -a*k*x1_m + k*u
    return dx1_m

def motor_controller(tau, tau_ref, taup_ref, erro_acum, d_erro, k_p, k_i, k_d):
    erro = tau_ref - tau
    v = k_p * erro + k_i * ts_ms * erro_acum + k_d * d_erro / ts_ms
    return (a + a_model_error) * tau + v / (k + k_model_error)

def connected_systems_model(states, t, tau_ref, taup_ref, erro_acum, d_erro, k_p, k_i, k_d):
    x1_m = states[0]
    dc_volts = motor_controller(x1_m, tau_ref, taup_ref, erro_acum, d_erro, k_p, k_i, k_d)
    dx1_m = motor_model(x1_m, dc_volts, a, k)
    return [dx1_m]

def calcular_itaite(kp, ki, kd):
    """
    Função Fitness usando as métricas de saída do ITA e ITE.
    NOTA: Precisa implementar Goodhart ainda
    """
    n = int((1 / (ts_ms / 1000.0)) * tf + 1)
    time_vector = np.linspace(0, tf, n)
    torque_ref = np.sin(time_vector)
    torquep_ref = np.cos(time_vector)
    
    states = np.zeros((n, 1))
    states[0] = [0]
    erro_anterior = 0
    
    for i in range(n-1):
        t_span = [time_vector[i], time_vector[i+1]]
        tau = states[i, 0]
        tau_ref_i = torque_ref[i]
        taup_ref_i = torquep_ref[i]
        erro_atual = tau_ref_i - tau
        erro_acum = erro_atual + erro_anterior
        d_erro = erro_atual - erro_anterior
        
        out_states = odeint(connected_systems_model, states[i], t_span,
                            args=(tau_ref_i, taup_ref_i, erro_acum, d_erro, kp, ki, kd))
        states[i+1] = out_states[-1]
        erro_anterior = erro_atual
    
    tau = states[:, 0]
    erro = torque_ref - tau
    
    ita = np.trapz(np.abs(erro), time_vector)
    steady_state_start = int(0.9 * n)
    esa = np.mean(np.abs(erro[steady_state_start:]))
    
    j = 1.0*ita + 10.0*esa
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
gbest_fit = calcular_itaite(*gbest)
for i in range(1,n_part):
    f = calcular_itaite(*part[i])
    if f < gbest_fit:
        gbest_fit = f
        gbest = part[i][:]

print("Busca Inicial:", gbest, gbest_fit)

for it in range(max_iter):
    for i in range(n_part):
        f = calcular_itaite(*part[i])
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