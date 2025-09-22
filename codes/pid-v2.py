import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# adicionados new_params PID
k_p = 350
k_i = 300
k_d = 10

# k_c1 Muito alto causa oscilações, sinal de controle pode saturar
k_c1 = 20

# k = ganho do motor
k = 2
# a = tempo motor
a = 0.05

# Tempos de execução, tf = tempo final
tf = 20
# ts_ms = tempo de compasso de amostragem
ts_ms = 0.5

a_model_error = 0.1
k_model_error = 0.3

# True pra salvar a saída
save_data = False

erro_atual = 0
erro_anterior = 0

def motor_model(x1_m, u, a, k):
    dx1_m = -a*k*x1_m + k*u
    y1_m = x1_m
    return dx1_m

def motor_controller(tau, tau_ref, taup_ref, erro_acum, d_erro, erro):
    # v = taup_ref - k_c1*(tau - tau_ref)
    #erro_acum = erro_atual + erro_anterior
    #d_erro = erro_atual - erro_anterior
    erro = tau_ref - tau
    v = k_p * erro + k_i * ts_ms * erro_acum + k_d * d_erro / ts_ms
    return (a+a_model_error)*tau + v/(k+k_model_error)

def connected_systems_model(states, t, tau_ref, taup_ref, erro_acum, d_erro, erro):
    x1_m = states[0]
    dc_volts = motor_controller(x1_m, tau_ref, taup_ref, erro_acum, d_erro, erro)
    dx1_m = motor_model(x1_m, dc_volts, a, k)
    return [dx1_m]


states0 = [0]
n = int((1/ (ts_ms / 1000.0)) * tf + 1)
time_vector = np.linspace(0, tf, n)
torque_ref = np.sin(time_vector)
torquep_ref = np.cos(time_vector)

# Para armazenar resultados
states = np.zeros((n, 1))
states[0] = states0

for i in range(n-1):
    t_span = [time_vector[i], time_vector[i+1]]
    tau = states[i, 0]
    tau_ref_i = torque_ref[i]
    taup_ref_i = torquep_ref[i]
    erro_atual = tau_ref_i - tau
    erro_acum = erro_atual + erro_anterior
    d_erro = erro_atual - erro_anterior
    out_states = odeint(connected_systems_model, states[i], t_span,
                        args=(tau_ref_i, taup_ref_i, erro_acum, d_erro, erro_atual))
    states[i+1] = out_states[-1]
    erro_anterior = erro_atual

# Cálculo do erro
tau = states[:, 0]
erro = torque_ref - tau

# ITA: Integral do Tempo Absoluto do Erro
ITA = np.trapz(np.abs(erro), time_vector)
print(f"ITA (Integral do Tempo Absoluto do Erro): {ITA:.4f}")

# ESA: Erro em Steady-State Absoluto (últimos 10% do tempo)
steady_state_start = int(0.9 * n)
ESA = np.mean(np.abs(erro[steady_state_start:]))
print(f"ESA (Erro em Steady-State Absoluto): {ESA:.4f}")

# plot config
plt.figure(figsize=(10,5))
plt.plot(time_vector, tau, label='tau (saída)')
plt.plot(time_vector, torque_ref, label='tau_ref (referência)', linestyle='--')
plt.xlabel('Tempo (s)')
plt.ylabel('Torque')
plt.legend()
plt.title('Resposta do Sistema')
plt.grid()
plt.show()