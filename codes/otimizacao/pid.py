import numpy as np
import pandas as pd
from scipy.integrate import odeint

#primeiro estado do motor
x1_m = 0

#valor empirico de k
k_c1 = 0.1
k = 1

v = 2
t = 0
taup = 0

u = 0
a = 0.1

tf = 10
ts_ms = 0.1
save_data = False

a_model_error = 0.1
k_model_error = 0.3

def motor_model(x1_m, u, a, k):
    dx1_m = -a*k*x1_m + k*u
    y1_m = x1_m
    return dx1_m, y1_m

def motor_controller(tau, tau_ref, taup_ref):
    v = taup_ref - k_c1*(tau - tau_ref)
    return (a+a_model_error)*tau + v/(k+k_model_error)

def connected_systems_model(states, t, tau_ref, taup_ref):
    x1_m, _ = states
    dc_volts = motor_controller(x1_m, tau_ref, taup_ref)

    tau, taup = motor_model(x1_m, dc_volts, a, k)
    out_states = [tau, dc_volts]
    return out_states

states0 = [0, 0]
n = int((1/ (ts_ms / 1000.0)) * tf + 1)
time_vector = np.linspace(0, tf, n)
t_sim_step = time_vector [1] - time_vector[0]

torque_ref = np.sin(time_vector)
torquep_ref = np.cos(time_vector)
states = np.zeros( (n-1, len(states0)) )

for i in range(n-1):
    out_states = odeint(connected_systems_model, states0,[0.0, tf/n],
                        args=(torque_ref[i], torquep_ref[i]))
    states0 = out_states[-1, :]

    states[i] = out_states[-1, :]

    if save_data:
        print("Saving simulation data...")
        sim_df = pd.DataFrame(states)
        sim_df = sim_df.transpose()
        sim_df.rename({0: 'tau', 1: 'tau_ref', 2: 'taup_ref',
                    3: 'dc_volts'}, inplace=True)
        sim_df.to_csv('ex4_motor_control.csv')
        
