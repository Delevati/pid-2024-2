"""
    This code has been developed by Juan Sandubete Lopez and all the rights
    belongs to him.
    Distribution or commercial use of the code is not allowed without previous
    agreement with the author.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import time
import pandas as pd

# Simulation parametrs
tf = 6.0  # final time
ts_ms = 0.01  # 0.001 = 1us, 1 = 1ms
save_data = False  # Attention: CSV file can be very big
show_fig = True
save_fig = False  # If False, figure is showed but not saved
title = "generic_control_system"

print("Starting generic control system simulation.")

# Models Parameters
# Generic Plant (First-order system: dx = -a*x + b*u)
plant_a = 1.0  # System pole
plant_b = 1.0  # System gain
plant_a_error = 0.1  # Model uncertainty in pole
plant_b_error = 0.3  # Model uncertainty in gain

# Generic Controller (PID)
kp = 1.0  # Proportional gain
ki = 0.0  # Integral gain
kd = 0.1  # Derivative gain

print("\n--- PARAMETERS --- \n ")
print("Plant Parameters: a = {}, b = {}".format(plant_a, plant_b))
print("Plant Model Errors: a_error = {}, b_error = {}".
      format(plant_a_error, plant_b_error))
print("Controller Gains: kp = {}, ki = {}, kd = {}".format(kp, ki, kd))

# Define models
def generic_plant_model(x, u):
    # Generic first-order system model:
    # dx/dt = -a*x + b*u
    # Where x is the system state and u is the control input
    dx = -plant_a*x + plant_b*u
    y = x  # Output equals state
    return dx

def generic_controller(output, reference, derivative_ref, integral_error=0):
    # Generic PID controller
    # Returns control input u
    error = reference - output
    
    # PID control law with model compensation
    # Basic PID: u = kp*e + ki*∫e + kd*(dref - dy)
    u_pid = kp*error + ki*integral_error + kd*derivative_ref
    
    # Add model-based feedforward (with uncertainty)
    u_ff = (plant_a + plant_a_error)*output
    
    # Total control input (accounting for plant gain uncertainty)
    u_total = (u_pid + u_ff) / (plant_b + plant_b_error)
    
    return u_total


# Function to generate different reference signals
def generate_reference_signal(time_vector, signal_type='sine', amplitude=1.0, frequency=1.0):
    """Generate different types of reference signals"""
    if signal_type == 'sine':
        return amplitude * np.sin(frequency * time_vector)
    elif signal_type == 'step':
        return amplitude * np.ones_like(time_vector)
    elif signal_type == 'ramp':
        return amplitude * time_vector
    elif signal_type == 'square':
        return amplitude * np.sign(np.sin(frequency * time_vector))
    else:
        return np.zeros_like(time_vector)

# The following function puts all equations together
def connected_systems_model(states, t, output_ref, output_dot_ref):
    # Input values. Check this with the out_states list
    system_output, integral_error = states

    # Compute generic controller
    control_input = generic_controller(system_output, output_ref, output_dot_ref, integral_error)
    
    # Compute plant response
    system_output_dot = generic_plant_model(system_output, control_input)
    
    # Update integral error for PI/PID control
    error = output_ref - system_output
    integral_error_dot = error

    # Output [system_output_dot, integral_error_dot]
    out_states = [system_output_dot, integral_error_dot]
    return out_states


# Initial conditions [system_output, integral_error]
states0 = [0.0, 0.0]
n = int((1 / (ts_ms / 1000.0))*tf + 1) # number of time points

# time span for the simulation, cycle every tf/n seconds
time_vector = np.linspace(0,tf,n)
t_sim_step = time_vector[1] - time_vector[0]

# Reference signal configuration
reference_type = 'sine'  # Options: 'sine', 'step', 'ramp', 'square'
reference_amplitude = 1.0
reference_frequency = 1.0

# Generate reference signal and its derivative
output_ref = generate_reference_signal(time_vector, reference_type, reference_amplitude, reference_frequency)
print("Reference type: {}".format(reference_type))
print("Max ref: {:.3f}".format(max(output_ref)))
print("Min ref: {:.3f}".format(min(output_ref)))

# Calculate derivative of reference (for feedforward)
if reference_type == 'sine':
    output_dot_ref = reference_amplitude * reference_frequency * np.cos(reference_frequency * time_vector)
else:
    output_dot_ref = np.gradient(output_ref, t_sim_step)

print("Max ref derivative: {:.3f}".format(max(output_dot_ref)))
print("Min ref derivative: {:.3f}".format(min(output_dot_ref)))
# Output arrays
states = np.zeros( (n-1, len(states0)) ) # States for each timestep

print("\n--- SIMULATION CONFIG. ---\n")
print("Simulation time: {} sec".format(tf))
print("Time granulatiry: {}".format(t_sim_step))
print("Initial states: {}".format(states0))

print("\n--- SIMULATION Begins ---\n")

initial_time = time.time()
# Simulate with ODEINT
t_counter = 0
for i in range(n-1):
    out_states = odeint(connected_systems_model, states0, [0.0, tf/n],
                        args=(output_ref[i], output_dot_ref[i]))
    states0 = out_states[-1,:]
    states[i] = out_states[-1,:]
    if i >= t_counter * int((n-1)/10):
        print("Simulation at {}%".format(t_counter*10))
        t_counter += 1

elapsed_time = time.time() - initial_time
print("\nElapsed time: {} sec.".format(elapsed_time))
print("\n--- SIMULATION Finished. ---\n")

if save_data:
    print("Saving simulation data...")
    # Create DataFrame with time, states, and reference
    data_dict = {
        'time': time_vector[:-1],
        'system_output': states[:, 0],
        'integral_error': states[:, 1],
        'reference': output_ref[:-1],
        'reference_derivative': output_dot_ref[:-1]
    }
    sim_df = pd.DataFrame(data_dict)
    sim_df.to_csv('sim_data/generic_control_system.csv', index=False)

# Plot results
# States are: [system_output, integral_error]
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(12, 8))

# Plot 1: System output vs reference
plt.subplot(3,1,1)
plt.plot(time_vector[:-1], output_ref[:-1], 'k--', linewidth=3, label='Reference')
plt.plot(time_vector[:-1], states[:,0], 'r', linewidth=2, label='System Output')
plt.ylabel('Output')
plt.legend()
plt.title(f'Generic Control System - Reference Type: {reference_type}')

# Plot 2: Control error
plt.subplot(3,1,2)
error_signal = output_ref[:-1] - states[:,0]
plt.plot(time_vector[:-1], error_signal, 'b', linewidth=2, label='Error')
plt.ylabel('Error')
plt.legend()

# Plot 3: Reference derivative (for feedforward)
plt.subplot(3,1,3)
plt.plot(time_vector[:-1], output_dot_ref[:-1], 'g--', linewidth=2, label='Reference Derivative')
plt.ylabel('Reference Derivative')
plt.xlabel('Time [s]')
plt.legend()

if save_fig:
    figname = "pictures/" + title + ".png"
    plt.savefig(figname)
if show_fig:
    plt.show()
