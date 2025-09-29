"""
    Otimização PID usando Particle Swarm Optimization (PSO)
    
    Este código implementa PSO para encontrar os parâmetros ótimos Kp, Ki, Kd
    do controlador PID no sistema de controle genérico.
    
    Objetivo: Minimizar RMSE do erro de controle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

# CONFIGURAÇÃO DO SISTEMA DE CONTROLE

# Parâmetros da planta (fixos)
plant_a = 1.0
plant_b = 1.0
plant_a_error = 0.1
plant_b_error = 0.3

# Configuração da simulação
tf = 4.0  # tempo de simulação reduzido para PSO
n = 400   # pontos de simulação (otimizado para velocidade)
reference_type = 'step'  # melhor para avaliar performance PID
reference_amplitude = 1.0
reference_frequency = 1.0

print(" OTIMIZAÇÃO PID")
print(f"Sistema: dx/dt = -{plant_a}*x + {plant_b}*u")
print(f"Referência: {reference_type}")
print(f"Tempo de simulação: {tf}s")
print()

# MODELOS DO SISTEMA

def generic_plant_model(x, u):
    dx = -plant_a*x + plant_b*u
    return dx

def generic_controller(output, reference, derivative_ref, integral_error, kp, ki, kd):
    """Controlador PID com parâmetros variáveis"""
    error = reference - output
    u_pid = kp*error + ki*integral_error + kd*derivative_ref
    u_ff = (plant_a + plant_a_error)*output
    u_total = (u_pid + u_ff) / (plant_b + plant_b_error)
    return u_total

def generate_reference_signal(time_vector, signal_type='sine', amplitude=1.0, frequency=1.0):
    if signal_type == 'sine':
        return amplitude * np.sin(frequency * time_vector)
    elif signal_type == 'step':
        return amplitude * np.ones_like(time_vector)
    elif signal_type == 'ramp':
        return amplitude * time_vector / time_vector[-1]
    elif signal_type == 'square':
        return amplitude * np.sign(np.sin(frequency * time_vector))
    else:
        return np.zeros_like(time_vector)

def connected_systems_model(states, t, output_ref, output_dot_ref, pid_params):
    """Sistema conectado com parâmetros PID variáveis"""
    system_output, integral_error = states
    kp, ki, kd = pid_params
    
    control_input = generic_controller(system_output, output_ref, output_dot_ref, 
                                     integral_error, kp, ki, kd)
    system_output_dot = generic_plant_model(system_output, control_input)
    error = output_ref - system_output
    integral_error_dot = error
    
    return [system_output_dot, integral_error_dot]

# FUNÇÃO OBJETIVO PARA PSO

def evaluate_pid(pid_params):
    """
    Avalia a performance de um conjunto de parâmetros PID
    Retorna: custo (menor é melhor)
    """
    kp, ki, kd = pid_params
    
    # Verificar se os parâmetros estão dentro dos limites válidos
    if kp < 0 or ki < 0 or kd < 0:
        return 1e6  # penalidade alta para valores inválidos
    
    try:
        # Configurar simulação
        time_vector = np.linspace(0, tf, n)
        t_sim_step = time_vector[1] - time_vector[0]
        
        # Gerar referência
        output_ref = generate_reference_signal(time_vector, reference_type, 
                                             reference_amplitude, reference_frequency)
        
        if reference_type == 'sine':
            output_dot_ref = reference_amplitude * reference_frequency * np.cos(reference_frequency * time_vector)
        else:
            output_dot_ref = np.gradient(output_ref, t_sim_step)
        
        # Condições iniciais
        states0 = [0.0, 0.0]
        states = np.zeros((n-1, 2))
        
        # Simular sistema
        for i in range(n-1):
            out_states = odeint(connected_systems_model, states0, [0.0, tf/n],
                              args=(output_ref[i], output_dot_ref[i], pid_params))
            states0 = out_states[-1, :]
            states[i] = out_states[-1, :]
        
        # Calcular métricas de performance
        error_signal = output_ref[:-1] - states[:, 0]
        
        # RMSE (objetivo principal)
        rmse = np.sqrt(np.mean(error_signal**2))
        
        # Penalidades adicionais
        max_error = np.max(np.abs(error_signal))
        overshoot = 0
        
        if reference_type == 'step':
            # Calcular overshoot para step response
            max_response = np.max(states[:, 0])
            if max_response > reference_amplitude:
                overshoot = (max_response - reference_amplitude) / reference_amplitude
        
        # Penalidade por control effort excessivo
        control_effort = 0
        for i in range(len(states)):
            if i < len(output_ref)-1:
                u = generic_controller(states[i, 0], output_ref[i], output_dot_ref[i], 
                                     states[i, 1], kp, ki, kd)
                control_effort += abs(u)
        
        control_effort = control_effort / len(states)
        
        # Função de custo combinada
        cost = rmse + 0.1 * overshoot + 0.01 * max(0, control_effort - 10)
        
        return cost
        
    except Exception as e:
        # Se houver erro na simulação, retornar custo alto
        return 1e6

# ALGORITMO PSO

class PSO:
    def __init__(self, num_particles=30, num_iterations=50, bounds=None):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.bounds = bounds if bounds else [(0.1, 20), (0.0, 10), (0.0, 5)]  # [kp, ki, kd]
        self.dimension = 3  # kp, ki, kd
        
        # Parâmetros do PSO
        self.w = 0.7        # inércia
        self.c1 = 1.5       # componente cognitiva
        self.c2 = 1.5       # componente social
        
        # Inicializar enxame
        self.particles = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(num_particles, self.dimension)
        )
        
        # Velocidades iniciais
        self.velocities = np.random.uniform(-1, 1, (num_particles, self.dimension))
        
        # Melhores posições
        self.personal_best = self.particles.copy()
        self.personal_best_cost = np.full(num_particles, float('inf'))
        
        self.global_best = None
        self.global_best_cost = float('inf')
        
        # Histórico
        self.cost_history = []
        
    def optimize(self):
        """Executar otimização PSO"""
        
        print(f"Iniciando PSO com {self.num_particles} partículas, {self.num_iterations} iterações")
        print("Limites de busca:")
        print(f"  Kp: [{self.bounds[0][0]:.1f}, {self.bounds[0][1]:.1f}]")
        print(f"  Ki: [{self.bounds[1][0]:.1f}, {self.bounds[1][1]:.1f}]")
        print(f"  Kd: [{self.bounds[2][0]:.1f}, {self.bounds[2][1]:.1f}]")
        print()
        
        start_time = time.time()
        
        for iteration in range(self.num_iterations):
            iter_start = time.time()
            
            # Avaliar todas as partículas
            for i in range(self.num_particles):
                cost = evaluate_pid(self.particles[i])
                
                # Atualizar melhor pessoal
                if cost < self.personal_best_cost[i]:
                    self.personal_best_cost[i] = cost
                    self.personal_best[i] = self.particles[i].copy()
                
                # Atualizar melhor global
                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best = self.particles[i].copy()
            
            # Atualizar velocidades e posições
            for i in range(self.num_particles):
                # Componentes aleatórios
                r1 = np.random.random(self.dimension)
                r2 = np.random.random(self.dimension)
                
                # Atualizar velocidade
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                    self.c2 * r2 * (self.global_best - self.particles[i])
                )
                
                # Atualizar posição
                self.particles[i] += self.velocities[i]
                
                # Aplicar limites
                for j in range(self.dimension):
                    if self.particles[i, j] < self.bounds[j][0]:
                        self.particles[i, j] = self.bounds[j][0]
                        self.velocities[i, j] = 0
                    elif self.particles[i, j] > self.bounds[j][1]:
                        self.particles[i, j] = self.bounds[j][1]
                        self.velocities[i, j] = 0
            
            self.cost_history.append(self.global_best_cost)
            
            iter_time = time.time() - iter_start
            
            # Progresso
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"Iteração {iteration+1:2d}/{self.num_iterations} | "
                      f"Melhor custo: {self.global_best_cost:.6f} | "
                      f"Tempo: {iter_time:.2f}s")
                print(f"    Kp={self.global_best[0]:.3f}, Ki={self.global_best[1]:.3f}, Kd={self.global_best[2]:.3f}")
        
        total_time = time.time() - start_time
        print(f"\nOtimização concluída em {total_time:.2f} segundos")
        
        return self.global_best, self.global_best_cost

# OTIMIZAÇÃO

def run_optimization():
    """Executar otimização PSO e mostrar resultados"""
    
    # Parâmetros originais para comparação
    original_params = [1.0, 0.0, 0.1]  # Do arquivo original
    original_cost = evaluate_pid(original_params)
    
    print("PARÂMETROS ORIGINAIS")
    print(f"Kp={original_params[0]:.3f}, Ki={original_params[1]:.3f}, Kd={original_params[2]:.3f}")
    print(f"Custo original: {original_cost:.6f}")
    print()
    
    # Executar PSO
    pso = PSO(num_particles=20, num_iterations=30)
    best_params, best_cost = pso.optimize()
    
    print("\n RESULTADOS")
    print(f"Melhores parâmetros encontrados:")
    print(f"  Kp = {best_params[0]:.4f}")
    print(f"  Ki = {best_params[1]:.4f}")
    print(f"  Kd = {best_params[2]:.4f}")
    print(f"Custo ótimo: {best_cost:.6f}")
    print(f"Melhoria: {((original_cost - best_cost) / original_cost * 100):.1f}%")
    
    return original_params, best_params, pso.cost_history

def compare_results(original_params, optimized_params):
    """Comparar performance dos parâmetros originais vs otimizados"""
    
    print("\n COMPARAÇÃO")
    
    # Simular ambos os sistemas
    time_vector = np.linspace(0, tf, n)
    t_sim_step = time_vector[1] - time_vector[0]
    output_ref = generate_reference_signal(time_vector, reference_type, 
                                         reference_amplitude, reference_frequency)
    
    if reference_type == 'sine':
        output_dot_ref = reference_amplitude * reference_frequency * np.cos(reference_frequency * time_vector)
    else:
        output_dot_ref = np.gradient(output_ref, t_sim_step)
    
    results = {}
    
    for name, params in [("Original", original_params), ("Otimizado", optimized_params)]:
        states0 = [0.0, 0.0]
        states = np.zeros((n-1, 2))
        
        for i in range(n-1):
            out_states = odeint(connected_systems_model, states0, [0.0, tf/n],
                              args=(output_ref[i], output_dot_ref[i], params))
            states0 = out_states[-1, :]
            states[i] = out_states[-1, :]
        
        error_signal = output_ref[:-1] - states[:, 0]
        rmse = np.sqrt(np.mean(error_signal**2))
        max_error = np.max(np.abs(error_signal))
        
        # Settling time (para step response)
        settling_time = None
        if reference_type == 'step':
            tolerance = 0.02 * reference_amplitude
            for i in range(len(states[:, 0])):
                if np.all(np.abs(states[i:, 0] - reference_amplitude) <= tolerance):
                    settling_time = time_vector[i]
                    break
        
        results[name] = {
            'params': params,
            'states': states,
            'rmse': rmse,
            'max_error': max_error,
            'settling_time': settling_time
        }
        
        print(f"\n{name}:")
        print(f"  Kp={params[0]:.4f}, Ki={params[1]:.4f}, Kd={params[2]:.4f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Erro Máximo: {max_error:.6f}")
        if settling_time:
            print(f"  Tempo de Acomodação: {settling_time:.3f}s")
    
    # Plotar comparação
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Resposta do sistema
    plt.subplot(2, 3, 1)
    plt.plot(time_vector[:-1], output_ref[:-1], 'k--', linewidth=2, label='Referência')
    plt.plot(time_vector[:-1], results['Original']['states'][:, 0], 'r', 
             linewidth=2, label='Original')
    plt.plot(time_vector[:-1], results['Otimizado']['states'][:, 0], 'b', 
             linewidth=2, label='Otimizado')
    plt.ylabel('Saída')
    plt.legend()
    plt.title('Resposta do Sistema')
    plt.grid(True)
    
    # Gráfico 2: Erro de controle
    plt.subplot(2, 3, 2)
    error_orig = output_ref[:-1] - results['Original']['states'][:, 0]
    error_opt = output_ref[:-1] - results['Otimizado']['states'][:, 0]
    plt.plot(time_vector[:-1], error_orig, 'r', linewidth=2, label='Original')
    plt.plot(time_vector[:-1], error_opt, 'b', linewidth=2, label='Otimizado')
    plt.ylabel('Erro')
    plt.legend()
    plt.title('Erro de Controle')
    plt.grid(True)
    
    # Gráfico 3: Convergência do PSO
    plt.subplot(2, 3, 3)
    plt.plot(pso_history, 'g', linewidth=2)
    plt.ylabel('Custo')
    plt.xlabel('Iteração')
    plt.title('Convergência do PSO')
    plt.grid(True)
    
    # Gráfico 4: Comparação de métricas
    plt.subplot(2, 3, 4)
    metrics = ['RMSE', 'Erro Máx']
    orig_vals = [results['Original']['rmse'], results['Original']['max_error']]
    opt_vals = [results['Otimizado']['rmse'], results['Otimizado']['max_error']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, orig_vals, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_vals, width, label='Otimizado', color='blue', alpha=0.7)
    plt.xlabel('Métricas')
    plt.ylabel('Valor')
    plt.title('Comparação de Performance')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True)
    
    # Gráfico 5: Parâmetros PID
    plt.subplot(2, 3, 5)
    params_names = ['Kp', 'Ki', 'Kd']
    orig_params = results['Original']['params']
    opt_params = results['Otimizado']['params']
    
    x = np.arange(len(params_names))
    plt.bar(x - width/2, orig_params, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_params, width, label='Otimizado', color='blue', alpha=0.7)
    plt.xlabel('Parâmetros')
    plt.ylabel('Valor')
    plt.title('Parâmetros PID')
    plt.xticks(x, params_names)
    plt.legend()
    plt.grid(True)
    
    # Gráfico 6: Resumo textual
    plt.subplot(2, 3, 6)
    improvement_rmse = ((results['Original']['rmse'] - results['Otimizado']['rmse']) / 
                       results['Original']['rmse'] * 100)
    improvement_max = ((results['Original']['max_error'] - results['Otimizado']['max_error']) / 
                      results['Original']['max_error'] * 100)
    
    summary_text = f"""RESUMO DA OTIMIZAÇÃO

Parâmetros Originais:
Kp = {orig_params[0]:.3f}
Ki = {orig_params[1]:.3f}  
Kd = {orig_params[2]:.3f}

Parâmetros Otimizados:
Kp = {opt_params[0]:.3f}
Ki = {opt_params[1]:.3f}
Kd = {opt_params[2]:.3f}

MELHORIAS:
RMSE: {improvement_rmse:+.1f}%
Erro Máx: {improvement_max:+.1f}%

Algoritmo: PSO
Partículas: 20
Iterações: 30"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('pso_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    # Executar otimização
    original_params, optimized_params, pso_history = run_optimization()
    
    # Comparar resultados
    results = compare_results(original_params, optimized_params)
    
    print("\n" + "="*60)
    print("OTIMIZAÇÃO CONCLUÍDA!")
    print("="*60)
    print(f"\nArquivos gerados:")
    print("- pso_optimization_results.png (gráficos de comparação)")
    
    print(f"\nPara usar os parâmetros otimizados no seu código:")
    print(f"kp = {optimized_params[0]:.4f}")
    print(f"ki = {optimized_params[1]:.4f}")
    print(f"kd = {optimized_params[2]:.4f}")