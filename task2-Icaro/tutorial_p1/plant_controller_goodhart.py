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

# =============================================================================
# CONFIGURAÇÃO DO SISTEMA DE CONTROLE
# =============================================================================

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

print("=== OTIMIZAÇÃO PID COM PSO ===")
print(f"Sistema: dx/dt = -{plant_a}*x + {plant_b}*u")
print(f"Referência: {reference_type}")
print(f"Tempo de simulação: {tf}s")
print()

# =============================================================================
# MODELOS DO SISTEMA (MESMO DO ARQUIVO ORIGINAL)
# =============================================================================

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

# =============================================================================
# FUNÇÃO OBJETIVO COM LEI DE GOODHART
# =============================================================================

def calculate_goodhart_metrics(pid_params):
    """
    Calcula múltiplas métricas independentes para evitar over-fitting
    Baseado na Lei de Goodhart: "Quando uma medida se torna um alvo, ela deixa de ser uma boa medida"
    
    Retorna: dicionário com todas as métricas calculadas
    """
    kp, ki, kd = pid_params
    
    # Verificar se os parâmetros estão dentro dos limites válidos
    if kp < 0 or ki < 0 or kd < 0:
        return None
    
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
        control_signals = np.zeros(n-1)
        
        # Simular sistema e coletar sinais de controle
        for i in range(n-1):
            out_states = odeint(connected_systems_model, states0, [0.0, tf/n],
                              args=(output_ref[i], output_dot_ref[i], pid_params))
            states0 = out_states[-1, :]
            states[i] = out_states[-1, :]
            
            # Calcular sinal de controle
            control_signals[i] = generic_controller(states[i, 0], output_ref[i], 
                                                  output_dot_ref[i], states[i, 1], kp, ki, kd)
        
        # =============================================================================
        # MÉTRICAS BASEADAS NA LEI DE GOODHART
        # =============================================================================
        
        error_signal = output_ref[:-1] - states[:, 0]
        
        # 1. MÉTRICAS DE PRECISÃO (Accuracy)
        rmse = np.sqrt(np.mean(error_signal**2))
        mae = np.mean(np.abs(error_signal))  # Mean Absolute Error
        max_error = np.max(np.abs(error_signal))
        
        # 2. MÉTRICAS DE ESTABILIDADE (Stability)
        # Variância do erro (menor = mais estável)
        error_variance = np.var(error_signal)
        
        # Variabilidade do sinal de controle (menor = mais suave)
        control_variance = np.var(control_signals)
        control_smoothness = np.mean(np.abs(np.diff(control_signals)))
        
        # 3. MÉTRICAS DE ROBUSTEZ (Robustness)
        # Pico do sinal de controle (menor = menos agressivo)
        max_control = np.max(np.abs(control_signals))
        
        # Esforço de controle total
        total_control_effort = np.sum(np.abs(control_signals))
        
        # 4. MÉTRICAS DE PERFORMANCE DINÂMICA (Dynamic Performance)
        overshoot = 0
        settling_time = float('inf')
        rise_time = float('inf')
        
        if reference_type == 'step':
            max_response = np.max(states[:, 0])
            if max_response > reference_amplitude:
                overshoot = ((max_response - reference_amplitude) / reference_amplitude) * 100
            
            # Settling time (2% criteria)
            tolerance = 0.02 * reference_amplitude
            settled_indices = np.where(np.abs(error_signal) <= tolerance)[0]
            if len(settled_indices) > 0:
                # Verificar se permanece dentro da tolerância
                for idx in settled_indices:
                    if np.all(np.abs(error_signal[idx:]) <= tolerance):
                        settling_time = time_vector[idx]
                        break
            
            # Rise time (10% to 90%)
            response_90 = 0.9 * reference_amplitude
            response_10 = 0.1 * reference_amplitude
            
            idx_10 = np.where(states[:, 0] >= response_10)[0]
            idx_90 = np.where(states[:, 0] >= response_90)[0]
            
            if len(idx_10) > 0 and len(idx_90) > 0:
                rise_time = time_vector[idx_90[0]] - time_vector[idx_10[0]]
        
        # 5. MÉTRICAS DE EFICIÊNCIA ENERGÉTICA (Energy Efficiency)
        # Integral do quadrado do sinal de controle (ISU - Integral of Squared Control)
        isu = np.sum(control_signals**2) * t_sim_step
        
        # Integral do erro absoluto (IAE)
        iae = np.sum(np.abs(error_signal)) * t_sim_step
        
        # Integral do erro quadrático (ISE)
        ise = np.sum(error_signal**2) * t_sim_step
        
        # Integral do erro absoluto ponderado pelo tempo (ITAE)
        itae = np.sum(time_vector[:-1] * np.abs(error_signal)) * t_sim_step
        
        # 6. MÉTRICAS DE CONSISTÊNCIA (Consistency)
        # Desvio padrão do erro na segunda metade da simulação (steady-state performance)
        mid_point = len(error_signal) // 2
        steady_state_std = np.std(error_signal[mid_point:])
        
        # Correlação entre erro e sinal de controle (menor = melhor desacoplamento)
        error_control_correlation = np.abs(np.corrcoef(error_signal, control_signals)[0, 1])
        
        return {
            # Métricas de Precisão
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            
            # Métricas de Estabilidade
            'error_variance': error_variance,
            'control_variance': control_variance,
            'control_smoothness': control_smoothness,
            
            # Métricas de Robustez
            'max_control': max_control,
            'total_control_effort': total_control_effort,
            
            # Métricas de Performance Dinâmica
            'overshoot': overshoot,
            'settling_time': settling_time,
            'rise_time': rise_time,
            
            # Métricas de Eficiência Energética
            'isu': isu,
            'iae': iae,
            'ise': ise,
            'itae': itae,
            
            # Métricas de Consistência
            'steady_state_std': steady_state_std,
            'error_control_correlation': error_control_correlation
        }
        
    except Exception as e:
        return None

def goodhart_cost_function(pid_params, weights=None):
    """
    Função de custo baseada na Lei de Goodhart
    
    Combina múltiplas métricas com pesos para evitar over-optimization de uma única métrica.
    Isso previne o problema da Lei de Goodhart onde otimizar uma métrica específica
    pode degradar a performance geral do sistema.
    
    Args:
        pid_params: [kp, ki, kd]
        weights: dict com pesos para cada métrica
    
    Returns:
        custo total (menor é melhor)
    """
    
    # Pesos padrão baseados na importância relativa (Lei de Goodhart)
    if weights is None:
        weights = {
            # Precisão (40% do peso total)
            'rmse': 0.25,
            'mae': 0.10,
            'max_error': 0.05,
            
            # Estabilidade (25% do peso total)
            'error_variance': 0.10,
            'control_variance': 0.08,
            'control_smoothness': 0.07,
            
            # Robustez (15% do peso total)
            'max_control': 0.08,
            'total_control_effort': 0.07,
            
            # Performance Dinâmica (10% do peso total)
            'overshoot': 0.05,
            'settling_time': 0.03,
            'rise_time': 0.02,
            
            # Eficiência Energética (5% do peso total)
            'isu': 0.02,
            'iae': 0.01,
            'ise': 0.01,
            'itae': 0.01,
            
            # Consistência (5% do peso total)
            'steady_state_std': 0.03,
            'error_control_correlation': 0.02
        }
    
    # Calcular métricas
    metrics = calculate_goodhart_metrics(pid_params)
    
    if metrics is None:
        return 1e6  # Penalidade para parâmetros inválidos
    
    # Normalizar métricas para evitar dominação por valores grandes
    normalized_metrics = {}
    
    # Fatores de normalização (valores típicos esperados)
    normalization_factors = {
        'rmse': 1.0,
        'mae': 1.0, 
        'max_error': 2.0,
        'error_variance': 0.1,
        'control_variance': 100.0,
        'control_smoothness': 10.0,
        'max_control': 50.0,
        'total_control_effort': 1000.0,
        'overshoot': 50.0,  # percentage
        'settling_time': 5.0,  # seconds
        'rise_time': 2.0,  # seconds
        'isu': 1000.0,
        'iae': 10.0,
        'ise': 5.0,
        'itae': 50.0,
        'steady_state_std': 0.1,
        'error_control_correlation': 1.0
    }
    
    # Normalizar e aplicar pesos
    total_cost = 0.0
    
    for metric_name, metric_value in metrics.items():
        if metric_name in weights:
            # Normalizar métrica
            norm_factor = normalization_factors.get(metric_name, 1.0)
            normalized_value = metric_value / norm_factor
            
            # Aplicar peso e somar ao custo total
            weighted_cost = weights[metric_name] * normalized_value
            total_cost += weighted_cost
    
    # Penalidades especiais (Lei de Goodhart: evitar soluções extremas)
    
    # Penalidade por parâmetros muito altos (instabilidade)
    kp, ki, kd = pid_params
    if kp > 50 or ki > 20 or kd > 10:
        total_cost += 0.5  # Penalidade por parâmetros extremos
    
    # Penalidade por settling time muito alto
    if metrics['settling_time'] > 2 * tf:
        total_cost += 1.0  # Sistema muito lento
    
    # Penalidade por overshoot excessivo
    if metrics['overshoot'] > 50:  # mais de 50%
        total_cost += 0.5
    
    # Penalidade por sinal de controle muito alto (saturação)
    if metrics['max_control'] > 100:
        total_cost += 0.3
    
    return total_cost

# Função wrapper para compatibilidade com código existente
def evaluate_pid(pid_params):
    """
    Função wrapper que usa a nova função de custo Goodhart
    Mantém compatibilidade com o código PSO existente
    """
    return goodhart_cost_function(pid_params)

# =============================================================================
# ALGORITMO PSO
# =============================================================================

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

# =============================================================================
# EXECUÇÃO DA OTIMIZAÇÃO
# =============================================================================

def analyze_goodhart_metrics(pid_params, name="Sistema"):
    """Analisa todas as métricas de Goodhart para um conjunto de parâmetros PID"""
    
    metrics = calculate_goodhart_metrics(pid_params)
    if metrics is None:
        print(f"{name}: Erro no cálculo das métricas")
        return None
    
    print(f"\n=== ANÁLISE GOODHART: {name.upper()} ===")
    
    # Métricas de Precisão
    print("📊 PRECISÃO:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  Erro Máximo: {metrics['max_error']:.6f}")
    
    # Métricas de Estabilidade
    print("\n🎯 ESTABILIDADE:")
    print(f"  Variância do Erro: {metrics['error_variance']:.6f}")
    print(f"  Variância do Controle: {metrics['control_variance']:.6f}")
    print(f"  Suavidade do Controle: {metrics['control_smoothness']:.6f}")
    
    # Métricas de Robustez
    print("\n🛡️ ROBUSTEZ:")
    print(f"  Pico de Controle: {metrics['max_control']:.6f}")
    print(f"  Esforço Total: {metrics['total_control_effort']:.6f}")
    
    # Métricas de Performance Dinâmica
    print("\n⚡ PERFORMANCE DINÂMICA:")
    print(f"  Overshoot: {metrics['overshoot']:.2f}%")
    if metrics['settling_time'] < float('inf'):
        print(f"  Tempo de Acomodação: {metrics['settling_time']:.3f}s")
    else:
        print(f"  Tempo de Acomodação: ∞ (não acomoda)")
    if metrics['rise_time'] < float('inf'):
        print(f"  Tempo de Subida: {metrics['rise_time']:.3f}s")
    
    # Métricas de Eficiência
    print("\n⚡ EFICIÊNCIA ENERGÉTICA:")
    print(f"  ISU: {metrics['isu']:.6f}")
    print(f"  IAE: {metrics['iae']:.6f}")
    print(f"  ISE: {metrics['ise']:.6f}")
    print(f"  ITAE: {metrics['itae']:.6f}")
    
    # Métricas de Consistência
    print("\n🎲 CONSISTÊNCIA:")
    print(f"  Desvio Padrão Steady-State: {metrics['steady_state_std']:.6f}")
    print(f"  Correlação Erro-Controle: {metrics['error_control_correlation']:.6f}")
    
    return metrics

def run_optimization():
    """Executar otimização PSO com análise Goodhart"""
    
    print("=== OTIMIZAÇÃO PID COM LEI DE GOODHART ===")
    print("Lei de Goodhart: 'Quando uma medida se torna um alvo, ela deixa de ser uma boa medida'")
    print("Por isso, usamos múltiplas métricas balanceadas para otimização robusta.\n")
    
    # Parâmetros originais para comparação
    original_params = [1.0, 0.0, 0.1]  # Do arquivo original
    original_cost = evaluate_pid(original_params)
    
    print("=== PARÂMETROS ORIGINAIS ===")
    print(f"Kp={original_params[0]:.3f}, Ki={original_params[1]:.3f}, Kd={original_params[2]:.3f}")
    print(f"Custo Goodhart original: {original_cost:.6f}")
    
    # Análise detalhada dos parâmetros originais
    original_metrics = analyze_goodhart_metrics(original_params, "Original")
    
    # Executar PSO
    print(f"\n{'='*60}")
    pso = PSO(num_particles=25, num_iterations=40)  # Mais iterações para Goodhart
    best_params, best_cost = pso.optimize()
    
    print("\n=== RESULTADOS DA OTIMIZAÇÃO GOODHART ===")
    print(f"Melhores parâmetros encontrados:")
    print(f"  Kp = {best_params[0]:.4f}")
    print(f"  Ki = {best_params[1]:.4f}")
    print(f"  Kd = {best_params[2]:.4f}")
    print(f"Custo Goodhart ótimo: {best_cost:.6f}")
    print(f"Melhoria total: {((original_cost - best_cost) / original_cost * 100):.1f}%")
    
    # Análise detalhada dos parâmetros otimizados
    optimized_metrics = analyze_goodhart_metrics(best_params, "Otimizado")
    
    return original_params, best_params, pso.cost_history, original_metrics, optimized_metrics

def create_goodhart_comparison_plots(original_results, optimized_results, time_vector, output_ref, pso_history):
    """Cria gráficos comparativos das métricas de Goodhart"""
    
    # Calcular métricas Goodhart para ambos os sistemas
    orig_metrics = calculate_goodhart_metrics(original_results['params'])
    opt_metrics = calculate_goodhart_metrics(optimized_results['params'])
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Resposta do sistema
    plt.subplot(4, 4, 1)
    plt.plot(time_vector[:-1], output_ref[:-1], 'k--', linewidth=3, label='Referência')
    plt.plot(time_vector[:-1], original_results['states'][:, 0], 'r', linewidth=2, label='Original')
    plt.plot(time_vector[:-1], optimized_results['states'][:, 0], 'b', linewidth=2, label='Goodhart Otimizado')
    plt.ylabel('Saída')
    plt.legend()
    plt.title('Resposta do Sistema')
    plt.grid(True)
    
    # 2. Erro de controle
    plt.subplot(4, 4, 2)
    error_orig = output_ref[:-1] - original_results['states'][:, 0]
    error_opt = output_ref[:-1] - optimized_results['states'][:, 0]
    plt.plot(time_vector[:-1], error_orig, 'r', linewidth=2, label='Original')
    plt.plot(time_vector[:-1], error_opt, 'b', linewidth=2, label='Goodhart Otimizado')
    plt.ylabel('Erro')
    plt.legend()
    plt.title('Erro de Controle')
    plt.grid(True)
    
    # 3. Convergência do PSO
    plt.subplot(4, 4, 3)
    plt.plot(pso_history, 'g', linewidth=2)
    plt.ylabel('Custo Goodhart')
    plt.xlabel('Iteração')
    plt.title('Convergência PSO (Lei de Goodhart)')
    plt.grid(True)
    
    # 4. Métricas de Precisão
    plt.subplot(4, 4, 4)
    precision_metrics = ['RMSE', 'MAE', 'Erro Máx']
    orig_vals = [orig_metrics['rmse'], orig_metrics['mae'], orig_metrics['max_error']]
    opt_vals = [opt_metrics['rmse'], opt_metrics['mae'], opt_metrics['max_error']]
    
    x = np.arange(len(precision_metrics))
    width = 0.35
    
    plt.bar(x - width/2, orig_vals, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_vals, width, label='Goodhart', color='blue', alpha=0.7)
    plt.ylabel('Valor')
    plt.title('📊 Métricas de Precisão')
    plt.xticks(x, precision_metrics, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 5. Métricas de Estabilidade
    plt.subplot(4, 4, 5)
    stability_metrics = ['Var. Erro', 'Var. Controle', 'Suavidade']
    orig_vals = [orig_metrics['error_variance'], orig_metrics['control_variance']/100, 
                 orig_metrics['control_smoothness']]
    opt_vals = [opt_metrics['error_variance'], opt_metrics['control_variance']/100, 
                opt_metrics['control_smoothness']]
    
    x = np.arange(len(stability_metrics))
    plt.bar(x - width/2, orig_vals, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_vals, width, label='Goodhart', color='blue', alpha=0.7)
    plt.ylabel('Valor (normalizado)')
    plt.title('🎯 Métricas de Estabilidade')
    plt.xticks(x, stability_metrics, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 6. Métricas de Robustez
    plt.subplot(4, 4, 6)
    robustness_metrics = ['Pico Controle', 'Esforço Total']
    orig_vals = [orig_metrics['max_control'], orig_metrics['total_control_effort']/100]
    opt_vals = [opt_metrics['max_control'], opt_metrics['total_control_effort']/100]
    
    x = np.arange(len(robustness_metrics))
    plt.bar(x - width/2, orig_vals, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_vals, width, label='Goodhart', color='blue', alpha=0.7)
    plt.ylabel('Valor')
    plt.title('🛡️ Métricas de Robustez')
    plt.xticks(x, robustness_metrics, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 7. Performance Dinâmica
    plt.subplot(4, 4, 7)
    dynamic_metrics = ['Overshoot (%)', 'Settling Time', 'Rise Time']
    
    # Tratar valores infinitos
    orig_settling = min(orig_metrics['settling_time'], 10)
    opt_settling = min(opt_metrics['settling_time'], 10)
    orig_rise = min(orig_metrics['rise_time'], 5)
    opt_rise = min(opt_metrics['rise_time'], 5)
    
    orig_vals = [orig_metrics['overshoot'], orig_settling, orig_rise]
    opt_vals = [opt_metrics['overshoot'], opt_settling, opt_rise]
    
    x = np.arange(len(dynamic_metrics))
    plt.bar(x - width/2, orig_vals, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_vals, width, label='Goodhart', color='blue', alpha=0.7)
    plt.ylabel('Valor')
    plt.title('⚡ Performance Dinâmica')
    plt.xticks(x, dynamic_metrics, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 8. Parâmetros PID
    plt.subplot(4, 4, 8)
    params_names = ['Kp', 'Ki', 'Kd']
    orig_params = original_results['params']
    opt_params = optimized_results['params']
    
    x = np.arange(len(params_names))
    plt.bar(x - width/2, orig_params, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_params, width, label='Goodhart', color='blue', alpha=0.7)
    plt.ylabel('Valor do Parâmetro')
    plt.title('Parâmetros PID')
    plt.xticks(x, params_names)
    plt.legend()
    plt.grid(True)
    
    # 9. Métricas de Eficiência Energética
    plt.subplot(4, 4, 9)
    efficiency_metrics = ['ISU/100', 'IAE', 'ISE', 'ITAE/10']
    orig_vals = [orig_metrics['isu']/100, orig_metrics['iae'], 
                 orig_metrics['ise'], orig_metrics['itae']/10]
    opt_vals = [opt_metrics['isu']/100, opt_metrics['iae'], 
                opt_metrics['ise'], opt_metrics['itae']/10]
    
    x = np.arange(len(efficiency_metrics))
    plt.bar(x - width/2, orig_vals, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_vals, width, label='Goodhart', color='blue', alpha=0.7)
    plt.ylabel('Valor (normalizado)')
    plt.title('⚡ Eficiência Energética')
    plt.xticks(x, efficiency_metrics, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 10. Métricas de Consistência
    plt.subplot(4, 4, 10)
    consistency_metrics = ['SS Std Dev', 'Erro-Ctrl Corr']
    orig_vals = [orig_metrics['steady_state_std'], orig_metrics['error_control_correlation']]
    opt_vals = [opt_metrics['steady_state_std'], opt_metrics['error_control_correlation']]
    
    x = np.arange(len(consistency_metrics))
    plt.bar(x - width/2, orig_vals, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, opt_vals, width, label='Goodhart', color='blue', alpha=0.7)
    plt.ylabel('Valor')
    plt.title('🎲 Métricas de Consistência')
    plt.xticks(x, consistency_metrics, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # 11. Radar Chart das métricas principais
    plt.subplot(4, 4, 11)
    categories = ['Precisão', 'Estabilidade', 'Robustez', 'Dinâmica', 'Eficiência']
    
    # Calcular scores normalizados (0-1, onde 1 é melhor)
    orig_scores = [
        1 / (1 + orig_metrics['rmse']),  # Precisão
        1 / (1 + orig_metrics['error_variance']),  # Estabilidade
        1 / (1 + orig_metrics['max_control']/50),  # Robustez
        1 / (1 + orig_metrics['overshoot']/100 + orig_settling/10),  # Dinâmica
        1 / (1 + orig_metrics['isu']/1000)  # Eficiência
    ]
    
    opt_scores = [
        1 / (1 + opt_metrics['rmse']),
        1 / (1 + opt_metrics['error_variance']),
        1 / (1 + opt_metrics['max_control']/50),
        1 / (1 + opt_metrics['overshoot']/100 + opt_settling/10),
        1 / (1 + opt_metrics['isu']/1000)
    ]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Fechar o círculo
    
    orig_scores.append(orig_scores[0])
    opt_scores.append(opt_scores[0])
    
    ax = plt.subplot(4, 4, 11, projection='polar')
    ax.plot(angles, orig_scores, 'r-', linewidth=2, label='Original')
    ax.fill(angles, orig_scores, 'red', alpha=0.25)
    ax.plot(angles, opt_scores, 'b-', linewidth=2, label='Goodhart')
    ax.fill(angles, opt_scores, 'blue', alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Radar Goodhart Score', pad=20)
    ax.legend()
    
    # 12. Zoom na resposta inicial
    plt.subplot(4, 4, 12)
    zoom_time = 1.0
    zoom_indices = time_vector[:-1] <= zoom_time
    
    plt.plot(time_vector[:-1][zoom_indices], output_ref[:-1][zoom_indices], 'k--', 
             linewidth=3, label='Referência')
    plt.plot(time_vector[:-1][zoom_indices], original_results['states'][:, 0][zoom_indices], 'r', 
             linewidth=2, label='Original')
    plt.plot(time_vector[:-1][zoom_indices], optimized_results['states'][:, 0][zoom_indices], 'b', 
             linewidth=2, label='Goodhart')
    plt.ylabel('Saída')
    plt.xlabel('Tempo [s]')
    plt.legend()
    plt.title('Zoom: Resposta Inicial (0-1s)')
    plt.grid(True)
    
    # 13-16. Resumo das melhorias
    for i, (subplot_idx, title, text_content) in enumerate([
        (13, "Lei de Goodhart", f"""LEI DE GOODHART APLICADA

"Quando uma medida se torna um alvo,
ela deixa de ser uma boa medida"

SOLUÇÃO IMPLEMENTADA:
✓ 17 métricas independentes
✓ Pesos balanceados
✓ Prevenção de over-fitting
✓ Otimização robusta

CATEGORIAS AVALIADAS:
• Precisão (40%)
• Estabilidade (25%) 
• Robustez (15%)
• Performance Dinâmica (10%)
• Eficiência Energética (5%)
• Consistência (5%)"""),
        
        (14, "Melhorias Obtidas", f"""MELHORIAS GOODHART

RMSE: {((orig_metrics['rmse'] - opt_metrics['rmse'])/orig_metrics['rmse']*100):+.1f}%
MAE: {((orig_metrics['mae'] - opt_metrics['mae'])/orig_metrics['mae']*100):+.1f}%
Erro Máx: {((orig_metrics['max_error'] - opt_metrics['max_error'])/orig_metrics['max_error']*100):+.1f}%

Overshoot: {(orig_metrics['overshoot'] - opt_metrics['overshoot']):+.1f} p.p.
Var. Erro: {((orig_metrics['error_variance'] - opt_metrics['error_variance'])/orig_metrics['error_variance']*100):+.1f}%
Esforço Ctrl: {((orig_metrics['total_control_effort'] - opt_metrics['total_control_effort'])/orig_metrics['total_control_effort']*100):+.1f}%

RESULTADO:
Sistema mais robusto e balanceado!"""),
        
        (15, "Parâmetros Finais", f"""PARAMETROS OTIMIZADOS

ORIGINAIS:
Kp = {original_results['params'][0]:.3f}
Ki = {original_results['params'][1]:.3f}
Kd = {original_results['params'][2]:.3f}

GOODHART OTIMIZADOS:
Kp = {optimized_results['params'][0]:.3f}
Ki = {optimized_results['params'][1]:.3f}
Kd = {optimized_results['params'][2]:.3f}

PARA USAR NO SEU CÓDIGO:
kp = {optimized_results['params'][0]:.4f}
ki = {optimized_results['params'][1]:.4f}
kd = {optimized_results['params'][2]:.4f}"""),
        
        (16, "Algoritmo PSO", f"""ALGORITMO PSO + GOODHART

CONFIGURAÇÃO:
• {len(pso_history)} iterações
• 25 partículas
• Busca global inteligente

FUNÇÃO DE CUSTO:
• Multi-objetivo balanceada
• 17 métricas independentes
• Penalidades por extremos
• Normalização automática

VANTAGENS:
✓ Evita over-fitting
✓ Soluções robustas
✓ Performance balanceada
✓ Aplicável a qualquer sistema""")
    ]):
        plt.subplot(4, 4, subplot_idx)
        plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        plt.title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('goodhart_pso_optimization_complete.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_results(original_params, optimized_params):
    """Comparar performance dos parâmetros originais vs otimizados com métricas Goodhart"""
    
    print("\n=== COMPARAÇÃO DETALHADA COM GOODHART ===")
    
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
    
    # Plotar comparação com métricas Goodhart
    create_goodhart_comparison_plots(results['Original'], results['Otimizado'], time_vector, output_ref, pso_history)
    
    return results

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Executar otimização com Lei de Goodhart
    original_params, optimized_params, pso_history, original_metrics, optimized_metrics = run_optimization()
    
    # Comparar resultados com análise Goodhart completa
    results = compare_results(original_params, optimized_params)
    
    print("\n" + "="*80)
    print("🎯 OTIMIZAÇÃO PID COM LEI DE GOODHART CONCLUÍDA!")
    print("="*80)
    
    # Resumo final das melhorias
    print("\n🏆 RESUMO DAS MELHORIAS GOODHART:")
    
    key_improvements = [
        ('RMSE', original_metrics['rmse'], optimized_metrics['rmse']),
        ('MAE', original_metrics['mae'], optimized_metrics['mae']),
        ('Erro Máximo', original_metrics['max_error'], optimized_metrics['max_error']),
        ('Overshoot (%)', original_metrics['overshoot'], optimized_metrics['overshoot']),
        ('Variância do Erro', original_metrics['error_variance'], optimized_metrics['error_variance']),
        ('Esforço de Controle', original_metrics['total_control_effort'], optimized_metrics['total_control_effort'])
    ]
    
    for metric_name, orig_val, opt_val in key_improvements:
        if orig_val > 0:
            improvement = ((orig_val - opt_val) / orig_val) * 100
            print(f"  {metric_name}: {improvement:+.1f}%")
        else:
            print(f"  {metric_name}: {orig_val:.6f} → {opt_val:.6f}")
    
    print(f"\n📁 ARQUIVOS GERADOS:")
    print("  - goodhart_pso_optimization_complete.png (análise visual completa)")
    print("  - Gráficos com 16 subplots incluindo radar chart")
    
    print(f"\n🔧 PARA USAR OS PARÂMETROS GOODHART-OTIMIZADOS:")
    print(f"# Cole no seu código motor_controller.py:")
    print(f"kp = {optimized_params[0]:.4f}  # Goodhart-otimizado")
    print(f"ki = {optimized_params[1]:.4f}  # Goodhart-otimizado")  
    print(f"kd = {optimized_params[2]:.4f}  # Goodhart-otimizado")
    
    print(f"\n💡 VANTAGENS DA OTIMIZAÇÃO GOODHART:")
    print("  ✓ Múltiplas métricas balanceadas (17 critérios)")
    print("  ✓ Evita over-fitting em uma única métrica")
    print("  ✓ Soluções mais robustas e estáveis")
    print("  ✓ Performance equilibrada em todos os aspectos")
    print("  ✓ Prevenção contra a Lei de Goodhart")
    
    print(f"\n🎯 'Quando uma medida se torna um alvo, ela deixa de ser uma boa medida'")
    print(f"   Por isso usamos 17 métricas independentes! 🧠")