# Sistema de Controle Genérico

## Transformação Realizada

O código original `motor_controller.py` foi transformado de um **sistema específico de controle de motor DC** para um **sistema de controle genérico** reutilizável.

## Principais Mudanças Implementadas

### 1. **Modelo da Planta Genérico**
- **Antes**: `dc_motor_model()` - modelo específico para motor DC
- **Depois**: `generic_plant_model()` - modelo genérico de 1ª ordem: `dx/dt = -a*x + b*u`

### 2. **Controlador Genérico**
- **Antes**: `motor_controller()` - controlador específico para motor
- **Depois**: `generic_controller()` - controlador PID configurável com feedforward

### 3. **Parametrização Flexível**
```python
# Parâmetros da Planta (Sistema de 1ª ordem)
plant_a = 2.0      # Polo do sistema (velocidade de resposta)
plant_b = 2.0      # Ganho do sistema
plant_a_error = 0.1  # Incerteza no modelo
plant_b_error = 0.2  # Incerteza no ganho

# Parâmetros do Controlador (PID)
kp = 3.0  # Ganho proporcional
ki = 1.0  # Ganho integral  
kd = 0.5  # Ganho derivativo
```

### 4. **Sinais de Referência Configuráveis**
```python
reference_type = 'sine'  # Opções: 'sine', 'step', 'ramp', 'square'
reference_amplitude = 1.0
reference_frequency = 0.5
```

### 5. **Variáveis com Nomes Genéricos**
- **tau, dc_volts** → **system_output, control_input**
- **torque_ref** → **output_ref**
- Todas as variáveis agora têm nomes genéricos

### 6. **Melhorias Adicionais**
- **Métricas de desempenho**: RMSE, erro máximo, tempo de acomodação
- **Gráficos aprimorados**: 6 subplots com análises completas
- **Exportação de dados**: CSV com todos os sinais relevantes
- **Documentação completa**: Comentários explicativos

## Como Usar o Sistema Genérico

### 1. **Instalação de Dependências**
```bash
pip install numpy matplotlib scipy pandas
```

### 2. **Configuração Básica**
Edite os parâmetros no início do arquivo `generic_control_system.py`:

```python
# Configurar a planta
plant_a = 1.5      # Ajuste para diferentes velocidades de resposta
plant_b = 1.0      # Ajuste o ganho do sistema

# Configurar o controlador
kp = 2.0   # Aumentar para resposta mais rápida
ki = 0.5   # Aumentar para eliminar erro estacionário
kd = 0.1   # Aumentar para reduzir overshooting

# Configurar referência
reference_type = 'step'  # Teste resposta ao degrau
```

### 3. **Tipos de Sistemas que Pode Simular**
- **Sistemas térmicos**: Controle de temperatura
- **Sistemas mecânicos**: Controle de posição/velocidade
- **Sistemas elétricos**: Controle de tensão/corrente
- **Sistemas químicos**: Controle de concentração
- **Qualquer sistema de 1ª ordem**

### 4. **Exemplos de Configuração**

#### Sistema Rápido (Eletrônico)
```python
plant_a = 10.0  # Resposta muito rápida
plant_b = 5.0   # Alto ganho
kp = 1.0        # Ganho moderado
```

#### Sistema Lento (Térmico)
```python
plant_a = 0.1   # Resposta lenta
plant_b = 0.5   # Baixo ganho
kp = 5.0        # Alto ganho necessário
ki = 2.0        # Integral para erro estacionário
```

#### Sistema com Incertezas
```python
plant_a_error = 0.5  # 50% de incerteza
plant_b_error = 0.3  # 30% de incerteza
# O controlador compensa automaticamente
```

## Vantagens do Sistema Genérico

1. **Reutilizável**: Pode simular qualquer sistema de controle de 1ª ordem
2. **Configurável**: Parâmetros facilmente ajustáveis
3. **Educativo**: Ideal para aprender conceitos de controle
4. **Robusto**: Lida com incertezas do modelo
5. **Completo**: Análise de desempenho integrada
6. **Profissional**: Código bem documentado e estruturado

## Estrutura do Código

```
generic_control_system.py
├── Configuração de parâmetros
├── Definição de modelos
│   ├── generate_reference_signal()
│   ├── generic_plant_model()
│   ├── generic_controller()
│   └── connected_systems_model()
├── Setup da simulação
├── Loop principal de simulação
├── Análise de resultados
├── Exportação de dados
└── Plotagem de gráficos
```

## Próximos Passos Sugeridos

1. **Expandir para sistemas de ordem superior** (2ª ordem, 3ª ordem)
2. **Adicionar mais tipos de controladores** (LQR, MPC)
3. **Incluir perturbações e ruído**
4. **Implementar identificação de sistema**
5. **Adicionar interface gráfica**

O sistema agora é completamente genérico e pode ser usado como base para qualquer projeto de controle automático!