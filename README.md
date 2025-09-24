# PID 2024-2 - Atividades de Controle Inteligente

Este repositório contém as implementações das atividades da disciplina de Controle Inteligente do semestre 2024-2.

## Tarefas

### Task 1 – Controladores Baseados em Conhecimento - Prof. Glauber

-**Status**: Projeto definido, faltam as regras de controle

-**Descrição**: Sistema de irrigação com pivô central usando controladores baseados em conhecimento

### Task 2 – Otimização de Controlador - Prof. Ícaro

-**Status**: Primeira versão do PSO já existe, pid-v3.py.

-**Análise necessária**

-**Descrição**: Otimização de parâmetros PID usando algoritmos de otimização

## Implementação

### Task 1

Sistema de irrigação por pivô central com controladores baseados em conhecimento.

### Task 2

Comparação de três algoritmos de otimização para encontrar parâmetros ótimos de controlador PID (Kp, Ki, Kd):

- [X] **Enxame de Partículas (PSO)** - Primeira versão implementada
- [ ] **Poliedros Flexíveis (Simplex)** - Aguardando implementação dos colegas
- [ ] **Algoritmos Genéticos** - Aguardando implementação dos colegas

Cada algoritmo deve ser testado nos seguintes cenários:

- Comportamento sem controle (malha aberta)
- Comportamento em malha fechada unitária
- Comportamento com PID otimizado

## Pendências

### Task 1

- [ ] Desenvolver regras de controle baseadas em conhecimento
- [ ] Implementar sistema de irrigação

### Task 2

- [ ] **Análise da implementação PSO atual**
- [ ] **Critério de Goodhart**: Substituir métricas ITA/ITE atuais
- [ ] **Aguardar implementações**: Simplex e Algoritmos Genéticos dos colegas
- [ ] **Plotagem**: Gerar gráficos comparativos dos três comportamentos
