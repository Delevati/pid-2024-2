
# PID 2024-2 - Atividades de Controle Inteligente

Este repositório contém as implementações das atividades da disciplina de Controle Inteligente do semestre 2024-2.

## Tarefas

### Task 1 – Controladores Baseados em Conhecimento - Prof. Glauber

-**Status**: Projeto definido, faltam as regras de controle

-**Descrição**: Sistema de irrigação com pivô central usando controladores baseados em conhecimento

### Task 2 – Otimização de Controlador - Prof. Ícaro

-**Status**: Existre uma primeira versão do PSO concluída (pid-v3.py)

-**Análise necessária**

-**Descrição**: Otimização de parâmetros kpkikd usando algoritmos de otimização

## Implementação

### Task 1

Sistema de irrigação por pivô central com controladores baseados em conhecimento.

### Task 2

Comparação de três algoritmos de otimização para encontrar parâmetros ótimos de controlador PID (Kp, Ki, Kd):

- [X] **Enxame de Partículas (PSO)** - Existe uma primeira versão
- [ ] **Poliedros Flexíveis (Simplex)** - Aguardando
- [ ] **Algoritmos Genéticos** - Aguardando

Cada algoritmo deve ser testado nos seguintes cenários:

- Comportamento sem controle (malha aberta)
- Comportamento em malha fechada unitária
- Comportamento com PID otimizado

## Pendências

### Task 1

- [ ] Definir regras de controle para o pivô
- [ ] Modelar dinâmica do sistema

### Task 2

- [ ] **Validar convergência do PSO atual**
- [ ] **Critério de Goodhart**: Substituir métricas ITA/ITE atuais
- [ ] **Comparar performance dos três algoritmos**
- [ ] **Documentar resultados dos três comportamentos**
