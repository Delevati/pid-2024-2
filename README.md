
# PID 2024-2 - Atividades de Controle Inteligente

## Task 1 – Simulação de Pivô Central com Controle Baseado em Conhecimento

**Status:**

- Simulação crisp funcionando, tem de melhorar, mas tá funcionando.
- Visualização em tempo real do pivô, setores, aspersores, motores e variáveis do sistema.

**O que é simulado:**

- **Movimento do pivô central** com braço de 120 m (ajustável).
- **Motores independentes** ao longo do braço (ligam/desligam conforme declive, ponta nunca desliga).
- **Declive real do terreno** (perfil interpolado a partir de dados reais).
- **Setores de solo** com diferentes tipos, áreas e capacidades de retenção.
- **Umidade do solo** dinâmica, com evaporação, chuva aleatória e irrigação.
- **Controle crisp** para pressão e vazão, baseado em sensores simulados.
  - Para pressão e vazão, baseado em sensores simulados.
  - Para ativação dos motores (ligados se declive local > 3°).
  - Para velocidade angular do pivô.
- **Temperatura ambiente** com sazonalidade e variação diária.
- **Visualização gráfica**: posição do pivô, setores, aspersores ativos, motores (ON/OFF), água aplicada, painel de variáveis em tempo real.

**Principais variáveis monitoradas:**

- Ângulo e velocidade do pivô
- Torque do motor principal
- Declive do terreno (radial)
- Estado de cada motor (ON/OFF, declive local)
- Umidade real e sensoriada dos setores
- Pressão e vazão do sistema
- Temperatura ambiente
- Consumo de água total e por setor

**Próximos passos:**

- Definir e implementar regras de controle detalhadas.
- Integrar controle fuzzy para comparação.
- Ajustar critérios de ativação dos motores intermediários.
- Explorar otimização de consumo energético.

---

## Task 2 – Otimização de Controlador (Prof. Ícaro)

- **Status:** Primeira versão do PSO concluída (`pid-v3.py`).
- **Descrição:** Otimização de parâmetros Kp, Ki, Kd usando algoritmos de otimização.
- **Próximos passos:**
  - Validar convergência do PSO.
  - Implementar Simplex e Algoritmos Genéticos.
  - Comparar desempenho dos três algoritmos.

---

## Estrutura dos principais arquivos da task 1

- `modelo.py` — Lógica da simulação física, controle crisp, dinâmica dos motores, umidade, pressão, etc.
- `visual.py` — Visualização gráfica e painel de variáveis em tempo real.
- `input/perfil_terreno.py` — Interpolação do perfil de terreno (altitude e declive radial).
- `simula-run.py` — Script principal para rodar a simulação.
