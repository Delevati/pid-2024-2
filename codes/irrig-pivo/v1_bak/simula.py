import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlim(-1.5, 2.0)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title("Simulação Realista do Pivô de Irrigação com Dinâmica Física")

setor_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
setor_labels = ['A', 'B', 'C', 'D']
setor_wedges = []

setores = [
    {"nome": "A", "umidade": 0.30, "capacidade": 0.45, "perda": 2.0/3600, "tipo": "Argiloso", "area_ha": 11.31},
    {"nome": "B", "umidade": 0.25, "capacidade": 0.40, "perda": 3.0/3600, "tipo": "Franco-argiloso", "area_ha": 11.31},
    {"nome": "C", "umidade": 0.35, "capacidade": 0.35, "perda": 4.0/3600, "tipo": "Franco", "area_ha": 11.31},
    {"nome": "D", "umidade": 0.20, "capacidade": 0.25, "perda": 6.0/3600, "tipo": "Arenoso", "area_ha": 11.31}
]

for i in range(4):
    wedge = Wedge((0,0), 1.2, i*90, (i+1)*90, facecolor=setor_colors[i], alpha=0.5, edgecolor='gray')
    ax.add_patch(wedge)
    setor_wedges.append(wedge)
    ang_label = np.deg2rad(i*90 + 45)
    ax.text(0.8*np.cos(ang_label), 0.8*np.sin(ang_label), setor_labels[i], fontsize=14, weight='bold', ha='center', va='center')

obstaculos = [90, 210]
for obs in obstaculos:
    ang = np.deg2rad(obs)
    ax.plot([1.1*np.cos(ang)], [1.1*np.sin(ang)], 'ro', markersize=10)

pivo_arm, = ax.plot([], [], 'b-', linewidth=4)
aspersores, = ax.plot([], [], 'ko', markersize=6)
pivo_centro = Circle((0,0), 0.07, color='black')
ax.add_patch(pivo_centro)
water, = ax.plot([], [], 'co', markersize=2, alpha=0.7)

info_box = plt.Rectangle((1.3, -1.4), 1.8, 2.8, fc='whitesmoke', ec='gray', lw=1.5)
ax.add_patch(info_box)
info_text = ax.text(1.35, 1.3, '', fontsize=9, va='top', ha='left', family='monospace')

J = 1500.0
torque_motor_max = 2000.0
resistencia_base = 80.0
dt = 1.0

# ACELERADOR DE TEMPO - EDITÁVEL AQUI
fator_aceleracao_tempo = 10.0  # Altere este valor: 1.0=normal, 2.0=2x mais rápido, 5.0=5x mais rápido

comprimento_braco = 120.0
num_aspersores = 20
vazao_por_aspersor_max = 150.0
pressao_bomba_max = 8.0
perda_pressao_por_metro = 0.003

ang_atual = 0.0
vel_angular = 0.0
aceleracao = 0.0
consumo_agua_total = 0.0
temperatura_ambiente = 25.0
modo_emergencia = False
pressao_atual = 5.0
tempo_no_setor = 0.0
tempo_simulacao_total = 0.0

def ler_sensor_umidade(umidade_real):
    ruido = np.random.normal(0, 0.01)
    return max(0.0, min(1.0, umidade_real + ruido))

def ler_sensor_pressao(pressao_real):
    ruido = np.random.normal(0, 0.1)
    return max(0.0, pressao_real + ruido)

def controle_crisp(umidade_sensor, capacidade):
    if umidade_sensor < capacidade * 0.3:
        return 1.0
    elif umidade_sensor < capacidade * 0.5:
        return 0.7
    elif umidade_sensor < capacidade * 0.7:
        return 0.4
    else:
        return 0.1

def calcular_pressao_necessaria(fator_controle):
    return 3.0 + (pressao_bomba_max - 3.0) * fator_controle

def calcular_pressao_aspersores(distancia_centro):
    perda_pressao = perda_pressao_por_metro * distancia_centro
    return max(2.0, pressao_atual - perda_pressao)

def calcular_vazao_aspersor(pressao_local, fator_controle, posicao):
    if pressao_local < 2.0:
        return 0.0
    vazao_base = vazao_por_aspersor_max * np.sqrt(pressao_local / 6.0)
    fator_posicao = 0.8 + 0.2 * (posicao / comprimento_braco)
    return vazao_base * fator_controle * fator_posicao

def calcular_torque_resistivo(setor_idx, vel_angular):
    resistencia_solo = {0: 1.0, 1: 1.2, 2: 1.4, 3: 1.8}
    resistencia = resistencia_base * resistencia_solo[setor_idx]
    resistencia += 0.5 * abs(vel_angular)
    return resistencia

def animate(frame):
    global ang_atual, vel_angular, aceleracao, consumo_agua_total, modo_emergencia
    global temperatura_ambiente, pressao_atual, tempo_no_setor, tempo_simulacao_total
    global fator_aceleracao_tempo
    
    # APLICAÇÃO DO ACELERADOR DE TEMPO
    dt_real = dt * fator_aceleracao_tempo
    tempo_simulacao_total += dt_real / 60.0  # em minutos
    
    # CÁLCULO DE HORA E DIA
    horas_totais = tempo_simulacao_total / 60.0
    dias_completos = int(horas_totais // 24)
    hora_do_dia = horas_totais % 24
    
    temperatura_ambiente = 25.0 + 10.0 * np.sin(2 * np.pi * horas_totais / 24)
    
    setor_idx = int((ang_atual % 360) // 90)
    setor_atual = setores[setor_idx]
    
    fator_temp = 1.0 + 0.03 * (temperatura_ambiente - 25.0)
    taxa_perda = setor_atual["perda"] * fator_temp
    setor_atual["umidade"] -= taxa_perda * setor_atual["umidade"] * dt_real / 3600
    
    umidade_sensor = ler_sensor_umidade(setor_atual["umidade"])
    fator_controle = controle_crisp(umidade_sensor, setor_atual["capacidade"])
    
    umidade_media = np.mean([s["umidade"] for s in setores])
    modo_emergencia = umidade_media < 0.20
    
    if modo_emergencia:
        pressao_desejada = pressao_bomba_max
    else:
        pressao_desejada = calcular_pressao_necessaria(fator_controle)
    
    erro_pressao = pressao_desejada - pressao_atual
    pressao_atual += erro_pressao * 0.1
    pressao_atual = max(3.0, min(pressao_bomba_max, pressao_atual))
    pressao_sensor = ler_sensor_pressao(pressao_atual)
    
    posicoes_aspersores = np.linspace(10, comprimento_braco, num_aspersores)
    vazao_total = 0.0
    vazoes_aspersores = []
    pressoes_aspersores = []
    
    for pos in posicoes_aspersores:
        pressao_local = calcular_pressao_aspersores(pos)
        vazao_asp = calcular_vazao_aspersor(pressao_local, fator_controle, pos)
        vazoes_aspersores.append(vazao_asp)
        pressoes_aspersores.append(pressao_local)
        vazao_total += vazao_asp
    
    if vazao_total > 0:
        lamina_aplicada_mm = (vazao_total * dt_real / 60.0) / (setor_atual["area_ha"] * 1000)
        incremento_umidade = lamina_aplicada_mm * 0.001
        setor_atual["umidade"] = min(setor_atual["capacidade"], 
                                   setor_atual["umidade"] + incremento_umidade)
        consumo_agua_total += vazao_total * dt_real / 60.0
    
    resistencia_total = calcular_torque_resistivo(setor_idx, vel_angular)
    resistencia_total += 0.3 * (vazao_total / 100)
    
    obstaculo_extra = 0
    for obs in obstaculos:
        if abs((ang_atual % 360) - obs) < 10:
            obstaculo_extra = 400
    resistencia_total += obstaculo_extra
    
    vel_desejada = 0.2 if modo_emergencia else 0.6
    erro_vel = vel_desejada - vel_angular
    torque_motor = min(torque_motor_max, 200.0 * erro_vel + resistencia_total)
    
    aceleracao = (torque_motor - resistencia_total) / J
    vel_angular += aceleracao * dt_real / 60
    vel_angular = max(0.05, min(2.0, vel_angular))
    ang_atual = (ang_atual + vel_angular * dt_real / 60) % 360
    
    tempo_no_setor += dt_real / 60
    if int(ang_atual / 90) != setor_idx:
        tempo_no_setor = 0.0
    
    ang_rad = np.deg2rad(ang_atual)
    x_arm = [0, 1.2*np.cos(ang_rad)]
    y_arm = [0, 1.2*np.sin(ang_rad)]
    pivo_arm.set_data(x_arm, y_arm)
    
    aspersores_norm = posicoes_aspersores / comprimento_braco * 1.2
    x_aspersores = aspersores_norm * np.cos(ang_rad)
    y_aspersores = aspersores_norm * np.sin(ang_rad)
    aspersores.set_data(x_aspersores, y_aspersores)
    
    if vazao_total > 10:
        x_water = []
        y_water = []
        for i, pos_norm in enumerate(aspersores_norm):
            vazao_asp = vazoes_aspersores[i]
            if vazao_asp > 5.0:
                x_asp = pos_norm * np.cos(ang_rad)
                y_asp = pos_norm * np.sin(ang_rad)
                raio_aspersao = min(0.06, vazao_asp / 500)
                n_drops = max(3, int(vazao_asp / 15))
                for j in range(n_drops):
                    angle_drop = np.random.uniform(0, 2*np.pi)
                    radius_drop = np.random.uniform(0, raio_aspersao)
                    x_drop = x_asp + radius_drop * np.cos(angle_drop)
                    y_drop = y_asp + radius_drop * np.sin(angle_drop)
                    x_water.append(x_drop)
                    y_water.append(y_drop)
        water.set_data(x_water, y_water)
    else:
        water.set_data([], [])
    
    for i, setor in enumerate(setores):
        if setor["umidade"] < setor["capacidade"] * 0.3:
            cor = 'red'
            alpha = 0.8
        elif setor["umidade"] < setor["capacidade"] * 0.5:
            cor = 'orange' 
            alpha = 0.6
        else:
            cor = setor_colors[i]
            alpha = 0.5
        setor_wedges[i].set_facecolor(cor)
        setor_wedges[i].set_alpha(alpha)
    
    aspersores_ativos = sum(1 for v in vazoes_aspersores if v > 5.0)
    pressao_min = min(pressoes_aspersores)
    pressao_max = max(pressoes_aspersores)
    vazao_l_s = vazao_total / 60.0
    consumo_setor = vazao_total * tempo_no_setor
    
    info_text.set_text(
        f"═══ TEMPO E DATA ═══\n"
        f"Dia:          {dias_completos:3d}\n"
        f"Hora:         {hora_do_dia:5.1f}h\n"
        f"Aceleração:   {fator_aceleracao_tempo:3.1f}x\n"
        f"\n═══ ESTADO DO SISTEMA ═══\n"
        f"Posição:      {ang_atual:6.1f}°\n"
        f"Velocidade:   {vel_angular:6.3f}°/min\n"
        f"Tempo setor:  {tempo_no_setor:6.1f} min\n"
        f"Torque motor: {torque_motor:7.0f} N⋅m\n"
        f"\n═══ SISTEMA DE IRRIGAÇÃO ═══\n"
        f"Pressão bomba:{pressao_atual:6.2f} bar\n"
        f"Pressão desej:{pressao_desejada:6.2f} bar\n"
        f"Press. mín/máx:{pressao_min:4.1f}/{pressao_max:4.1f}\n"
        f"Vazão total:  {vazao_total:6.0f} L/min\n"
        f"Vazão:        {vazao_l_s:6.1f} L/s\n"
        f"Aspersores:   {aspersores_ativos}/{num_aspersores} ativos\n"
        f"Consumo setor:{consumo_setor:6.0f} L\n"
        f"\n═══ SETOR ATUAL: {setor_atual['nome']} ═══\n"
        f"Tipo solo:    {setor_atual['tipo']}\n"
        f"Área:         {setor_atual['area_ha']:5.1f} ha\n"
        f"Umidade real: {setor_atual['umidade']:6.1%}\n"
        f"Umidade sens: {umidade_sensor:6.1%}\n"
        f"Capacidade:   {setor_atual['capacidade']:6.1%}\n"
        f"\n═══ AMBIENTE ═══\n"
        f"Temperatura:  {temperatura_ambiente:5.1f}°C\n"
        f"Consumo total:{consumo_agua_total:8.0f} L\n"
        f"Umidade média:{umidade_media:6.1%}\n"
        f"{'🚨 EMERGÊNCIA' if modo_emergencia else '✅ NORMAL'}\n"
        f"{'⚠️  OBSTÁCULO' if obstaculo_extra > 0 else ''}"
    )
    
    return pivo_arm, aspersores, water, info_text

ani = FuncAnimation(fig, animate, frames=3600, interval=100, blit=True)
plt.tight_layout()
plt.show()