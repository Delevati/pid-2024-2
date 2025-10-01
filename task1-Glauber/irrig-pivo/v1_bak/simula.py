import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.animation import FuncAnimation

def gerar_sensores_distribuidos(raio_sensor, area_raio, num_tentativas=100):
    """Gera sensores distribuídos pela área circular sem sobreposição"""
    sensores = []
    raio_minimo = raio_sensor * 2  # Distância mínima entre centros
    
    # Calcular número aproximado de sensores que cabem na área
    area_total = np.pi * area_raio**2
    area_sensor = np.pi * raio_sensor**2
    num_sensores_estimado = int(area_total / (area_sensor * 3))  # Fator de empacotamento
    
    tentativas = 0
    max_tentativas_totais = num_tentativas * num_sensores_estimado
    
    while len(sensores) < num_sensores_estimado and tentativas < max_tentativas_totais:
        # Gerar posição aleatória
        r = np.sqrt(np.random.uniform(0, 1)) * (area_raio - raio_sensor)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Verificar sobreposição com sensores existentes
        sobrepoe = False
        for sx, sy, _, _ in sensores:
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            if dist < raio_minimo:
                sobrepoe = True
                break
        
        if not sobrepoe:
            # Umidade inicial randomizada (entre 0.25 e 0.35, diferença máxima de ~29%)
            umidade_base = 0.30
            variacao = np.random.uniform(-0.075, 0.075)  # ±25% de 0.30
            umidade_inicial = np.clip(umidade_base + variacao, 0.15, 0.45)
            sensores.append([x, y, umidade_inicial, raio_sensor])
        
        tentativas += 1
    
    return sensores

fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlim(-1.5, 2.0)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title("Simulação Realista do Pivô de Irrigação com Dinâmica Física")

# Gerar sensores de umidade distribuídos
RAIO_SENSOR = 3.0 / 120.0  # Normalizado para escala do gráfico (3m em 120m)
AREA_RAIO = 1.2  # Raio da área de irrigação no gráfico
sensores_umidade = gerar_sensores_distribuidos(RAIO_SENSOR, AREA_RAIO)
print(f"Sensores gerados: {len(sensores_umidade)}")

# Criar círculos visuais para os sensores
sensor_circles = []
for sx, sy, umidade, raio in sensores_umidade:
    circle = Circle((sx, sy), raio, facecolor='blue', alpha=0.3, edgecolor='navy', linewidth=0.5)
    ax.add_patch(circle)
    sensor_circles.append(circle)

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

def obter_umidade_sensores_proximos(x, y, raio_busca=0.15):
    """Retorna umidade média dos sensores próximos à posição (x,y)"""
    umidades = []
    for sx, sy, umidade, _ in sensores_umidade:
        dist = np.sqrt((x - sx)**2 + (y - sy)**2)
        if dist <= raio_busca:
            # Peso inversamente proporcional à distância
            peso = 1.0 / (dist + 0.01)
            umidades.append((umidade, peso))
    
    if not umidades:
        return 0.30  # Valor padrão se não houver sensores próximos
    
    soma_ponderada = sum(u * p for u, p in umidades)
    soma_pesos = sum(p for _, p in umidades)
    return soma_ponderada / soma_pesos

def calcular_vazao_aspersor(pressao_local, fator_controle, posicao, umidade_local):
    if pressao_local < 2.0:
        return 0.0
    
    # Ajustar vazão baseado na umidade local
    # Quanto menor a umidade, maior a vazão necessária
    if umidade_local < 0.20:
        fator_umidade = 1.5  # Aumentar vazão em área seca
    elif umidade_local < 0.30:
        fator_umidade = 1.2
    elif umidade_local > 0.40:
        fator_umidade = 0.6  # Reduzir vazão em área úmida
    else:
        fator_umidade = 1.0
    
    vazao_base = vazao_por_aspersor_max * np.sqrt(pressao_local / 6.0)
    fator_posicao = 0.8 + 0.2 * (posicao / comprimento_braco)
    return vazao_base * fator_controle * fator_posicao * fator_umidade

def verificar_sensores_sob_braco(angulo_atual, largura_braco=0.15):
    """
    Verifica a umidade dos sensores sob o braço do pivô.
    Retorna: (umidade_minima, num_sensores_baixos, num_sensores_criticos)
    """
    ang_rad = np.deg2rad(angulo_atual)
    sensores_sob_braco = []
    
    # Verificar cada sensor
    for sx, sy, umidade, raio in sensores_umidade:
        # Calcular distância do sensor em relação ao centro
        dist_centro = np.sqrt(sx**2 + sy**2)
        
        # Calcular ângulo do sensor em relação ao centro
        ang_sensor = np.arctan2(sy, sx)
        
        # Diferença angular entre braço e sensor
        diff_ang = abs(np.rad2deg(ang_sensor - ang_rad))
        # Normalizar para 0-180 graus
        diff_ang = min(diff_ang, 360 - diff_ang)
        
        # Verificar se sensor está próximo ao braço (dentro da largura angular)
        largura_angular = np.rad2deg(largura_braco / max(dist_centro, 0.1))
        
        if diff_ang <= largura_angular and dist_centro <= 1.2:
            sensores_sob_braco.append(umidade)
    
    if not sensores_sob_braco:
        return 0.40, 0, 0  # Sem sensores, pode avançar
    
    umidade_min = min(sensores_sob_braco)
    sensores_baixos = sum(1 for u in sensores_sob_braco if u < 0.40)
    sensores_criticos = sum(1 for u in sensores_sob_braco if u < 0.25)
    
    return umidade_min, sensores_baixos, sensores_criticos

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
    
    ang_rad = np.deg2rad(ang_atual)
    
    for pos in posicoes_aspersores:
        # Calcular posição do aspersor no plano
        pos_norm = pos / comprimento_braco * 1.2
        x_asp = pos_norm * np.cos(ang_rad)
        y_asp = pos_norm * np.sin(ang_rad)
        
        # Obter umidade dos sensores próximos
        umidade_local = obter_umidade_sensores_proximos(x_asp, y_asp)
        
        pressao_local = calcular_pressao_aspersores(pos)
        vazao_asp = calcular_vazao_aspersor(pressao_local, fator_controle, pos, umidade_local)
        vazoes_aspersores.append(vazao_asp)
        pressoes_aspersores.append(pressao_local)
        vazao_total += vazao_asp
        
        # Irrigar sensores próximos ao aspersor
        if vazao_asp > 5.0:
            for i, (sx, sy, umidade, raio) in enumerate(sensores_umidade):
                dist = np.sqrt((x_asp - sx)**2 + (y_asp - sy)**2)
                raio_irrigacao = 0.08  # Raio de alcance do aspersor
                if dist <= raio_irrigacao:
                    # Incremento proporcional à vazão e inversamente à distância
                    fator_dist = 1.0 - (dist / raio_irrigacao)
                    incremento = (vazao_asp / 10000.0) * fator_dist * dt_real
                    sensores_umidade[i][2] = min(0.50, umidade + incremento)
    
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
    
    # CONTROLE DE VELOCIDADE BASEADO EM SENSORES
    umidade_min_braco, sensores_abaixo_40, sensores_criticos_braco = verificar_sensores_sob_braco(ang_atual)
    
    # Definir velocidade desejada baseada na umidade dos sensores
    if sensores_abaixo_40 > 0:
        # Há sensores com umidade < 40%, reduzir velocidade proporcionalmente
        if umidade_min_braco < 0.25:
            vel_desejada = 0.05  # Quase parado em área crítica
        elif umidade_min_braco < 0.30:
            vel_desejada = 0.15  # Muito devagar
        elif umidade_min_braco < 0.35:
            vel_desejada = 0.30  # Devagar
        else:
            vel_desejada = 0.45  # Moderado
    elif modo_emergencia:
        vel_desejada = 0.2  # Modo emergência geral
    else:
        vel_desejada = 0.8  # Velocidade normal - área bem irrigada
    
    erro_vel = vel_desejada - vel_angular
    torque_motor = min(torque_motor_max, 200.0 * erro_vel + resistencia_total)
    
    aceleracao = (torque_motor - resistencia_total) / J
    vel_angular += aceleracao * dt_real / 60
    vel_angular = max(0.05, min(2.0, vel_angular))
    ang_atual = (ang_atual + vel_angular * dt_real / 60) % 360
    
    tempo_no_setor += dt_real / 60
    if int(ang_atual / 90) != setor_idx:
        tempo_no_setor = 0.0
    
    # Aplicar perda de umidade por evaporação nos sensores
    for i, (sx, sy, umidade, raio) in enumerate(sensores_umidade):
        # Taxa de perda varia com temperatura e umidade atual
        taxa_evaporacao = (0.0005 / 3600) * fator_temp * dt_real
        nova_umidade = max(0.10, umidade - taxa_evaporacao * umidade)
        sensores_umidade[i][2] = nova_umidade
        
        # Atualizar cor do círculo do sensor baseado na umidade
        if umidade < 0.20:
            cor = 'red'
            alpha = 0.6
        elif umidade < 0.30:
            cor = 'orange'
            alpha = 0.5
        elif umidade < 0.40:
            cor = 'yellow'
            alpha = 0.4
        else:
            cor = 'green'
            alpha = 0.4
        sensor_circles[i].set_facecolor(cor)
        sensor_circles[i].set_alpha(alpha)
    
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
    
    # Estatísticas dos sensores
    umidades_sensores = [s[2] for s in sensores_umidade]
    umidade_media_sensores = np.mean(umidades_sensores)
    umidade_min_sensores = min(umidades_sensores)
    umidade_max_sensores = max(umidades_sensores)
    sensores_criticos = sum(1 for u in umidades_sensores if u < 0.20)
    
    # Status do controle de velocidade
    status_velocidade = "✅ NORMAL"
    if sensores_abaixo_40 > 0:
        status_velocidade = f"⚠️  IRRIGANDO ({sensores_abaixo_40} sens.)"
    if sensores_criticos_braco > 0:
        status_velocidade = f"🚨 CRÍTICO ({sensores_criticos_braco} sens.)"
    
    info_text.set_text(
        f"═══ TEMPO E DATA ═══\n"
        f"Dia:          {dias_completos:3d}\n"
        f"Hora:         {hora_do_dia:5.1f}h\n"
        f"Aceleração:   {fator_aceleracao_tempo:3.1f}x\n"
        f"\n═══ ESTADO DO SISTEMA ═══\n"
        f"Posição:      {ang_atual:6.1f}°\n"
        f"Velocidade:   {vel_angular:6.3f}°/min\n"
        f"Vel. desejada:{vel_desejada:6.3f}°/min\n"
        f"Tempo setor:  {tempo_no_setor:6.1f} min\n"
        f"Torque motor: {torque_motor:7.0f} N⋅m\n"
        f"Status:       {status_velocidade}\n"
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
        f"\n═══ CONTROLE DE VELOCIDADE ═══\n"
        f"Umid. sob braço:{umidade_min_braco:6.1%}\n"
        f"Sens. < 40%:  {sensores_abaixo_40:3d}\n"
        f"Sens. críticos:{sensores_criticos_braco:3d}\n"
        f"\n═══ SENSORES DE SOLO ({len(sensores_umidade)}) ═══\n"
        f"Umidade média:{umidade_media_sensores:6.1%}\n"
        f"Umidade mín:  {umidade_min_sensores:6.1%}\n"
        f"Umidade máx:  {umidade_max_sensores:6.1%}\n"
        f"Críticos:     {sensores_criticos:3d} sensores\n"
        f"\n═══ AMBIENTE ═══\n"
        f"Temperatura:  {temperatura_ambiente:5.1f}°C\n"
        f"Consumo total:{consumo_agua_total:8.0f} L\n"
        f"Umidade setor:{umidade_media:6.1%}\n"
        f"{'🚨 EMERGÊNCIA' if modo_emergencia else '✅ NORMAL'}\n"
        f"{'⚠️  OBSTÁCULO' if obstaculo_extra > 0 else ''}"
    )
    
    return pivo_arm, aspersores, water, info_text

ani = FuncAnimation(fig, animate, frames=3600, interval=100, blit=True)
plt.tight_layout()
plt.show()