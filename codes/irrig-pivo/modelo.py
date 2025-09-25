import numpy as np
import numpy as np
from input.perfil_terreno import get_declive

setores = [
    {"nome": "A", "umidade": 0.30, "capacidade": 0.45, "perda": 2.0/3600, "tipo": "Argiloso", "area_ha": 11.31},
    {"nome": "B", "umidade": 0.25, "capacidade": 0.40, "perda": 3.0/3600, "tipo": "Franco-argiloso", "area_ha": 11.31},
    {"nome": "C", "umidade": 0.35, "capacidade": 0.35, "perda": 4.0/3600, "tipo": "Franco", "area_ha": 11.31},
    {"nome": "D", "umidade": 0.20, "capacidade": 0.25, "perda": 6.0/3600, "tipo": "Arenoso", "area_ha": 11.31}
]   

obstaculos = [90, 210]

parametros = {
    "J": 1500.0,
    "torque_motor_max": 2000.0,
    "resistencia_base": 80.0,
    "dt": 1.0,
    "fator_aceleracao_tempo": 10.0,
    "comprimento_braco": 800.0,
    "num_aspersores": 20,
    "vazao_por_aspersor_max": 150.0,
    "pressao_bomba_max": 8.0,
    "perda_pressao_por_metro": 0.003
}

estado_inicial = {
    "ang_atual": 0.0,
    "vel_angular": 0.0,
    "aceleracao": 0.0,
    "consumo_agua_total": 0.0,
    "temperatura_ambiente": 25.0,
    "modo_emergencia": False,
    "pressao_atual": 5.0,
    "tempo_no_setor": 0.0,
    "tempo_simulacao_total": 0.0,
    "ang_anterior": 0.0
}

def ler_sensor_umidade(umidade_real):
    ruido = np.random.normal(0, 0.01)
    return max(0.0, min(1.0, umidade_real + ruido))

def ler_sensor_pressao(pressao_real):
    ruido = np.random.normal(0, 0.1)
    return max(0.0, pressao_real + ruido)

def controle_crisp(umidade_sensor, capacidade):
    x = umidade_sensor / capacidade
    return max(0.1, min(1.0, 1.2 - x*1.5))

def calcular_pressao_necessaria(fator_controle, pressao_bomba_max):
    return 3.0 + (pressao_bomba_max - 3.0) * fator_controle

def calcular_pressao_aspersores(distancia_centro, pressao_atual, perda_pressao_por_metro):
    perda_pressao = perda_pressao_por_metro * distancia_centro
    return max(2.0, pressao_atual - perda_pressao)

def calcular_vazao_aspersor(pressao_local, fator_controle, posicao, vazao_por_aspersor_max, comprimento_braco):
    if pressao_local < 2.0:
        return 0.0
    vazao_base = vazao_por_aspersor_max * np.sqrt(pressao_local / 6.0)
    fator_posicao = 0.8 + 0.2 * (posicao / comprimento_braco)
    return vazao_base * fator_controle * fator_posicao

def calcular_torque_resistivo(
    setor_idx, vel_angular, resistencia_base, dias_completos=0, ang_atual=0.0, umidade=0.3
):
    resistencia_solo = {0: 1.0, 1: 1.2, 2: 1.4, 3: 1.8}
    desgaste = 1.0 + 0.05 * dias_completos
    variacao = 1.0 + np.random.normal(0, 0.05)
    solo_encharcado = 1.0 + 0.5 * (1 if umidade > 0.9 else 0)
    resistencia = (
        resistencia_base
        * resistencia_solo[setor_idx]
        * desgaste
        * variacao
        * solo_encharcado
    )
    resistencia += 0.5 * abs(vel_angular)
    return resistencia


def atualizar_estado(estado, setores, obstaculos, parametros):
    J = parametros["J"]
    torque_motor_max = parametros["torque_motor_max"]
    resistencia_base = parametros["resistencia_base"]
    dt = parametros["dt"]
    fator_aceleracao_tempo = parametros["fator_aceleracao_tempo"]
    comprimento_braco = parametros["comprimento_braco"]
    num_aspersores = parametros["num_aspersores"]
    vazao_por_aspersor_max = parametros["vazao_por_aspersor_max"]
    pressao_bomba_max = parametros["pressao_bomba_max"]
    perda_pressao_por_metro = parametros["perda_pressao_por_metro"]
    
    ang_atual = estado["ang_atual"]
    vel_angular = estado["vel_angular"]
    aceleracao = estado["aceleracao"]
    consumo_agua_total = estado["consumo_agua_total"]
    temperatura_ambiente = estado["temperatura_ambiente"]
    modo_emergencia = estado["modo_emergencia"]
    pressao_atual = estado["pressao_atual"]
    tempo_no_setor = estado["tempo_no_setor"]
    tempo_simulacao_total = estado["tempo_simulacao_total"]

    NUM_MOTORES = 10
    posicoes_motores = np.linspace(0, comprimento_braco, NUM_MOTORES)
    estados_motores = []
    for i, pos in enumerate(posicoes_motores):
        declive_motor = get_declive(ang_atual, comprimento_braco=pos)
        if i == NUM_MOTORES - 1:
            ligado = True
        else:
            ligado = abs(declive_motor) > 8.0
        estados_motores.append({
            "pos": pos,
            "declive": declive_motor,
            "ligado": ligado
        })
    motores_ativos = sum(m["ligado"] for m in estados_motores)
    
    dt_real = dt * fator_aceleracao_tempo
    tempo_simulacao_total += dt_real / 60.0
    
    ang_atual = (ang_atual + vel_angular * dt_real / 60) % 360

    horas_totais = tempo_simulacao_total / 60.0
    dias_completos = int(horas_totais // 24)
    hora_do_dia = horas_totais % 24
    
    dias_do_ano = (dias_completos % 365)
    temp_sazonal = 5.0 * np.sin(2 * np.pi * (dias_do_ano - 172) / 365)
    temperatura_ambiente = 25.0 + temp_sazonal + 10.0 * np.sin(2 * np.pi * (horas_totais - 6) / 24)

    setor_idx = int((ang_atual % 360) // 90)
    setor_atual = setores[setor_idx]
    
    vento = 1.0 + 0.5 * np.sin(2 * np.pi * (horas_totais - 10) / 24)
    radiacao = 1.0 + 0.5 * np.sin(2 * np.pi * (horas_totais - 12) / 24)
    fator_temp = 1.0 + 0.03 * (temperatura_ambiente - 25.0)
    taxa_perda = setor_atual["perda"] * fator_temp * vento * radiacao
    setor_atual["umidade"] -= taxa_perda * setor_atual["umidade"] * dt_real / 3600
    
    # add - se tem chuva nao ligar pivo
    chuva = np.random.binomial(1, 0.01)
    if chuva:
        for setor in setores:
            setor["umidade"] = min(setor["capacidade"], setor["umidade"] + 0.01)

    umidade_sensor = ler_sensor_umidade(setor_atual["umidade"])
    fator_controle = controle_crisp(umidade_sensor, setor_atual["capacidade"])

    umidade_media = np.mean([s["umidade"] for s in setores])
    modo_emergencia = umidade_media < 0.20

    if modo_emergencia:
        pressao_desejada = pressao_bomba_max
    else:
        pressao_desejada = calcular_pressao_necessaria(fator_controle, pressao_bomba_max)

    erro_pressao = pressao_desejada - pressao_atual
    pressao_atual += erro_pressao * 0.1
    pressao_atual = max(3.0, min(pressao_bomba_max, pressao_atual))
    pressao_sensor = ler_sensor_pressao(pressao_atual)

    posicoes_aspersores = np.linspace(10, comprimento_braco, num_aspersores)
    vazao_total = 0.0
    vazoes_aspersores = []
    pressoes_aspersores = []

    for pos in posicoes_aspersores:
        pressao_local = calcular_pressao_aspersores(pos, pressao_atual, perda_pressao_por_metro)
        vazao_asp = calcular_vazao_aspersor(pressao_local, fator_controle, pos, vazao_por_aspersor_max, comprimento_braco)
        vazoes_aspersores.append(vazao_asp)
        pressoes_aspersores.append(pressao_local)
        vazao_total += vazao_asp

    if vazao_total > 0:
        lamina_aplicada_mm = (vazao_total * dt_real / 60.0) / (setor_atual["area_ha"] * 1000)
        incremento_umidade = lamina_aplicada_mm * 0.001
        setor_atual["umidade"] = min(setor_atual["capacidade"], setor_atual["umidade"] + incremento_umidade)
        consumo_agua_total += vazao_total * dt_real / 60.0

    declive_graus = get_declive(ang_atual, comprimento_braco)
    declive_fator = 1.0 + abs(declive_graus) / 30.0
    resistencia_total = calcular_torque_resistivo(
        setor_idx, vel_angular, resistencia_base, dias_completos, ang_atual, setor_atual["umidade"]
    )
    resistencia_total *= declive_fator
    resistencia_total += 0.3 * (vazao_total / 100)

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

    return {
        "ang_atual": ang_atual,
        "vel_angular": vel_angular,
        "aceleracao": aceleracao,
        "consumo_agua_total": consumo_agua_total,
        "temperatura_ambiente": temperatura_ambiente,
        "modo_emergencia": modo_emergencia,
        "pressao_atual": pressao_atual,
        "tempo_no_setor": tempo_no_setor,
        "tempo_simulacao_total": tempo_simulacao_total,
        "umidade_media": umidade_media,
        "dias_completos": dias_completos,
        "hora_do_dia": hora_do_dia,
        "pressao_desejada": pressao_desejada,
        "pressao_sensor": pressao_sensor,
        "setor_idx": setor_idx,
        "setor_atual": setor_atual,
        "posicoes_aspersores": posicoes_aspersores,
        "vazoes_aspersores": vazoes_aspersores,
        "pressoes_aspersores": pressoes_aspersores,
        "vazao_total": vazao_total,
        "torque_motor": torque_motor,
        "declive": declive_graus,
        "estados_motores": estados_motores,
        "motores_ativos": motores_ativos
    }