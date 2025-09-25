import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

def criar_figura(setores, obstaculos):
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(right=0.75)
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title("Simulação do Pivô de Irrigação com Dinâmica Física")
   
    setor_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    setor_labels = ['A', 'B', 'C', 'D']
    setor_wedges = []
    for i in range(4):
        wedge = Wedge((0,0), 1.2, i*90, (i+1)*90, facecolor=setor_colors[i], alpha=0.5, edgecolor='gray')
        ax.add_patch(wedge)
        setor_wedges.append(wedge)
        ang_label = np.deg2rad(i*90 + 45)
        ax.text(0.8*np.cos(ang_label), 0.8*np.sin(ang_label), setor_labels[i], fontsize=14, weight='bold', ha='center', va='center')

    for obs in obstaculos:
        ang = np.deg2rad(obs)
        ax.plot([1.1*np.cos(ang)], [1.1*np.sin(ang)], 'ro', markersize=10)

    pivo_arm, = ax.plot([], [], 'b-', linewidth=4)
    aspersores, = ax.plot([], [], 'ko', markersize=6)
    pivo_centro = Circle((0,0), 0.07, color='black')
    ax.add_patch(pivo_centro)
    water, = ax.plot([], [], 'co', markersize=2, alpha=0.7)

    info_text = fig.text(
        0.77, 0.97, '', fontsize=9, va='top', ha='left', family='monospace',
        bbox=dict(facecolor='whitesmoke', edgecolor='gray', boxstyle='round,pad=0.5')
    )

    elementos = {
        "setor_wedges": setor_wedges,
        "pivo_arm": pivo_arm,
        "aspersores": aspersores,
        "water": water,
        "info_text": info_text,
        "ax": ax
    }
    return fig, ax, elementos

def atualizar_visual(elementos, setores, estado, parametros):
    setor_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    pivo_arm = elementos["pivo_arm"]
    aspersores = elementos["aspersores"]
    water = elementos["water"]
    info_text = elementos["info_text"]
    setor_wedges = elementos["setor_wedges"]
    ax = elementos["ax"]

    comprimento_braco = parametros["comprimento_braco"]
    ang_atual = estado["ang_atual"]
    posicoes_aspersores = estado["posicoes_aspersores"]
    vazoes_aspersores = estado["vazoes_aspersores"]
    obstaculo_extra = estado["obstaculo_extra"]
    ang_rad = np.deg2rad(ang_atual)
    x_arm = [0, 1.2*np.cos(ang_rad)]
    y_arm = [0, 1.2*np.sin(ang_rad)]
    pivo_arm.set_data(x_arm, y_arm)

    aspersores_norm = posicoes_aspersores / comprimento_braco * 1.2
    x_aspersores = aspersores_norm * np.cos(ang_rad)
    y_aspersores = aspersores_norm * np.sin(ang_rad)
    aspersores.set_data(x_aspersores, y_aspersores)

    vazao_total = estado["vazao_total"]
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
    pressao_min = min(estado["pressoes_aspersores"])
    pressao_max = max(estado["pressoes_aspersores"])
    vazao_l_s = estado["vazao_total"] / 60.0
    consumo_setor = estado["vazao_total"] * estado["tempo_no_setor"]

    info_texto = (
        f"═══ TEMPO E DATA ═══\n"
        f"Dia:          {estado['dias_completos']:3d}\n"
        f"Hora:         {estado['hora_do_dia']:5.1f}h\n"
        f"Aceleração:   {parametros['fator_aceleracao_tempo']:3.1f}x\n"
        f"\n═══ ESTADO DO SISTEMA ═══\n"
        f"Posição:      {estado['ang_atual']:6.1f}°\n"
        f"Velocidade:   {estado['vel_angular']:6.3f}°/min\n"
        f"Tempo setor:  {estado['tempo_no_setor']:6.1f} min\n"
        f"Torque motor: {estado['torque_motor']:7.0f} N⋅m\n"
        f"Declive:       {estado['declive']:.2f}x\n"
        f"\n═══ SISTEMA DE IRRIGAÇÃO ═══\n"
        f"Pressão bomba:{estado['pressao_atual']:6.2f} bar\n"
        f"Pressão desej:{estado['pressao_desejada']:6.2f} bar\n"
        f"Press. mín/máx:{pressao_min:4.1f}/{pressao_max:4.1f}\n"
        f"Vazão total:  {estado['vazao_total']:6.0f} L/min\n"
        f"Vazão:        {vazao_l_s:6.1f} L/s\n"
        f"Aspersores:   {aspersores_ativos}/{parametros['num_aspersores']} ativos\n"
        f"Consumo setor:{consumo_setor:6.0f} L\n"
        f"\n═══ SETOR ATUAL: {estado['setor_atual']['nome']} ═══\n"
        f"Tipo solo:    {estado['setor_atual']['tipo']}\n"
        f"Área:         {estado['setor_atual']['area_ha']:5.1f} ha\n"
        f"Umidade real: {estado['setor_atual']['umidade']:6.1%}\n"
        f"Umidade sens: {ler_sensor_umidade(estado['setor_atual']['umidade']):6.1%}\n"
        f"Capacidade:   {estado['setor_atual']['capacidade']:6.1%}\n"
        f"\n═══ AMBIENTE ═══\n"
        f"Temperatura:  {estado['temperatura_ambiente']:5.1f}°C\n"
        f"Consumo total:{estado['consumo_agua_total']:8.0f} L\n"
        f"Umidade média:{estado['umidade_media']:6.1%}\n"
        f"{'EMERGÊNCIA' if estado['modo_emergencia'] else 'NORMAL'}\n"
        f"{'OBSTÁCULO' if obstaculo_extra > 0 else ''}"
    )

    estados_motores = estado.get("estados_motores", [])
    info_motores = "\n═══ MOTORES ═══\n"
    for i, m in enumerate(estados_motores):
        status = "ON " if m["ligado"] else "OFF"
        info_motores += f"Motor {i+1:2d}: {status} | Declive: {m['declive']:+5.1f}°\n"

    info_text.set_text(info_texto + info_motores)

    if hasattr(atualizar_visual, "motores_plot"):
        for mp in atualizar_visual.motores_plot:
            mp.remove()
    atualizar_visual.motores_plot = []
    for m in estados_motores:
        cor = 'red' if m["ligado"] else 'gray'
        x = (m["pos"] / comprimento_braco) * 1.2 * np.cos(ang_rad)
        y = (m["pos"] / comprimento_braco) * 1.2 * np.sin(ang_rad)
        mp = ax.plot(x, y, 'o', color=cor, markersize=6, alpha=0.8)[0]
        atualizar_visual.motores_plot.append(mp)

    return pivo_arm, aspersores, water, info_text

from modelo import ler_sensor_umidade