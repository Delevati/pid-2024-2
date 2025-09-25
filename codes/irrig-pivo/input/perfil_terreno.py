import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

perfil = pd.read_csv('input/pontos_elevacao.csv')
perfil = perfil.sort_values('angulo')
f_altitude = interp1d(perfil['angulo'], perfil['SAMPLE_1'], kind='linear', fill_value="extrapolate")

def get_altitude(ang_atual):
    return float(f_altitude(ang_atual % 360))

def get_declive(ang_atual, comprimento_braco=800):
    alt_perimetro = get_altitude(ang_atual)
    alt_centro = perfil['SAMPLE_1'].mean()
    delta_s = comprimento_braco
    if delta_s == 0:
        return 0
    return np.degrees(np.arctan((alt_perimetro - alt_centro) / delta_s))