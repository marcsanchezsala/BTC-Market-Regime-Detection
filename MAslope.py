import pandas as pd
import numpy as np

def ma_slope(df, ma_type='SMA', length=20, slope_mode='Raw', col_name='slope'):
    """
    Calcula la pendiente (slope) de una Media Móvil y la añade al DataFrame.
    
    Parámetros:
    - df: DataFrame de pandas (debe contener una columna 'close').
    - ma_type: Tipo de media móvil ('SMA' o 'EMA').
    - length: Periodo de la media móvil (int).
    - slope_mode: 'Raw' (diferencia absoluta) o 'Normalized' (% de cambio).
    - col_name: Nombre de la nueva columna a crear.
    
    Retorna:
    - El df original con la nueva columna añadida.
    """
    
    # 1. Calcular la Media Móvil (MA) según el tipo
    if ma_type.upper() == 'SMA':
        ma = df['Close'].rolling(window=length).mean()
    elif ma_type.upper() == 'EMA':
        ma = df['Close'].ewm(span=length, adjust=False, min_periods=length).mean()
    else:
        raise ValueError("ma_type debe ser 'SMA' o 'EMA'")

    # 2. Calcular el Slope Crudo (Raw)
    # Equivalente a: ma - ma[1]
    slope_raw = ma.diff()

    # 3. Determinar el Slope final según el modo
    if slope_mode == 'Normalized':
        # Equivalente a: (slopeRaw / ma) * 100
        # Usamos fillna(0) para manejar posibles divisiones por 0 o NaNs iniciales
        slope_final = (slope_raw / ma) * 100.0
    else:
        slope_final = slope_raw

    df[col_name] = slope_final

    return df