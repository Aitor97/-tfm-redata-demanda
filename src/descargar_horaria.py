"""
Descarga la demanda peninsular HORARIA via Open Power System Data (OPSD).

OPSD reempaqueta los datos horarios de ENTSO-E Transparency en un CSV unico,
gratuito y sin token. Ultimo release: 2020-10-06 -> cobertura 2015-01-01 .. 2020-09-30.

Columna ES: 'ES_load_actual_entsoe_transparency' (MW, marca horaria UTC).

Salida: data/processed/demanda_peninsular_horaria.csv
        columnas: timestamp (Europe/Madrid, naive), demanda_MWh
"""

import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPSD_CSV = os.path.join(ROOT_DIR, "data", "raw", "opsd_time_series_60min.csv.csv")
SALIDA = os.path.join(ROOT_DIR, "data", "processed", "demanda_peninsular_horaria.csv")

COL = "ES_load_actual_entsoe_transparency"


def descargar() -> pd.DataFrame:
    print(f"[lectura] OPSD local -> {OPSD_CSV}", flush=True)
    cols = ["utc_timestamp", COL]
    df = pd.read_csv(OPSD_CSV, usecols=cols, parse_dates=["utc_timestamp"])
    print(f"  cargadas {len(df):,} filas", flush=True)

    df = df.dropna(subset=[COL])
    idx = df["utc_timestamp"]
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize("UTC")
    else:
        idx = idx.dt.tz_convert("UTC")
    s = (pd.Series(df[COL].values, index=idx)
            .tz_convert("Europe/Madrid")
            .tz_localize(None))
    s = s[s.index >= pd.Timestamp("2015-01-01")]

    out = pd.DataFrame({
        "timestamp":   s.index,
        "demanda_MWh": s.values.astype(float),  # OPSD entrega MW por hora = MWh
    })
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return out


if __name__ == "__main__":
    if os.path.exists(SALIDA):
        print(f"[cache] Ya existe {SALIDA}", flush=True)
        df = pd.read_csv(SALIDA, parse_dates=["timestamp"])
    else:
        df = descargar()
        os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
        df.to_csv(SALIDA, index=False)

    inicio, fin = df["timestamp"].min(), df["timestamp"].max()
    print(f"\n[total] {len(df):,} filas, rango {inicio} .. {fin}", flush=True)

    horas_esperadas = pd.date_range(inicio.floor("H"), fin.floor("H"), freq="H")
    huecos = len(horas_esperadas) - len(df)
    print(f"[validacion] horas esperadas {len(horas_esperadas):,}; huecos {huecos:,}", flush=True)

    print(f"Guardado: {SALIDA}")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))
