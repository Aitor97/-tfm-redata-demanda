"""
Descarga la demanda peninsular diaria 2015-2025 desde REData.

Reutiliza las funciones de src/redata_demanda.py.
Salida: data/processed/demanda_peninsular_diaria.csv (fecha, demanda_MWh).
"""

import os
import sys
import time
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from redata_demanda import fetch_demanda, json_a_dataframe

SALIDA = os.path.join(ROOT_DIR, "data", "processed", "demanda_peninsular_diaria.csv")


def descargar(start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    print(f"[descarga] Demanda peninsular DIARIA {start_year}-{end_year}...")
    frames = []
    for ano in range(start_year, end_year + 1):
        print(f"  {ano} ...")
        data = fetch_demanda(
            indicador="evolucion",
            start_date=f"{ano}-01-01T00:00",
            end_date=f"{ano}-12-31T23:59",
            time_trunc="day",
            geo_limit="peninsular",
        )
        df = json_a_dataframe(data)
        if not df.empty:
            frames.append(df)
        time.sleep(0.3)
    if not frames:
        raise RuntimeError("Respuesta REE vacia")
    full = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(["timestamp"], keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True))
    out = pd.DataFrame({
        "fecha":       full["timestamp"].dt.tz_localize(None).dt.normalize(),
        "demanda_MWh": full["valor"].astype(float),
    })
    out = out.drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    return out


if __name__ == "__main__":
    if os.path.exists(SALIDA):
        print(f"[cache] Ya existe {SALIDA}")
        df = pd.read_csv(SALIDA, parse_dates=["fecha"])
    else:
        df = descargar(2015, 2025)
        os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
        df.to_csv(SALIDA, index=False)
    print(f"\nFilas: {len(df)}, rango {df['fecha'].min().date()} .. {df['fecha'].max().date()}")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))
