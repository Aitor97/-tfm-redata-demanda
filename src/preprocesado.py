"""
Une la serie diaria de demanda peninsular con la temperatura ponderada
y guarda el dataset definitivo para el pipeline de modelado.

Salida: data/processed/dataset_diario.csv
Columnas: fecha, demanda_MWh, temp_media_C, HDD18, CDD22
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEM  = ROOT / "data" / "raw" / "demanda_peninsular_diaria.csv"
TEMP = ROOT / "data" / "raw" / "temperatura_peninsular_diaria.csv"
OUT  = ROOT / "data" / "processed" / "dataset_diario.csv"


def construir() -> pd.DataFrame:
    dem  = pd.read_csv(DEM,  parse_dates=["fecha"])
    temp = pd.read_csv(TEMP, parse_dates=["fecha"])

    df = dem.merge(temp, on="fecha", how="inner").sort_values("fecha").reset_index(drop=True)

    # Validaciones basicas.
    if df["demanda_MWh"].isna().any():
        raise ValueError("Hay valores nulos en demanda_MWh.")
    if df[["temp_media_C", "HDD18", "CDD22"]].isna().any().any():
        raise ValueError("Hay valores nulos en variables de temperatura.")

    fechas_esperadas = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
    if len(fechas_esperadas) != len(df):
        faltan = sorted(set(fechas_esperadas) - set(df["fecha"]))
        raise ValueError(f"Serie con huecos. Faltan {len(faltan)} dias: {faltan[:5]}...")

    return df


if __name__ == "__main__":
    df = construir()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Guardado: {OUT}")
    print(f"  {len(df)} dias, de {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print()
    print(df.describe().round(2).to_string())
    print()
    print("Primeras filas:")
    print(df.head(3).to_string(index=False))
    print("Ultimas filas:")
    print(df.tail(3).to_string(index=False))
