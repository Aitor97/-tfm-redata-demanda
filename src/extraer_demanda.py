"""
Descarga la serie diaria de demanda electrica peninsular desde REData
y la guarda en CSV listo para el pipeline de modelado.

Salida: data/raw/demanda_peninsular_diaria.csv
Columnas: fecha (YYYY-MM-DD), demanda_MWh
"""

from pathlib import Path
import pandas as pd

from redata_demanda import descargar_peninsular


ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "data" / "raw" / "demanda_peninsular_diaria.csv"


def extraer(start: str = "2015-01-01T00:00", end: str = "2025-12-31T23:59") -> pd.DataFrame:
    series = descargar_peninsular(start_date=start, end_date=end, time_trunc="day")

    df = series.get("Demanda peninsular")
    if df is None or df.empty:
        raise RuntimeError("No se obtuvo la serie 'Demanda peninsular'.")

    # Quedarse solo con el indicador 'Demanda' (no porcentajes/variaciones).
    df = df[df["indicador"].str.lower() == "demanda"].copy()

    # Normalizar fecha al dia (en hora local Madrid, sin timezone).
    df["fecha"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"valor": "demanda_MWh"})
    df = df[["fecha", "demanda_MWh"]].sort_values("fecha").reset_index(drop=True)
    df = df.drop_duplicates("fecha")

    return df


if __name__ == "__main__":
    print("Descargando demanda diaria peninsular 2015-2025 desde REData...")
    df = extraer()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"\nGuardado: {SALIDA}")
    print(f"  {len(df)} dias, de {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))
