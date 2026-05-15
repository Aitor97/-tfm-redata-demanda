"""
Descarga la serie mensual de demanda electrica peninsular desde REData
y la guarda en CSV listo para el pipeline de modelado.

Salida: data/raw/demanda_peninsular_mensual.csv
Columnas: fecha (YYYY-MM-01), demanda_MWh
"""

from datetime import date
from pathlib import Path
import pandas as pd

from redata_demanda import descargar_peninsular


ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "data" / "raw" / "demanda_peninsular_mensual.csv"

_HOY = date.today()
_FIN_DEFAULT = f"{_HOY.year}-{_HOY.month:02d}-01T00:00"


def extraer(start: str = "2015-01-01T00:00", end: str = _FIN_DEFAULT) -> pd.DataFrame:
    series = descargar_peninsular(start_date=start, end_date=end, time_trunc="month")

    df = series.get("Demanda peninsular")
    if df is None or df.empty:
        raise RuntimeError("No se obtuvo la serie 'Demanda peninsular'.")

    # Quedarse solo con el indicador 'Demanda' (no porcentajes/variaciones).
    df = df[df["indicador"].str.lower() == "demanda"].copy()

    # Normalizar fecha al primer dia del mes (en hora local Madrid).
    df["fecha"] = df["timestamp"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    df = df.rename(columns={"valor": "demanda_MWh"})
    df = df[["fecha", "demanda_MWh"]].sort_values("fecha").reset_index(drop=True)
    df = df.drop_duplicates("fecha")

    return df


if __name__ == "__main__":
    print(f"Descargando demanda mensual peninsular 2015-{_HOY.year} desde REData...")
    df = extraer()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"\nGuardado: {SALIDA}")
    print(f"  {len(df)} meses, de {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))
