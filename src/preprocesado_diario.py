"""
Une demanda peninsular diaria + temperatura diaria + features de calendario.

Salida: data/processed/dataset_diario.csv
Columnas: fecha, demanda_MWh, temp_media_C, HDD18, CDD22,
          dow (0=lunes..6=domingo), is_weekend, is_holiday, mes, doy
"""

from pathlib import Path
import pandas as pd

try:
    import holidays as _holidays
except ImportError:
    raise SystemExit("Instala 'holidays': pip install holidays")


ROOT = Path(__file__).resolve().parents[1]
DEM  = ROOT / "data" / "processed" / "demanda_peninsular_diaria.csv"
TEMP = ROOT / "data" / "raw"       / "temperatura_peninsular_diaria.csv"
OUT  = ROOT / "data" / "processed" / "dataset_diario.csv"


def construir() -> pd.DataFrame:
    dem  = pd.read_csv(DEM,  parse_dates=["fecha"])
    temp = pd.read_csv(TEMP, parse_dates=["fecha"])

    df = dem.merge(temp, on="fecha", how="inner").sort_values("fecha").reset_index(drop=True)

    if df["demanda_MWh"].isna().any():
        raise ValueError("Hay valores nulos en demanda_MWh.")
    if df[["temp_media_C", "HDD18", "CDD22"]].isna().any().any():
        raise ValueError("Hay valores nulos en variables de temperatura.")

    fechas_esperadas = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
    if len(fechas_esperadas) != len(df):
        faltan = sorted(set(fechas_esperadas) - set(df["fecha"]))
        raise ValueError(f"Serie con huecos. Faltan {len(faltan)} dias: {faltan[:5]}...")

    anos = range(df["fecha"].dt.year.min(), df["fecha"].dt.year.max() + 1)
    festivos_es = _holidays.country_holidays("ES", years=list(anos))

    df["dow"]        = df["fecha"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["fecha"].dt.date.map(lambda d: 1 if d in festivos_es else 0)
    df["mes"]        = df["fecha"].dt.month
    df["doy"]        = df["fecha"].dt.dayofyear

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
