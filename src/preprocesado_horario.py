"""
Une demanda peninsular horaria + temperatura horaria + features calendario.

Salida: data/processed/dataset_horario.csv
Columnas: timestamp, demanda_MWh, temp_media_C, HDD18, CDD22,
          hora (0..23), dow (0=lunes..6=domingo), is_weekend, is_holiday,
          mes, doy
"""

from pathlib import Path
import pandas as pd

try:
    import holidays as _holidays
except ImportError:
    raise SystemExit("Instala 'holidays': pip install holidays")


ROOT = Path(__file__).resolve().parents[1]
DEM  = ROOT / "data" / "processed" / "demanda_peninsular_horaria.csv"
TEMP = ROOT / "data" / "raw"       / "temperatura_peninsular_horaria.csv"
OUT  = ROOT / "data" / "processed" / "dataset_horario.csv"


def construir() -> pd.DataFrame:
    dem  = pd.read_csv(DEM,  parse_dates=["timestamp"])
    temp = pd.read_csv(TEMP, parse_dates=["timestamp"])

    # Truncar a hora exacta por si vienen distintos formatos de tz
    dem["timestamp"]  = dem["timestamp"].dt.floor("H")
    temp["timestamp"] = temp["timestamp"].dt.floor("H")

    df = dem.merge(temp, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)

    if df["demanda_MWh"].isna().any():
        raise ValueError("Hay valores nulos en demanda_MWh.")
    if df[["temp_media_C", "HDD18", "CDD22"]].isna().any().any():
        raise ValueError("Hay valores nulos en variables de temperatura.")

    horas_esperadas = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="H")
    huecos = len(horas_esperadas) - len(df)
    if huecos > 0:
        faltan = sorted(set(horas_esperadas) - set(df["timestamp"]))
        print(f"Aviso: faltan {len(faltan)} horas (ej: {faltan[:3]}). Se rellenan con interpolacion.")
        idx = pd.DatetimeIndex(horas_esperadas, name="timestamp")
        df = df.set_index("timestamp").reindex(idx)
        df["demanda_MWh"]  = df["demanda_MWh"].interpolate("linear", limit_direction="both")
        df["temp_media_C"] = df["temp_media_C"].interpolate("linear", limit_direction="both")
        df["HDD18"] = (18.0 - df["temp_media_C"]).clip(lower=0)
        df["CDD22"] = (df["temp_media_C"] - 22.0).clip(lower=0)
        df = df.reset_index()

    anos = range(df["timestamp"].dt.year.min(), df["timestamp"].dt.year.max() + 1)
    festivos_es = _holidays.country_holidays("ES", years=list(anos))

    df["hora"]       = df["timestamp"].dt.hour
    df["dow"]        = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["timestamp"].dt.date.map(lambda d: 1 if d in festivos_es else 0)
    df["mes"]        = df["timestamp"].dt.month
    df["doy"]        = df["timestamp"].dt.dayofyear

    return df


if __name__ == "__main__":
    df = construir()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Guardado: {OUT}")
    print(f"  {len(df)} horas, de {df['timestamp'].min()} a {df['timestamp'].max()}")
    print()
    print(df.describe().round(2).to_string())
    print()
    print("Primeras filas:")
    print(df.head(3).to_string(index=False))
    print("Ultimas filas:")
    print(df.tail(3).to_string(index=False))
