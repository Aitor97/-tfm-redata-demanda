"""
Temperatura peninsular de Espana ponderada por poblacion (Open-Meteo Archive).

Para cada ciudad representativa se descarga la temperatura media diaria
(2015-2025) desde la ERA5 reanalysis via Open-Meteo Archive API (gratis,
sin API key). Se calcula una temperatura peninsular ponderada por
poblacion y, mensualmente:
    - temp_media_C  (media simple de los dias del mes)
    - HDD18         (sum de max(0, 18 - T_dia))
    - CDD22         (sum de max(0, T_dia - 22))

Las bases 18 / 22 son las habituales en literatura de demanda electrica
para clima mediterraneo continental.

Salida: data/raw/temperatura_peninsular_mensual.csv
"""

from pathlib import Path
import time
import requests
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "data" / "raw" / "temperatura_peninsular_mensual.csv"
SALIDA_DIARIA = ROOT / "data" / "raw" / "temperatura_peninsular_diaria.csv"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Ciudades peninsulares representativas (lat, lon, poblacion ~ INE 2024).
# Cobertura: Mediterraneo, Atlantico, meseta, sur.
CIUDADES: dict[str, dict] = {
    "Madrid":     {"lat": 40.4168, "lon": -3.7038, "pob": 3_340_000},
    "Barcelona":  {"lat": 41.3874, "lon":  2.1686, "pob": 1_660_000},
    "Valencia":   {"lat": 39.4699, "lon": -0.3763, "pob":   800_000},
    "Sevilla":    {"lat": 37.3886, "lon": -5.9823, "pob":   685_000},
    "Zaragoza":   {"lat": 41.6488, "lon": -0.8891, "pob":   680_000},
    "Malaga":     {"lat": 36.7213, "lon": -4.4214, "pob":   590_000},
    "Murcia":     {"lat": 37.9922, "lon": -1.1307, "pob":   465_000},
    "Bilbao":     {"lat": 43.2630, "lon": -2.9350, "pob":   345_000},
    "Valladolid": {"lat": 41.6523, "lon": -4.7245, "pob":   295_000},
    "Coruna":     {"lat": 43.3623, "lon": -8.4115, "pob":   250_000},
}

BASE_HDD = 18.0
BASE_CDD = 22.0


def fetch_temp_diaria(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily":      "temperature_2m_mean",
        "timezone":   "Europe/Madrid",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    js = resp.json()
    daily = js.get("daily", {})
    df = pd.DataFrame({
        "fecha": pd.to_datetime(daily["time"]),
        "temp_C": daily["temperature_2m_mean"],
    })
    return df


def temperatura_diaria_peninsular(start: str, end: str) -> pd.DataFrame:
    """Descarga T diaria por ciudad y devuelve serie ponderada por poblacion."""
    poblaciones = {c: info["pob"] for c, info in CIUDADES.items()}
    pob_total = sum(poblaciones.values())
    pesos = {c: p / pob_total for c, p in poblaciones.items()}

    series = {}
    for ciudad, info in CIUDADES.items():
        print(f"  {ciudad:<11} ({info['lat']:.2f}, {info['lon']:+.2f})  peso={pesos[ciudad]:.3f}")
        df = fetch_temp_diaria(info["lat"], info["lon"], start, end)
        series[ciudad] = df.set_index("fecha")["temp_C"]
        time.sleep(0.3)  # cortesia

    matriz = pd.DataFrame(series)  # index=fecha, cols=ciudad
    if matriz.isna().any().any():
        n = int(matriz.isna().sum().sum())
        print(f"  Aviso: {n} valores diarios nulos rellenados por interpolacion lineal.")
        matriz = matriz.interpolate("linear", limit_direction="both")

    pesos_arr = matriz.columns.map(pesos)
    matriz["temp_peninsular_C"] = (matriz * pesos_arr.values).sum(axis=1)
    return matriz.reset_index()


def agregar_mensual(diaria: pd.DataFrame) -> pd.DataFrame:
    s = diaria.set_index("fecha")["temp_peninsular_C"]
    hdd = (BASE_HDD - s).clip(lower=0)
    cdd = (s - BASE_CDD).clip(lower=0)

    mensual = pd.DataFrame({
        "temp_media_C": s,
        "HDD18":        hdd,
        "CDD22":        cdd,
    }).resample("MS").agg({
        "temp_media_C": "mean",
        "HDD18":        "sum",
        "CDD22":        "sum",
    })
    mensual = mensual.round(3).reset_index().rename(columns={"fecha": "fecha"})
    return mensual


def serie_diaria(diaria: pd.DataFrame) -> pd.DataFrame:
    """Devuelve T peninsular diaria con HDD/CDD a nivel diario."""
    s = diaria.set_index("fecha")["temp_peninsular_C"]
    out = pd.DataFrame({
        "temp_media_C": s,
        "HDD18":        (BASE_HDD - s).clip(lower=0),
        "CDD22":        (s - BASE_CDD).clip(lower=0),
    }).round(3).reset_index()
    return out


if __name__ == "__main__":
    print(f"Descargando temperatura diaria de {len(CIUDADES)} ciudades (Open-Meteo Archive)...")
    diaria = temperatura_diaria_peninsular("2015-01-01", "2025-12-31")
    mensual = agregar_mensual(diaria)
    diaria_pen = serie_diaria(diaria)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    mensual.to_csv(SALIDA, index=False)
    diaria_pen.to_csv(SALIDA_DIARIA, index=False)

    print(f"\nGuardado mensual: {SALIDA}")
    print(f"  {len(mensual)} meses, de {mensual['fecha'].min().date()} a {mensual['fecha'].max().date()}")
    print(mensual.head(3).to_string(index=False))
    print("...")
    print(mensual.tail(3).to_string(index=False))

    print(f"\nGuardado diario: {SALIDA_DIARIA}")
    print(f"  {len(diaria_pen)} dias, de {diaria_pen['fecha'].min().date()} a {diaria_pen['fecha'].max().date()}")
    print(diaria_pen.head(3).to_string(index=False))
    print("...")
    print(diaria_pen.tail(3).to_string(index=False))
