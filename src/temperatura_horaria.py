"""
Temperatura peninsular HORARIA 2015-2025 ponderada por poblacion (Open-Meteo Archive).

Para cada ciudad de CIUDADES descarga temperature_2m horaria (ERA5) y agrega
con pesos por poblacion -> temp_peninsular_C horaria.
HDD18/CDD22 horarios: max(0, 18 - T) y max(0, T - 22) por hora.

Salida: data/raw/temperatura_peninsular_horaria.csv
        columnas: timestamp, temp_media_C, HDD18, CDD22

Descarga en bloques de 2 anos para no agotar timeouts.
"""

from pathlib import Path
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "data" / "raw" / "temperatura_peninsular_horaria.csv"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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

ANO_INI = 2015
ANO_FIN = 2025
CHUNK_ANOS = 2


def _bloques(ini: int, fin: int, chunk: int):
    cursor = ini
    while cursor <= fin:
        cierre = min(cursor + chunk - 1, fin)
        yield (f"{cursor}-01-01", f"{cierre}-12-31")
        cursor = cierre + 1


def fetch_temp_horaria(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     "temperature_2m",
        "timezone":   "Europe/Madrid",
    }
    for intento in range(1, 5):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
            resp.raise_for_status()
            js = resp.json()
            hourly = js.get("hourly", {})
            return pd.DataFrame({
                "timestamp": pd.to_datetime(hourly["time"]),
                "temp_C":    hourly["temperature_2m"],
            })
        except Exception as e:
            print(f"    [intento {intento}] error: {e}")
            time.sleep(2 * intento)
    raise RuntimeError(f"Open-Meteo no responde para {lat},{lon} {start}..{end}")


def temperatura_horaria_peninsular() -> pd.DataFrame:
    poblaciones = {c: info["pob"] for c, info in CIUDADES.items()}
    pob_total = sum(poblaciones.values())
    pesos = {c: p / pob_total for c, p in poblaciones.items()}

    series = {}
    for ciudad, info in CIUDADES.items():
        print(f"  {ciudad:<11} ({info['lat']:.2f}, {info['lon']:+.2f})  peso={pesos[ciudad]:.3f}")
        frames = []
        for b0, b1 in _bloques(ANO_INI, ANO_FIN, CHUNK_ANOS):
            print(f"    bloque {b0}..{b1}")
            frames.append(fetch_temp_horaria(info["lat"], info["lon"], b0, b1))
            time.sleep(0.4)
        df = (pd.concat(frames, ignore_index=True)
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True))
        series[ciudad] = df.set_index("timestamp")["temp_C"]

    matriz = pd.DataFrame(series)
    if matriz.isna().any().any():
        n = int(matriz.isna().sum().sum())
        print(f"  Aviso: {n} valores horarios nulos rellenados por interpolacion lineal.")
        matriz = matriz.interpolate("linear", limit_direction="both")

    pesos_arr = matriz.columns.map(pesos)
    matriz["temp_peninsular_C"] = (matriz * pesos_arr.values).sum(axis=1)
    return matriz.reset_index()


def construir_salida(horaria: pd.DataFrame) -> pd.DataFrame:
    s = horaria.set_index("timestamp")["temp_peninsular_C"]
    out = pd.DataFrame({
        "temp_media_C": s,
        "HDD18":        (BASE_HDD - s).clip(lower=0),
        "CDD22":        (s - BASE_CDD).clip(lower=0),
    }).round(3).reset_index()
    return out


if __name__ == "__main__":
    print(f"Descargando temperatura HORARIA de {len(CIUDADES)} ciudades (Open-Meteo Archive)...")
    horaria = temperatura_horaria_peninsular()
    out = construir_salida(horaria)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SALIDA, index=False)

    print(f"\nGuardado horario: {SALIDA}")
    print(f"  {len(out)} horas, de {out['timestamp'].min()} a {out['timestamp'].max()}")
    print(out.head(3).to_string(index=False))
    print("...")
    print(out.tail(3).to_string(index=False))
