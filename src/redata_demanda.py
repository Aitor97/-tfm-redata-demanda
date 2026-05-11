"""
Extraccion de datos de demanda electrica desde la API REData (Red Electrica de Espana).
Documentacion: https://apidatos.ree.es/
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
import time

BASE_URL = "https://apidatos.ree.es/es/datos"

INDICADORES_DEMANDA = {
    "evolucion":            "demanda/evolucion",
    "ire_general":          "demanda/ire-general",
    "ire_industria":        "demanda/ire-industria",
    "ire_servicios":        "demanda/ire-servicios",
    "perdidas_transporte":  "demanda/perdidas-transporte",
    "demanda_tiempo_real":  "demanda/demanda-tiempo-real",
    "componentes":          "demanda/componentes",
}

# IDs de CCAA peninsulares segun REData
CCAA = {
    "Andalucia":                  4,
    "Aragon":                     5,
    "Cantabria":                  6,
    "Castilla-La Mancha":         7,
    "Castilla y Leon":            8,
    "Cataluna":                   9,
    "Pais Vasco":                 10,
    "Principado de Asturias":     11,
    "Comunidad Foral de Navarra": 13,
    "La Rioja":                   14,
    "Extremadura":                15,
    "Galicia":                    16,
    "Comunidad de Madrid":        17,
    "Region de Murcia":           18,
    "Comunitat Valenciana":       21,
}

TimeTrunc = Literal["ten-minutes", "hour", "day", "month", "year"]


# ---------------------------------------------------------------------------
# Capa de red
# ---------------------------------------------------------------------------

def fetch_demanda(
    indicador: str,
    start_date: str,
    end_date: str,
    time_trunc: TimeTrunc = "month",
    geo_limit: str = "peninsular",
    geo_ids: list[int] | None = None,
    retries: int = 3,
    delay: float = 1.0,
) -> dict:
    ruta = INDICADORES_DEMANDA.get(indicador, indicador)
    url  = f"{BASE_URL}/{ruta}"
    params = {
        "start_date": start_date,
        "end_date":   end_date,
        "time_trunc": time_trunc,
        "geo_limit":  geo_limit,
    }
    if geo_ids:
        params["geo_ids"] = ",".join(str(i) for i in geo_ids)

    for intento in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            print(f"  [HTTP {resp.status_code}] {e}")
            if resp.status_code == 429:
                time.sleep(delay * intento * 5)
            else:
                break
        except requests.exceptions.RequestException as e:
            print(f"  [Intento {intento}/{retries}] {e}")
            if intento < retries:
                time.sleep(delay * intento)
            else:
                raise
    return {}


# ---------------------------------------------------------------------------
# Parseo JSON -> DataFrame
# ---------------------------------------------------------------------------

def json_a_dataframe(data: dict) -> pd.DataFrame:
    """Convierte la respuesta JSON de REData en un DataFrame plano.

    Columnas: timestamp, anio, mes, indicador, valor, porcentaje, unidad
    """
    frames = []
    for serie in data.get("included", []):
        atributos = serie.get("attributes", {})
        titulo    = atributos.get("title", serie.get("type", ""))
        unidad    = atributos.get("magnitude") or ""
        valores   = atributos.get("values", [])

        df = pd.DataFrame(valores)
        if df.empty:
            continue

        df.rename(columns={
            "datetime":   "timestamp",
            "value":      "valor",
            "percentage": "porcentaje",
        }, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Madrid")
        df["anio"]      = df["timestamp"].dt.year
        df["mes"]       = df["timestamp"].dt.month
        df["indicador"] = titulo
        df["unidad"]    = unidad

        cols = ["timestamp", "anio", "mes", "indicador", "valor", "porcentaje", "unidad"]
        cols = [c for c in cols if c in df.columns]
        frames.append(df[cols])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Ventanas de descarga
# ---------------------------------------------------------------------------

def _ventanas(start_date: str, end_date: str, chunk_years: int) -> list[tuple[str, str]]:
    inicio = datetime.strptime(start_date, "%Y-%m-%dT%H:%M")
    fin    = datetime.strptime(end_date,   "%Y-%m-%dT%H:%M")
    result = []
    cursor = inicio
    while cursor < fin:
        cursor_fin = min(
            datetime(cursor.year + chunk_years, cursor.month, cursor.day, 23, 59),
            fin,
        )
        result.append((cursor.strftime("%Y-%m-%dT%H:%M"), cursor_fin.strftime("%Y-%m-%dT%H:%M")))
        # Advance to first of month containing cursor_fin; if that would not
        # advance cursor (end_date lands mid-month), jump to the next month.
        new_cursor = (cursor_fin.replace(hour=0, minute=0) + timedelta(days=1)).replace(day=1)
        if new_cursor <= cursor:
            m, y = cursor_fin.month, cursor_fin.year
            new_cursor = datetime(y + 1, 1, 1) if m == 12 else datetime(y, m + 1, 1)
        cursor = new_cursor
    return result


# ---------------------------------------------------------------------------
# Descarga por CCAA
# ---------------------------------------------------------------------------

def descargar_ccaa(
    start_date: str = "2020-01-01T00:00",
    end_date: str   = "2024-12-31T23:59",
    time_trunc: TimeTrunc = "month",
    ccaa_sel: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Demanda total por CCAA (endpoint evolucion, iterando comunidad a comunidad)."""
    comunidades = ccaa_sel if ccaa_sel is not None else CCAA
    chunk_years = 1 if time_trunc == "month" else 3
    ventanas    = _ventanas(start_date, end_date, chunk_years)
    frames      = []

    for nombre, geo_id in comunidades.items():
        print(f"  [{geo_id:>2}] {nombre}")
        ccaa_frames = []
        for v_start, v_end in ventanas:
            data = fetch_demanda(
                indicador="evolucion",
                start_date=v_start,
                end_date=v_end,
                time_trunc=time_trunc,
                geo_limit="ccaa",
                geo_ids=[geo_id],
            )
            df = json_a_dataframe(data)
            if not df.empty:
                ccaa_frames.append(df)
            time.sleep(0.25)

        if not ccaa_frames:
            print("       Sin datos.")
            continue

        df_ccaa = pd.concat(ccaa_frames, ignore_index=True).drop_duplicates("timestamp")
        df_ccaa.insert(1, "geo_id", geo_id)
        df_ccaa.insert(2, "ccaa",   nombre)
        frames.append(df_ccaa)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Descarga series peninsulares (IRE, perdidas, evolucion)
# ---------------------------------------------------------------------------

SERIES_PENINSULARES = {
    "Demanda peninsular":  "evolucion",
    "IRE General":         "ire_general",
    "IRE Industria":       "ire_industria",
    "IRE Servicios":       "ire_servicios",
    "Perdidas transporte": "perdidas_transporte",
}

def descargar_peninsular(
    start_date: str = "2015-01-01T00:00",
    end_date: str   = "2024-12-31T23:59",
    time_trunc: TimeTrunc = "month",
) -> dict[str, pd.DataFrame]:
    """Descarga todas las series peninsulares disponibles.

    Devuelve un dict {nombre_serie: DataFrame} listo para escribir en hojas Excel.
    """
    chunk_years = 1 if time_trunc == "month" else 3
    ventanas    = _ventanas(start_date, end_date, chunk_years)
    resultados  = {}

    for nombre, indicador in SERIES_PENINSULARES.items():
        print(f"  {nombre} ...")
        frames = []
        for v_start, v_end in ventanas:
            data = fetch_demanda(
                indicador=indicador,
                start_date=v_start,
                end_date=v_end,
                time_trunc=time_trunc,
                geo_limit="peninsular",
            )
            df = json_a_dataframe(data)
            if not df.empty:
                frames.append(df)
            time.sleep(0.25)

        if frames:
            # keep='last' porque _ventanas solapa el primer dia de cada anio:
            # la primera aparicion es el mes parcial (1 dia), la segunda es el mes completo.
            resultados[nombre] = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(["timestamp", "indicador"], keep="last")
                .sort_values(["indicador", "timestamp"])
                .reset_index(drop=True)
            )

    return resultados


# ---------------------------------------------------------------------------
# Guardado Excel multi-hoja
# ---------------------------------------------------------------------------

def _prep_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Quita el timezone para compatibilidad Excel (ya viene en Europe/Madrid)."""
    df = df.copy()
    for col in df.select_dtypes(include=["datetimetz"]).columns:
        df[col] = df[col].dt.tz_localize(None)
    return df


def guardar_excel(hojas: dict[str, pd.DataFrame], ruta: str) -> None:
    """Escribe un Excel con una hoja por cada entrada del dict."""
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        for nombre_hoja, df in hojas.items():
            df_clean = _prep_excel(df)
            df_clean.to_excel(writer, sheet_name=nombre_hoja[:31], index=False)
            print(f"  Hoja '{nombre_hoja}': {len(df_clean):,} filas, {len(df_clean.columns)} columnas")
    print(f"Guardado: {ruta}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _max_timestamp(df: pd.DataFrame) -> str | None:
    """Returns the next start date (day after max timestamp) as 'YYYY-MM-DDT00:00'."""
    if df.empty or "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"]).max()
    if pd.isna(ts):
        return None
    next_day = ts.normalize() + timedelta(days=1)
    return next_day.strftime("%Y-%m-%dT00:00")


def _merge(existing: pd.DataFrame, new: pd.DataFrame, dedup_cols: list[str]) -> pd.DataFrame:
    """Concatenates existing and new data, dropping duplicates on dedup_cols."""
    combined = pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(subset=dedup_cols, keep="last").sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    print("=== REData - Demanda electrica (actualizacion incremental) ===\n")

    end_date = datetime.now().strftime("%Y-%m-%dT%H:%M")
    print(f"Fecha fin: {end_date}\n")

    ROOT = Path(__file__).resolve().parents[1]
    SALIDA = str(ROOT / "data" / "raw" / "demanda_REData.xlsx")

    # Load existing Excel data
    existing: dict[str, pd.DataFrame] = {}
    try:
        xl = pd.ExcelFile(SALIDA)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            existing[sheet] = df
        print(f"Cargado existente: {SALIDA} ({len(existing)} hojas)\n")
    except FileNotFoundError:
        print("No existe archivo previo. Descarga completa.\n")

    hojas: dict[str, pd.DataFrame] = {}

    # --- CCAA Mensual ---
    ex_mes = existing.get("CCAA_Mensual", pd.DataFrame())
    start_mes = _max_timestamp(ex_mes) or "2020-01-01T00:00"
    print(f"Descargando demanda mensual por CCAA ({start_mes} -> {end_date})...")
    df_mes_new = descargar_ccaa(start_date=start_mes, end_date=end_date, time_trunc="month")
    if not df_mes_new.empty:
        hojas["CCAA_Mensual"] = _merge(ex_mes, df_mes_new, ["timestamp", "geo_id"])
    elif not ex_mes.empty:
        hojas["CCAA_Mensual"] = ex_mes
        print("  Sin datos nuevos; conservando existentes.")

    print()
    # --- CCAA Anual ---
    ex_anio = existing.get("CCAA_Anual", pd.DataFrame())
    start_anio = _max_timestamp(ex_anio) or "2015-01-01T00:00"
    print(f"Descargando demanda anual por CCAA ({start_anio} -> {end_date})...")
    df_anio_new = descargar_ccaa(start_date=start_anio, end_date=end_date, time_trunc="year")
    if not df_anio_new.empty:
        hojas["CCAA_Anual"] = _merge(ex_anio, df_anio_new, ["timestamp", "geo_id"])
    elif not ex_anio.empty:
        hojas["CCAA_Anual"] = ex_anio
        print("  Sin datos nuevos; conservando existentes.")

    print()
    # --- Series peninsulares mensuales ---
    ex_pen_sample = existing.get("Pen_Demanda peninsular", pd.DataFrame())
    start_pen_mes = _max_timestamp(ex_pen_sample) or "2015-01-01T00:00"
    print(f"Descargando series peninsulares mensuales ({start_pen_mes} -> {end_date})...")
    pen_mes = descargar_peninsular(start_date=start_pen_mes, end_date=end_date, time_trunc="month")
    for nombre, df_new in pen_mes.items():
        key = f"Pen_{nombre[:24]}"
        ex = existing.get(key, pd.DataFrame())
        hojas[key] = _merge(ex, df_new, ["timestamp", "indicador"]) if not df_new.empty else ex
    for key in [k for k in existing if k.startswith("Pen_") and k not in hojas]:
        hojas[key] = existing[key]

    print()
    # --- Series peninsulares anuales ---
    ex_pen_an_sample = existing.get("PenAnual_Demanda peninsular", pd.DataFrame())
    start_pen_anio = _max_timestamp(ex_pen_an_sample) or "2015-01-01T00:00"
    print(f"Descargando series peninsulares anuales ({start_pen_anio} -> {end_date})...")
    pen_anio = descargar_peninsular(start_date=start_pen_anio, end_date=end_date, time_trunc="year")
    for nombre, df_new in pen_anio.items():
        key = f"PenAnual_{nombre[:19]}"
        ex = existing.get(key, pd.DataFrame())
        hojas[key] = _merge(ex, df_new, ["timestamp", "indicador"]) if not df_new.empty else ex
    for key in [k for k in existing if k.startswith("PenAnual_") and k not in hojas]:
        hojas[key] = existing[key]

    # --- Guardar todo en un Excel ---
    print()
    guardar_excel(hojas, SALIDA)
    print("Listo.")
