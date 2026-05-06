"""
Paso 2 hibrido: SARIMAX + Chronos (Amazon).

Estrategia:
  1. Ajusta SARIMAX(0,1,2)(1,1,1,12) + HDD18/CDD22 sobre train 2015-2023.
  2. Predice hold-out 2024-2025 con SARIMAX.
  3. Alimenta Chronos con los residuos in-sample (series historica de errores)
     y le pide predecir los 24 pasos del hold-out (zero-shot).
  4. Prediccion hibrida = pred_SARIMAX + pred_Chronos_residuos.
  5. Calcula MAPE y lo compara con la linea base SARIMAX (1.96%).
  6. Guarda resultados en reports/.

Uso:
    python src/hibrido_sarimax_chronos.py
"""

import os
import sys
import subprocess
import warnings

# ---------------------------------------------------------------------------
# Instalacion de dependencias si faltan
# ---------------------------------------------------------------------------

def _ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"[setup] Instalando {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])

_ensure_package("numpy", "numpy")
_ensure_package("pandas", "pandas")
_ensure_package("statsmodels", "statsmodels")
_ensure_package("torch", "torch --index-url https://download.pytorch.org/whl/cpu")
_ensure_package("chronos", "chronos-forecasting")

# ---------------------------------------------------------------------------
# Imports tras instalacion
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

DATOS         = os.path.join(ROOT_DIR, "data", "processed", "dataset_mensual.csv")
RESIDUOS_CSV  = os.path.join(ROOT_DIR, "reports", "residuos_sarimax.csv")
REPORTS_DIR   = os.path.join(ROOT_DIR, "reports")

ORDER         = (0, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 12)
EXOG_COLS     = ["HDD18", "CDD22"]

TRAIN_INI = "2015-01-01"
TRAIN_FIN = "2023-12-01"
TEST_INI  = "2024-01-01"
TEST_FIN  = "2025-12-01"

MAPE_BASELINE = 1.96  # % MAPE SARIMAX puro en hold-out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mape(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((real - pred) / real)) * 100)


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATOS, parse_dates=["fecha"], index_col="fecha")
    df = df.asfreq("MS")
    return df


def ajustar_sarimax(df: pd.DataFrame):
    train = df.loc[TRAIN_INI:TRAIN_FIN]
    y_train = train["demanda_MWh"]
    x_train = train[EXOG_COLS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            y_train,
            exog=x_train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
    return modelo, y_train, x_train


def predecir_sarimax(modelo, df: pd.DataFrame) -> pd.Series:
    test = df.loc[TEST_INI:TEST_FIN]
    x_test = test[EXOG_COLS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = modelo.forecast(steps=len(test), exog=x_test)
    pred.index = test.index
    return pred


def predecir_residuos_chronos(residuos_train: pd.Series, n_pasos: int) -> np.ndarray:
    from chronos import ChronosPipeline

    print("[chronos] Cargando modelo amazon/chronos-t5-small (zero-shot)...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    contexto = torch.tensor(residuos_train.values, dtype=torch.float32).unsqueeze(0)

    print(f"[chronos] Prediciendo {n_pasos} pasos de residuos...")
    quantiles, mean = pipeline.predict(
        context=contexto,
        prediction_length=n_pasos,
        num_samples=100,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        limit_prediction_length=False,
    )
    # mean shape: (1, n_pasos)
    return mean.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PASO 2 HIBRIDO: SARIMAX + Chronos")
    print("=" * 60)

    df = cargar_datos()
    print(f"Datos cargados: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} meses)")

    # 1. Ajustar SARIMAX
    print("\n[1/4] Ajustando SARIMAX sobre train 2015-2023...")
    modelo, y_train, x_train = ajustar_sarimax(df)
    print(f"      AIC = {modelo.aic:.2f}")

    # 2. Predecir hold-out con SARIMAX
    print("\n[2/4] Prediccion SARIMAX en hold-out 2024-2025...")
    pred_sarimax = predecir_sarimax(modelo, df)
    y_test = df.loc[TEST_INI:TEST_FIN, "demanda_MWh"]
    mape_sarimax = mape(y_test.values, pred_sarimax.values)
    print(f"      MAPE SARIMAX hold-out: {mape_sarimax:.4f}%")

    # 3. Cargar residuos in-sample y predecir con Chronos
    print("\n[3/4] Prediccion de residuos con Chronos (zero-shot)...")
    residuos_df = pd.read_csv(RESIDUOS_CSV, parse_dates=["fecha"], index_col="fecha")
    residuos_train = residuos_df["residuo_sarimax"]
    n_pasos = len(y_test)
    pred_residuos = predecir_residuos_chronos(residuos_train, n_pasos)

    # 4. Prediccion hibrida
    print("\n[4/4] Combinando predicciones (hibrido)...")
    pred_hibrida = pred_sarimax.values + pred_residuos
    mape_hibrido = mape(y_test.values, pred_hibrida)
    mejora = mape_sarimax - mape_hibrido

    print(f"\n{'='*60}")
    print(f"  MAPE SARIMAX puro   : {mape_sarimax:.4f}%  (baseline publicado: {MAPE_BASELINE}%)")
    print(f"  MAPE hibrido        : {mape_hibrido:.4f}%")
    print(f"  Mejora absoluta     : {mejora:+.4f} pp")
    print(f"  Mejora relativa     : {mejora/mape_sarimax*100:+.2f}%")
    print(f"{'='*60}")

    # Guardar tabla de predicciones
    resultados = pd.DataFrame({
        "fecha"         : y_test.index,
        "real_MWh"      : y_test.values,
        "pred_sarimax"  : pred_sarimax.values,
        "pred_residuos" : pred_residuos,
        "pred_hibrida"  : pred_hibrida,
        "error_sarimax" : y_test.values - pred_sarimax.values,
        "error_hibrido" : y_test.values - pred_hibrida,
    }).set_index("fecha")

    out_csv = os.path.join(REPORTS_DIR, "hibrido_predicciones.csv")
    resultados.to_csv(out_csv)
    print(f"\nPredicciones guardadas en: {out_csv}")

    # Guardar resumen en markdown
    out_md = os.path.join(REPORTS_DIR, "hibrido_resultado.md")
    with open(out_md, "w") as f:
        f.write("# Resultado Hibrido SARIMAX + Chronos\n\n")
        f.write(f"**Fecha de ejecucion**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write(f"- Modelo base: SARIMAX{ORDER}{SEASONAL_ORDER} + HDD18/CDD22\n")
        f.write(f"- Corrector: amazon/chronos-t5-small (zero-shot, 100 muestras)\n")
        f.write(f"- Hold-out: 2024-01 a 2025-12 ({n_pasos} meses)\n\n")
        f.write("## Resultados\n\n")
        f.write(f"| Modelo | MAPE hold-out |\n")
        f.write(f"|--------|---------------|\n")
        f.write(f"| SARIMAX puro (baseline) | {mape_sarimax:.4f}% |\n")
        f.write(f"| SARIMAX + Chronos (hibrido) | {mape_hibrido:.4f}% |\n\n")
        f.write(f"**Mejora absoluta**: {mejora:+.4f} pp  \n")
        f.write(f"**Mejora relativa**: {mejora/mape_sarimax*100:+.2f}%\n\n")
        f.write("## Predicciones mensuales\n\n")
        f.write(resultados[["real_MWh", "pred_sarimax", "pred_hibrida"]].to_markdown())
        f.write("\n")

    print(f"Reporte guardado en: {out_md}")
    print("\nFin del paso 2.")


if __name__ == "__main__":
    main()
