"""
Paso 2 hibrido: SARIMAX + Lag-Llama (zero-shot sobre residuos).

Estrategia:
  1. Ajusta SARIMAX(0,1,2)(1,1,1,12) + HDD18/CDD22 sobre train 2015-2023.
  2. Predice hold-out 2024-2025 con SARIMAX.
  3. Alimenta Lag-Llama con los residuos in-sample y predice 24 pasos (zero-shot).
  4. Prediccion hibrida = pred_SARIMAX + pred_LagLlama_residuos.
  5. Calcula MAPE y compara con baseline SARIMAX (1.96%).
  6. Guarda resultados en reports/.

Uso:
    python src/hibrido_sarimax_lagllama.py

Requisitos:
    pip install "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama.git"
    pip install statsmodels pandas numpy torch
"""

import os
import sys
import subprocess
import warnings


# ---------------------------------------------------------------------------
# Instalacion de dependencias si faltan
# ---------------------------------------------------------------------------

def _ensure_package(import_name: str, *pip_args: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"[setup] Instalando {pip_args[0]}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pip_args, "-q"])

_ensure_package("numpy", "numpy")
_ensure_package("pandas", "pandas")
_ensure_package("statsmodels", "statsmodels")
_ensure_package("torch", "torch")

try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
except ImportError:
    print("[setup] Instalando lag-llama desde GitHub...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama.git",
        "-q",
    ])
    from lag_llama.gluon.estimator import LagLlamaEstimator


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from gluonts.dataset.common import ListDataset
from huggingface_hub import hf_hub_download

SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SRC_DIR)

DATOS        = os.path.join(ROOT_DIR, "data", "processed", "dataset_mensual.csv")
RESIDUOS_CSV = os.path.join(ROOT_DIR, "reports", "residuos_sarimax.csv")
REPORTS_DIR  = os.path.join(ROOT_DIR, "reports")

ORDER          = (0, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 12)
EXOG_COLS      = ["HDD18", "CDD22"]

TRAIN_INI = "2015-01-01"
TRAIN_FIN = "2023-12-01"
TEST_INI  = "2024-01-01"
TEST_FIN  = "2025-12-01"

MAPE_BASELINE = 1.96


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            train["demanda_MWh"],
            exog=train[EXOG_COLS],
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
    return modelo


def predecir_sarimax(modelo, df: pd.DataFrame) -> pd.Series:
    test = df.loc[TEST_INI:TEST_FIN]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = modelo.forecast(steps=len(test), exog=test[EXOG_COLS])
    pred.index = test.index
    return pred


def predecir_residuos_lagllama(residuos_train: pd.Series, n_pasos: int) -> np.ndarray:
    print("[lag-llama] Descargando checkpoint desde HuggingFace...")
    ckpt_path = hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="lag-llama.ckpt",
    )
    print(f"[lag-llama] Checkpoint en: {ckpt_path}")

    # PyTorch 2.6+ usa weights_only=True por defecto y rechaza el checkpoint
    # (contiene clases gluonts no listadas). Forzamos weights_only=False ya
    # que confiamos en el repo oficial de HuggingFace.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lag-llama] Dispositivo: {device}")

    # Leer hiperparametros del checkpoint para construir el estimator con la
    # misma arquitectura (input_size, n_layer, n_embd_per_head, n_head, ...).
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    # gluonts/pandas Period no acepta "MS"; usamos "M" para el freq del dataset
    start_period = pd.Period(residuos_train.index[0], freq="M")
    dataset = ListDataset(
        [{"start": start_period, "target": residuos_train.values}],
        freq="M",
    )

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=n_pasos,
        context_length=len(residuos_train),
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        device=device,
        batch_size=1,
        num_parallel_samples=100,
    )

    print(f"[lag-llama] Prediciendo {n_pasos} pasos de residuos (zero-shot)...")
    lightning_module = estimator.create_lightning_module()
    transformation   = estimator.create_transformation()
    predictor        = estimator.create_predictor(transformation, lightning_module)

    forecasts = list(predictor.predict(dataset))
    return forecasts[0].mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PASO 2 HIBRIDO: SARIMAX + Lag-Llama")
    print("=" * 60)

    df = cargar_datos()
    print(f"Datos cargados: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} meses)")

    # 1. Ajustar SARIMAX
    print("\n[1/4] Ajustando SARIMAX sobre train 2015-2023...")
    modelo = ajustar_sarimax(df)
    print(f"      AIC = {modelo.aic:.2f}")

    # 2. Predecir hold-out con SARIMAX
    print("\n[2/4] Prediccion SARIMAX en hold-out 2024-2025...")
    pred_sarimax = predecir_sarimax(modelo, df)
    y_test       = df.loc[TEST_INI:TEST_FIN, "demanda_MWh"]
    mape_sarimax = mape(y_test.values, pred_sarimax.values)
    print(f"      MAPE SARIMAX hold-out: {mape_sarimax:.4f}%")

    # 3. Predecir residuos con Lag-Llama
    print("\n[3/4] Prediccion de residuos con Lag-Llama (zero-shot)...")
    residuos_df    = pd.read_csv(RESIDUOS_CSV, parse_dates=["fecha"], index_col="fecha")
    residuos_train = residuos_df["residuo_sarimax"]
    n_pasos        = len(y_test)
    pred_residuos  = predecir_residuos_lagllama(residuos_train, n_pasos)

    # 4. Prediccion hibrida
    print("\n[4/4] Combinando predicciones...")
    pred_hibrida = pred_sarimax.values + pred_residuos
    mape_hibrido = mape(y_test.values, pred_hibrida)
    mejora       = mape_sarimax - mape_hibrido

    print(f"\n{'='*60}")
    print(f"  MAPE SARIMAX puro   : {mape_sarimax:.4f}%  (baseline: {MAPE_BASELINE}%)")
    print(f"  MAPE hibrido        : {mape_hibrido:.4f}%")
    print(f"  Mejora absoluta     : {mejora:+.4f} pp")
    print(f"  Mejora relativa     : {mejora/mape_sarimax*100:+.2f}%")
    print(f"{'='*60}")

    # Guardar predicciones CSV
    resultados = pd.DataFrame({
        "fecha"         : y_test.index,
        "real_MWh"      : y_test.values,
        "pred_sarimax"  : pred_sarimax.values,
        "pred_residuos" : pred_residuos,
        "pred_hibrida"  : pred_hibrida,
        "error_sarimax" : y_test.values - pred_sarimax.values,
        "error_hibrido" : y_test.values - pred_hibrida,
    }).set_index("fecha")

    out_csv = os.path.join(REPORTS_DIR, "hibrido_lagllama_predicciones.csv")
    resultados.to_csv(out_csv)
    print(f"\nPredicciones guardadas en: {out_csv}")

    # Guardar reporte markdown
    out_md = os.path.join(REPORTS_DIR, "hibrido_lagllama_resultado.md")
    with open(out_md, "w") as f:
        f.write("# Resultado Hibrido SARIMAX + Lag-Llama\n\n")
        f.write(f"**Fecha de ejecucion**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write(f"- Modelo base: SARIMAX{ORDER}{SEASONAL_ORDER} + HDD18/CDD22\n")
        f.write(f"- Corrector: Lag-Llama (zero-shot, 100 muestras probabilisticas)\n")
        f.write(f"- Hold-out: 2024-01 a 2025-12 ({n_pasos} meses)\n\n")
        f.write("## Resultados\n\n")
        f.write("| Modelo | MAPE hold-out |\n")
        f.write("|--------|---------------|\n")
        f.write(f"| SARIMAX puro (baseline) | {mape_sarimax:.4f}% |\n")
        f.write(f"| SARIMAX + Lag-Llama (hibrido) | {mape_hibrido:.4f}% |\n\n")
        f.write(f"**Mejora absoluta**: {mejora:+.4f} pp  \n")
        f.write(f"**Mejora relativa**: {mejora/mape_sarimax*100:+.2f}%\n\n")
        f.write("## Predicciones mensuales\n\n")
        f.write(resultados[["real_MWh", "pred_sarimax", "pred_hibrida"]].to_markdown())
        f.write("\n")

    print(f"Reporte guardado en: {out_md}")
    print("\nFin del paso 2 - Lag-Llama.")


if __name__ == "__main__":
    main()
