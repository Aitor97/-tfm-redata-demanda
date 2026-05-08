"""
Paso 2 hibrido: SARIMAX + Lag-Llama (zero-shot sobre residuos).

Estrategia identica a hibrido_sarimax_chronos.py pero usando Lag-Llama
como corrector de residuos.

Uso:
    python src/hibrido_sarimax_lagllama.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

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


def predecir_residuos_lagllama(residuos_train: pd.Series, n_pasos: int) -> np.ndarray:
    from huggingface_hub import hf_hub_download
    from gluonts.dataset.common import ListDataset
    from lag_llama.gluon.estimator import LagLlamaEstimator

    print("[lag-llama] Descargando checkpoint desde HuggingFace...")
    ckpt_path = hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="lag-llama.ckpt",
    )
    print(f"[lag-llama] Checkpoint en: {ckpt_path}")

    # Contexto: todos los residuos in-sample
    context_length = len(residuos_train)

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=n_pasos,
        context_length=context_length,
        device=torch.device("cpu"),
        batch_size=1,
        num_parallel_samples=100,
    )

    # GluonTS requiere un ListDataset con freq y target
    dataset = ListDataset(
        [{"start": residuos_train.index[0], "target": residuos_train.values}],
        freq="MS",
    )

    print(f"[lag-llama] Prediciendo {n_pasos} pasos de residuos (zero-shot)...")
    predictor = estimator.create_predictor(
        training_data=dataset,
        num_samples=100,
    )

    forecasts = list(predictor.predict(dataset))
    # mean de las 100 muestras
    pred_mean = forecasts[0].mean
    return pred_mean


def main():
    print("=" * 60)
    print("PASO 2 HIBRIDO: SARIMAX + Lag-Llama")
    print("=" * 60)

    df = cargar_datos()
    print(f"Datos cargados: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} meses)")

    print("\n[1/4] Ajustando SARIMAX sobre train 2015-2023...")
    modelo, y_train, x_train = ajustar_sarimax(df)
    print(f"      AIC = {modelo.aic:.2f}")

    print("\n[2/4] Prediccion SARIMAX en hold-out 2024-2025...")
    pred_sarimax = predecir_sarimax(modelo, df)
    y_test = df.loc[TEST_INI:TEST_FIN, "demanda_MWh"]
    mape_sarimax = mape(y_test.values, pred_sarimax.values)
    print(f"      MAPE SARIMAX hold-out: {mape_sarimax:.4f}%")

    print("\n[3/4] Prediccion de residuos con Lag-Llama (zero-shot)...")
    residuos_df   = pd.read_csv(RESIDUOS_CSV, parse_dates=["fecha"], index_col="fecha")
    residuos_train = residuos_df["residuo_sarimax"]
    n_pasos        = len(y_test)
    pred_residuos  = predecir_residuos_lagllama(residuos_train, n_pasos)

    print("\n[4/4] Combinando predicciones (hibrido)...")
    pred_hibrida = pred_sarimax.values + pred_residuos
    mape_hibrido = mape(y_test.values, pred_hibrida)
    mejora       = mape_sarimax - mape_hibrido

    print(f"\n{'='*60}")
    print(f"  MAPE SARIMAX puro   : {mape_sarimax:.4f}%  (baseline: {MAPE_BASELINE}%)")
    print(f"  MAPE hibrido        : {mape_hibrido:.4f}%")
    print(f"  Mejora absoluta     : {mejora:+.4f} pp")
    print(f"  Mejora relativa     : {mejora/mape_sarimax*100:+.2f}%")
    print(f"{'='*60}")

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

    out_md = os.path.join(REPORTS_DIR, "hibrido_lagllama_resultado.md")
    with open(out_md, "w") as f:
        f.write("# Resultado Hibrido SARIMAX + Lag-Llama\n\n")
        f.write(f"**Fecha de ejecucion**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write(f"- Modelo base: SARIMAX{ORDER}{SEASONAL_ORDER} + HDD18/CDD22\n")
        f.write(f"- Corrector: Lag-Llama (zero-shot, 100 muestras)\n")
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
    print("\nFin del paso 2 Lag-Llama.")


if __name__ == "__main__":
    main()
