"""
Chronos-2 directo (Estrategia A): demanda mensual peninsular con HDD18+CDD22
como known future covariates.

Comparable de tu a tu contra SARIMAX(0,1,2)(1,1,1,12) + HDD18/CDD22.

Train: 2015-01 .. 2023-12 (108 obs)
Test : 2024-01 .. 2025-12 (24 obs)

Uso:
    python src/chronos2_directo.py
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path


def _ensure_package(import_name, *pip_args):
    try:
        __import__(import_name)
    except ImportError:
        print(f"[setup] Instalando {pip_args[0]}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pip_args, "-q"])


_ensure_package("numpy", "numpy")
_ensure_package("pandas", "pandas")
_ensure_package("statsmodels", "statsmodels")
_ensure_package("torch", "torch")
_ensure_package("chronos", "chronos-forecasting")
_ensure_package("tabulate", "tabulate")


def _ensure_chronos2():
    """Chronos2Pipeline solo existe a partir de chronos-forecasting 1.6+."""
    try:
        from chronos import Chronos2Pipeline  # noqa: F401
        return
    except ImportError:
        print("[setup] chronos-forecasting no expone Chronos2Pipeline; actualizando...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "chronos-forecasting", "-q"]
        )


_ensure_chronos2()


import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "data" / "processed" / "dataset_mensual.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

ORDER = (0, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 12)
EXOG_COLS = ["HDD18", "CDD22"]

TRAIN_INI = "2015-01-01"
TRAIN_FIN = "2023-12-01"
TEST_INI = "2024-01-01"
TEST_FIN = "2025-12-01"


def mape(real, pred):
    return float(np.mean(np.abs((real - pred) / real)) * 100)


def cargar_datos():
    df = pd.read_csv(DATOS, parse_dates=["fecha"], index_col="fecha")
    df = df.asfreq("MS")
    return df


def sarimax_baseline(df):
    train = df.loc[TRAIN_INI:TRAIN_FIN]
    test = df.loc[TEST_INI:TEST_FIN]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            train["demanda_MWh"],
            exog=train[EXOG_COLS],
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
        ).fit(disp=False)
        pred = modelo.forecast(steps=len(test), exog=test[EXOG_COLS])
    pred.index = test.index
    return pred


def chronos2_directo(df):
    """Chronos-2 zero-shot con HDD18+CDD22 como known future covariates."""
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chronos2] Cargando amazon/chronos-2 en {device}...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
    )

    train = df.loc[TRAIN_INI:TRAIN_FIN].reset_index()
    test = df.loc[TEST_INI:TEST_FIN].reset_index()

    context_df = pd.DataFrame({
        "id": "demanda_es",
        "timestamp": train["fecha"],
        "target": train["demanda_MWh"].astype(float),
        "HDD18": train["HDD18"].astype(float),
        "CDD22": train["CDD22"].astype(float),
    })

    future_df = pd.DataFrame({
        "id": "demanda_es",
        "timestamp": test["fecha"],
        "HDD18": test["HDD18"].astype(float),
        "CDD22": test["CDD22"].astype(float),
    })

    print(f"[chronos2] Prediciendo {len(test)} pasos (HDD/CDD futuros)...")
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=len(test),
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Mediana
    median_col = ("target", "0.5") if ("target", "0.5") in pred_df.columns else "0.5"
    median = pred_df[median_col].values
    pred = pd.Series(median, index=pd.to_datetime(test["fecha"].values), name="pred_chronos2")
    pred.index.name = "fecha"
    return pred


def main():
    print("=" * 60)
    print("CHRONOS-2 DIRECTO: demanda mensual + HDD/CDD como covariates")
    print("=" * 60)

    df = cargar_datos()
    y_test = df.loc[TEST_INI:TEST_FIN, "demanda_MWh"]
    print(f"Train: {TRAIN_INI} -> {TRAIN_FIN}")
    print(f"Test : {TEST_INI} -> {TEST_FIN}  ({len(y_test)} meses)")

    print("\n[1/2] SARIMAX baseline...")
    pred_sarimax = sarimax_baseline(df)
    mape_sarimax = mape(y_test.values, pred_sarimax.values)
    print(f"      MAPE SARIMAX = {mape_sarimax:.4f}%")

    print("\n[2/2] Chronos-2 directo (zero-shot, HDD/CDD futuros)...")
    pred_c2 = chronos2_directo(df)
    mape_c2 = mape(y_test.values, pred_c2.values)
    print(f"      MAPE Chronos-2 = {mape_c2:.4f}%")

    mejora = mape_sarimax - mape_c2
    print("\n" + "=" * 60)
    print(f"  SARIMAX puro       : {mape_sarimax:.4f}%")
    print(f"  Chronos-2 directo  : {mape_c2:.4f}%")
    print(f"  Mejora absoluta    : {mejora:+.4f} pp")
    print(f"  Mejora relativa    : {mejora/mape_sarimax*100:+.2f}%")
    print("=" * 60)

    res = pd.DataFrame({
        "fecha": y_test.index,
        "real_MWh": y_test.values,
        "pred_sarimax": pred_sarimax.values,
        "pred_chronos2": pred_c2.values,
        "error_sarimax": y_test.values - pred_sarimax.values,
        "error_chronos2": y_test.values - pred_c2.values,
    }).set_index("fecha")
    res.to_csv(REPORTS / "chronos2_directo_predicciones.csv")

    with open(REPORTS / "chronos2_directo_resultado.md", "w", encoding="utf-8") as f:
        f.write("# Resultado Chronos-2 directo (covariates HDD/CDD)\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write("- Modelo: amazon/chronos-2 (zero-shot, 120 M params, encoder-only)\n")
        f.write("- Covariates futuras conocidas: HDD18, CDD22\n")
        f.write(f"- Train: {TRAIN_INI} -> {TRAIN_FIN}\n")
        f.write(f"- Test : {TEST_INI} -> {TEST_FIN}  ({len(y_test)} meses)\n")
        f.write("- Punto: cuantil 0.5 (mediana)\n\n")
        f.write("## Resultados\n\n")
        f.write("| Modelo | MAPE hold-out |\n")
        f.write("|---|---|\n")
        f.write(f"| SARIMAX puro (baseline) | {mape_sarimax:.4f}% |\n")
        f.write(f"| Chronos-2 directo | {mape_c2:.4f}% |\n\n")
        f.write(f"**Mejora absoluta**: {mejora:+.4f} pp\n\n")
        f.write(f"**Mejora relativa**: {mejora/mape_sarimax*100:+.2f}%\n\n")
        f.write("## Predicciones\n\n")
        f.write(res[["real_MWh", "pred_sarimax", "pred_chronos2"]].to_markdown())
        f.write("\n")

    print("\nGuardado en reports/chronos2_directo_*.{csv,md}")


if __name__ == "__main__":
    main()
