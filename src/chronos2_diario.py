"""
Chronos-2 directo sobre demanda diaria peninsular, rolling expanding 6 ventanas.

Replica el setup de src/modelos_diarios.py:
  - Train: expanding desde 2015-01-01
  - Test: anos completos 2020..2025 (6 iter)
  - Horizonte: 365 / 366 dias

Future covariates conocidas: HDD18, CDD22, is_holiday.
(dow, mes, doy, is_weekend son deducibles del timestamp y los maneja el modelo.)

Salidas:
  - reports/chronos2_diario_predicciones.csv (filas por modelo/ano_test/fecha)
  - reports/chronos2_diario_resumen.md      (MAPE diario, mensual, anual)

Uso:
    python src/chronos2_diario.py
"""

import sys
import subprocess
import warnings
import time
from pathlib import Path


def _ensure(import_name, *pip_args):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pip_args, "-q"])


_ensure("numpy", "numpy")
_ensure("pandas", "pandas")
_ensure("torch", "torch")
_ensure("chronos", "chronos-forecasting")
_ensure("tabulate", "tabulate")


def _ensure_chronos2():
    try:
        from chronos import Chronos2Pipeline  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "chronos-forecasting", "-q"]
        )


_ensure_chronos2()

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "data" / "processed" / "dataset_diario.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

ANOS_TEST = [2020, 2021, 2022, 2023, 2024, 2025]
COVARIATES = ["HDD18", "CDD22", "is_holiday"]


def mape(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((real - pred) / real)) * 100)


def cargar_datos():
    df = pd.read_csv(DATOS, parse_dates=["fecha"], index_col="fecha")
    df = df.asfreq("D")
    return df


def predecir_anio(df, pipeline, ano_test):
    """Chronos-2 zero-shot para el anio test, train expanding desde 2015-01-01."""
    train = df.loc["2015-01-01":f"{ano_test - 1}-12-31"]
    test = df.loc[f"{ano_test}-01-01":f"{ano_test}-12-31"]

    context_df = pd.DataFrame({
        "id": "demanda_es",
        "timestamp": train.index,
        "target": train["demanda_MWh"].astype(float).values,
        "HDD18": train["HDD18"].astype(float).values,
        "CDD22": train["CDD22"].astype(float).values,
        "is_holiday": train["is_holiday"].astype(float).values,
    })
    future_df = pd.DataFrame({
        "id": "demanda_es",
        "timestamp": test.index,
        "HDD18": test["HDD18"].astype(float).values,
        "CDD22": test["CDD22"].astype(float).values,
        "is_holiday": test["is_holiday"].astype(float).values,
    })

    t0 = time.time()
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=len(test),
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
    elapsed = time.time() - t0
    median_col = ("target", "0.5") if ("target", "0.5") in pred_df.columns else "0.5"
    pred = pd.Series(
        pred_df[median_col].values,
        index=test.index,
        name="pred_chronos2",
    )
    return pred, test["demanda_MWh"], elapsed


def main():
    print("=" * 70)
    print("CHRONOS-2 DIARIO - rolling expanding 6 ventanas (2020..2025)")
    print("=" * 70)

    df = cargar_datos()
    print(f"Dataset: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} dias)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nCargando amazon/chronos-2 en {device}...")
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)
    print("Modelo cargado.\n")

    filas = []
    resumen_mape_d = []
    resumen_mape_m = []
    resumen_anual = []  # (ano, real_anual, pred_anual)

    for ano in ANOS_TEST:
        print(f"[{ano}] Prediciendo...", flush=True)
        pred, real, elapsed = predecir_anio(df, pipeline, ano)
        mape_d = mape(real.values, pred.values)
        # mensual
        df_iter = pd.DataFrame({"real": real.values, "pred": pred.values}, index=real.index)
        mens = df_iter.resample("MS").sum()
        mape_m = mape(mens["real"].values, mens["pred"].values)
        # anual
        real_anual = float(real.values.sum())
        pred_anual = float(pred.values.sum())

        resumen_mape_d.append((ano, mape_d))
        resumen_mape_m.append((ano, mape_m))
        resumen_anual.append((ano, real_anual, pred_anual))

        for f, r, p in zip(real.index, real.values, pred.values):
            filas.append({
                "modelo": "chronos2",
                "ano_test": ano,
                "fecha": f,
                "real": float(r),
                "pred": float(p),
            })

        print(f"  MAPE diario   = {mape_d:.4f}%")
        print(f"  MAPE mensual  = {mape_m:.4f}%")
        print(f"  Tiempo Chronos-2: {elapsed:.1f}s")
        print()

    # Tabla por ano
    df_pred = pd.DataFrame(filas)
    df_pred.to_csv(REPORTS / "chronos2_diario_predicciones.csv", index=False)

    df_mape_d = pd.DataFrame(resumen_mape_d, columns=["ano_test", "MAPE_diario"])
    df_mape_m = pd.DataFrame(resumen_mape_m, columns=["ano_test", "MAPE_mensual"])
    df_anual = pd.DataFrame(resumen_anual, columns=["ano_test", "real_anual", "pred_anual"])
    df_anual["MAPE_anual"] = (df_anual["real_anual"] - df_anual["pred_anual"]).abs() \
        / df_anual["real_anual"] * 100

    # Resumen global
    mape_d_mean = df_mape_d["MAPE_diario"].mean()
    mape_d_std = df_mape_d["MAPE_diario"].std()
    mape_m_mean = df_mape_m["MAPE_mensual"].mean()
    mape_m_std = df_mape_m["MAPE_mensual"].std()
    mape_a_mean = df_anual["MAPE_anual"].mean()
    mape_a_std = df_anual["MAPE_anual"].std()

    print("=" * 70)
    print(f"  MAPE diario  (media 6 iter): {mape_d_mean:.4f}% +/- {mape_d_std:.4f}")
    print(f"  MAPE mensual (media 6 iter): {mape_m_mean:.4f}% +/- {mape_m_std:.4f}")
    print(f"  MAPE anual   (media 6 iter): {mape_a_mean:.4f}% +/- {mape_a_std:.4f}")
    print("=" * 70)
    print("\nReferencia ensemble diario actual (memoria):")
    print("  MAPE mensual: 2.9564% +/- 1.72")
    print("  MAPE anual:   2.4237% +/- 1.85")

    # Reporte md
    with open(REPORTS / "chronos2_diario_resumen.md", "w", encoding="utf-8") as f:
        f.write("# Chronos-2 diario (rolling expanding 6 ventanas)\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write("- Modelo: amazon/chronos-2 zero-shot (120 M params)\n")
        f.write(f"- Train expanding desde 2015-01-01; test cada ano 2020..2025\n")
        f.write(f"- Future covariates: {', '.join(COVARIATES)}\n")
        f.write("- Punto: cuantil 0.5 (mediana)\n\n")
        f.write("## Resumen global (media 6 iter)\n\n")
        f.write("| Metric | MAPE | std |\n|---|---|---|\n")
        f.write(f"| Diario | {mape_d_mean:.4f}% | {mape_d_std:.4f} |\n")
        f.write(f"| Mensual | {mape_m_mean:.4f}% | {mape_m_std:.4f} |\n")
        f.write(f"| Anual | {mape_a_mean:.4f}% | {mape_a_std:.4f} |\n\n")
        f.write("## Detalle por iteracion\n\n")
        det = df_mape_d.merge(df_mape_m, on="ano_test").merge(
            df_anual[["ano_test", "MAPE_anual"]], on="ano_test"
        )
        f.write(det.to_markdown(index=False))
        f.write("\n\n## Comparacion con ensemble diario actual\n\n")
        f.write("Referencia (memoria, 6 iter):\n\n")
        f.write("- Ensemble (naive_trend + Prophet + LightGBM): MAPE mensual 2.9564 % +/- 1.72, anual 2.4237 % +/- 1.85\n")
        f.write("- Baseline mensual SARIMAX -> anual: 2.9168 % +/- 1.97\n")

    print("\nGuardado en reports/chronos2_diario_*.{csv,md}")


if __name__ == "__main__":
    main()
