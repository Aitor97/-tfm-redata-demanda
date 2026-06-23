"""
Ensemble SARIMAX + Chronos-2 (50/50) sobre demanda mensual peninsular.

Promedio simple de:
  - SARIMAX(0,1,2)(1,1,1,12) + HDD18/CDD22
  - Chronos-2 zero-shot con HDD18/CDD22 como future covariates (mediana)

Reutiliza las predicciones de hold-out generadas por src/chronos2_directo.py.

Hold-out: 2024-01 .. 2025-12 (24 meses)

Uso:
    python src/ensemble_sarimax_chronos2.py
"""

import sys
import subprocess
from pathlib import Path


def _ensure(import_name, *pip_args):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pip_args, "-q"])


_ensure("numpy", "numpy")
_ensure("pandas", "pandas")
_ensure("tabulate", "tabulate")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PRED_CSV = REPORTS / "chronos2_directo_predicciones.csv"


def mape(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((real - pred) / real)) * 100)


def main():
    print("=" * 60)
    print("ENSEMBLE 50/50 SARIMAX + CHRONOS-2 (mensual)")
    print("=" * 60)

    pred = pd.read_csv(PRED_CSV, parse_dates=["fecha"], index_col="fecha")
    y = pred["real_MWh"]
    pred_s = pred["pred_sarimax"]
    pred_c = pred["pred_chronos2"]
    pred_ens = 0.5 * pred_s + 0.5 * pred_c

    mape_s = mape(y, pred_s)
    mape_c = mape(y, pred_c)
    mape_ens = mape(y, pred_ens)

    print(f"\nHold-out: {y.index[0].date()} -> {y.index[-1].date()} ({len(y)} meses)\n")
    print(f"  {'Modelo':<28} {'MAPE':>10}")
    print("  " + "-" * 40)
    print(f"  {'SARIMAX puro':<28} {mape_s:>9.4f}%")
    print(f"  {'Chronos-2 directo':<28} {mape_c:>9.4f}%")
    print(f"  {'Ensemble 50/50 (FINAL)':<28} {mape_ens:>9.4f}%")
    print("=" * 60)
    print(f"  Mejora absoluta vs SARIMAX : {mape_s - mape_ens:+.4f} pp")
    print(f"  Mejora relativa vs SARIMAX : {(mape_s - mape_ens)/mape_s*100:+.2f}%")
    print("=" * 60)

    res = pd.DataFrame({
        "real_MWh": y.values,
        "pred_sarimax": pred_s.values,
        "pred_chronos2": pred_c.values,
        "pred_ensemble": pred_ens.values,
        "error_ensemble": y.values - pred_ens.values,
    }, index=y.index)
    res.index.name = "fecha"
    res.to_csv(REPORTS / "ensemble_sarimax_chronos2_predicciones.csv")

    with open(REPORTS / "ensemble_sarimax_chronos2_resultado.md", "w", encoding="utf-8") as f:
        f.write("# Ensemble SARIMAX + Chronos-2 (50/50, mensual)\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write("- SARIMAX(0,1,2)(1,1,1,12) + HDD18/CDD22\n")
        f.write("- Chronos-2 zero-shot, HDD18/CDD22 como future covariates, mediana (q=0.5)\n")
        f.write("- Combinacion: promedio simple 50/50, sin tuning\n")
        f.write(f"- Hold-out: {y.index[0].date()} -> {y.index[-1].date()} ({len(y)} meses)\n\n")
        f.write("## Resultados\n\n")
        f.write("| Modelo | MAPE |\n|---|---|\n")
        f.write(f"| SARIMAX puro | {mape_s:.4f}% |\n")
        f.write(f"| Chronos-2 directo | {mape_c:.4f}% |\n")
        f.write(f"| **Ensemble 50/50 (FINAL)** | **{mape_ens:.4f}%** |\n\n")
        f.write(f"**Mejora vs SARIMAX**: {mape_s - mape_ens:+.4f} pp absoluta, "
                f"{(mape_s - mape_ens)/mape_s*100:+.2f}% relativa\n\n")
        f.write("## Predicciones mensuales\n\n")
        f.write(res.to_markdown())
        f.write("\n")

    print("\nGuardado en reports/ensemble_sarimax_chronos2_*.{csv,md}")


if __name__ == "__main__":
    main()
