"""
Ensemble diario 50/50: Chronos-2 + ensemble actual (naive_trend + Prophet + LightGBM).

Reutiliza predicciones ya generadas:
  - reports/diario_predicciones.csv          (modelos clasicos + ensemble actual)
  - reports/chronos2_diario_predicciones.csv (Chronos-2 zero-shot)

Combina dia a dia para los 6 anos test (2020..2025) y reporta MAPE diario,
mensual y anual.

Uso:
    python src/ensemble_diario_chronos2.py
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
PRED_DIARIO = REPORTS / "diario_predicciones.csv"
PRED_CHRONOS = REPORTS / "chronos2_diario_predicciones.csv"

ANOS_TEST = [2020, 2021, 2022, 2023, 2024, 2025]


def mape(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((real - pred) / real)) * 100)


def metricas_iter(real, pred, fechas):
    df = pd.DataFrame({"real": real, "pred": pred}, index=pd.to_datetime(fechas))
    md = mape(df["real"].values, df["pred"].values)
    mens = df.resample("MS").sum()
    mm = mape(mens["real"].values, mens["pred"].values)
    ra = float(df["real"].sum())
    pa = float(df["pred"].sum())
    ma = abs(ra - pa) / ra * 100
    return md, mm, ma


def main():
    print("=" * 70)
    print("ENSEMBLE DIARIO 50/50: Chronos-2 + ensemble actual")
    print("=" * 70)

    df_d = pd.read_csv(PRED_DIARIO, parse_dates=["fecha"])
    df_c = pd.read_csv(PRED_CHRONOS, parse_dates=["fecha"])

    df_ens = df_d[df_d["modelo"] == "ensemble"][["ano_test", "fecha", "real", "pred"]] \
        .rename(columns={"pred": "pred_ens_actual"})
    df_chr = df_c[df_c["modelo"] == "chronos2"][["ano_test", "fecha", "pred"]] \
        .rename(columns={"pred": "pred_chronos2"})

    m = df_ens.merge(df_chr, on=["ano_test", "fecha"], how="inner")
    if len(m) != len(df_ens):
        print(f"AVISO: cruce parcial ({len(m)} vs ensemble {len(df_ens)})")

    m["pred_50_50"] = 0.5 * m["pred_ens_actual"] + 0.5 * m["pred_chronos2"]

    filas_res = []
    for ano in ANOS_TEST:
        sub = m[m["ano_test"] == ano].sort_values("fecha")
        md_e, mm_e, ma_e = metricas_iter(sub["real"].values, sub["pred_ens_actual"].values, sub["fecha"])
        md_c, mm_c, ma_c = metricas_iter(sub["real"].values, sub["pred_chronos2"].values, sub["fecha"])
        md_n, mm_n, ma_n = metricas_iter(sub["real"].values, sub["pred_50_50"].values, sub["fecha"])
        filas_res.append({
            "ano": ano,
            "MAPE_mes_ens_actual": mm_e,
            "MAPE_mes_chronos2": mm_c,
            "MAPE_mes_50_50": mm_n,
            "MAPE_anual_ens_actual": ma_e,
            "MAPE_anual_chronos2": ma_c,
            "MAPE_anual_50_50": ma_n,
            "MAPE_dia_50_50": md_n,
        })

    df_res = pd.DataFrame(filas_res)
    print("\nPor ano:\n")
    print(df_res[["ano", "MAPE_mes_ens_actual", "MAPE_mes_chronos2", "MAPE_mes_50_50",
                  "MAPE_anual_50_50"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("RESUMEN (media 6 iter)")
    print("=" * 70)
    for col, label in [
        ("MAPE_mes_ens_actual", "Mensual ensemble actual"),
        ("MAPE_mes_chronos2",   "Mensual Chronos-2"),
        ("MAPE_mes_50_50",      "Mensual 50/50 (FINAL)"),
        ("MAPE_anual_ens_actual", "Anual   ensemble actual"),
        ("MAPE_anual_chronos2",   "Anual   Chronos-2"),
        ("MAPE_anual_50_50",      "Anual   50/50 (FINAL)"),
    ]:
        mu = df_res[col].mean()
        sd = df_res[col].std()
        print(f"  {label:<32}: {mu:.4f}% +/- {sd:.4f}")
    print("=" * 70)

    mens_actual = df_res["MAPE_mes_ens_actual"].mean()
    mens_5050 = df_res["MAPE_mes_50_50"].mean()
    anual_actual = df_res["MAPE_anual_ens_actual"].mean()
    anual_5050 = df_res["MAPE_anual_50_50"].mean()
    print(f"\n  Mejora vs ensemble actual:")
    print(f"    Mensual: {mens_actual - mens_5050:+.4f} pp  ({(mens_actual - mens_5050)/mens_actual*100:+.2f}%)")
    print(f"    Anual:   {anual_actual - anual_5050:+.4f} pp  ({(anual_actual - anual_5050)/anual_actual*100:+.2f}%)")

    # Guardar predicciones combinadas
    m.to_csv(REPORTS / "ensemble_diario_chronos2_predicciones.csv", index=False)
    df_res.to_csv(REPORTS / "ensemble_diario_chronos2_resumen_por_ano.csv", index=False)

    # Reporte
    with open(REPORTS / "ensemble_diario_chronos2_resultado.md", "w", encoding="utf-8") as f:
        f.write("# Ensemble diario 50/50: Chronos-2 + ensemble actual\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write("- A: ensemble actual = mean(naive_trend + Prophet + LightGBM)\n")
        f.write("- B: Chronos-2 zero-shot + HDD18/CDD22/is_holiday (future covariates)\n")
        f.write("- Combinacion: 0.5 A + 0.5 B (sin tuning)\n")
        f.write("- Rolling expanding 6 ventanas, anos test 2020..2025\n\n")
        f.write("## Resumen global (media 6 iter)\n\n")
        f.write("| Modelo | MAPE mensual | MAPE anual |\n|---|---|---|\n")
        f.write(f"| Ensemble actual (A) | {df_res['MAPE_mes_ens_actual'].mean():.4f}% +/- {df_res['MAPE_mes_ens_actual'].std():.4f} | "
                f"{df_res['MAPE_anual_ens_actual'].mean():.4f}% +/- {df_res['MAPE_anual_ens_actual'].std():.4f} |\n")
        f.write(f"| Chronos-2 (B) | {df_res['MAPE_mes_chronos2'].mean():.4f}% +/- {df_res['MAPE_mes_chronos2'].std():.4f} | "
                f"{df_res['MAPE_anual_chronos2'].mean():.4f}% +/- {df_res['MAPE_anual_chronos2'].std():.4f} |\n")
        f.write(f"| **Ensemble 50/50 (FINAL)** | **{df_res['MAPE_mes_50_50'].mean():.4f}% +/- {df_res['MAPE_mes_50_50'].std():.4f}** | "
                f"**{df_res['MAPE_anual_50_50'].mean():.4f}% +/- {df_res['MAPE_anual_50_50'].std():.4f}** |\n\n")
        f.write(f"**Mejora vs ensemble actual**: mensual {mens_actual - mens_5050:+.4f} pp "
                f"({(mens_actual - mens_5050)/mens_actual*100:+.2f}%), "
                f"anual {anual_actual - anual_5050:+.4f} pp "
                f"({(anual_actual - anual_5050)/anual_actual*100:+.2f}%)\n\n")
        f.write("## Detalle por ano (MAPE)\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n")

    print("\nGuardado en reports/ensemble_diario_chronos2_*.{csv,md}")


if __name__ == "__main__":
    main()
