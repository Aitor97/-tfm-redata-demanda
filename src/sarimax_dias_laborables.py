"""
SARIMAX extendido con dias laborables como exogena adicional.

Estrategia:
  1. Calcula dias_laborables por mes (lunes-viernes menos festivos nacionales
     de Espana) para todo el rango 2015-2025.
  2. Ajusta SARIMAX(0,1,2)(1,1,1,12) con dos conjuntos de exogenas:
       (a) baseline : [HDD18, CDD22]
       (b) extendido: [HDD18, CDD22, dias_laborables]
  3. Compara MAPE en hold-out 2024-2025 entre ambos.
  4. Guarda predicciones y reporte en reports/.

Uso:
    python src/sarimax_dias_laborables.py
"""

import os
import sys
import subprocess
import warnings


def _ensure_package(import_name: str, *pip_args: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"[setup] Instalando {pip_args[0]}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pip_args, "-q"])

_ensure_package("numpy", "numpy")
_ensure_package("pandas", "pandas")
_ensure_package("statsmodels", "statsmodels")
_ensure_package("holidays", "holidays")
_ensure_package("tabulate", "tabulate")

import numpy as np
import pandas as pd
import holidays
from statsmodels.tsa.statespace.sarimax import SARIMAX

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATOS    = os.path.join(ROOT_DIR, "data", "processed", "dataset_mensual.csv")
REPORTS  = os.path.join(ROOT_DIR, "reports")

ORDER          = (0, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 12)

TRAIN_INI = "2015-01-01"
TRAIN_FIN = "2023-12-01"
TEST_INI  = "2024-01-01"
TEST_FIN  = "2025-12-01"


def mape(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((real - pred) / real)) * 100)


def calcular_dias_laborables(idx: pd.DatetimeIndex) -> pd.Series:
    """Lunes-viernes del mes menos festivos nacionales de Espana."""
    anos = sorted({d.year for d in idx})
    fest_es = holidays.country_holidays("ES", years=anos)
    festivos = pd.to_datetime(list(fest_es.keys()))
    # solo cuentan los que caen en L-V
    festivos = festivos[festivos.weekday < 5]

    out = []
    for fecha_ini in idx:
        fecha_fin = fecha_ini + pd.offsets.MonthEnd(0)
        # rango diario del mes
        rango = pd.date_range(fecha_ini, fecha_fin, freq="D")
        laborables = rango[rango.weekday < 5]
        # restar festivos nacionales del mes
        fest_mes = festivos[(festivos >= fecha_ini) & (festivos <= fecha_fin)]
        out.append(len(laborables) - len(fest_mes))
    return pd.Series(out, index=idx, name="dias_laborables")


def ajustar_y_predecir(df: pd.DataFrame, exog_cols: list) -> tuple:
    train = df.loc[TRAIN_INI:TRAIN_FIN]
    test  = df.loc[TEST_INI:TEST_FIN]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            train["demanda_MWh"],
            exog=train[exog_cols],
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
        pred = modelo.forecast(steps=len(test), exog=test[exog_cols])
    pred.index = test.index
    return modelo, pred


def main():
    print("=" * 60)
    print("SARIMAX extendido: + dias_laborables")
    print("=" * 60)

    df = pd.read_csv(DATOS, parse_dates=["fecha"], index_col="fecha").asfreq("MS")
    df["dias_laborables"] = calcular_dias_laborables(df.index)
    print(f"Datos: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} meses)")
    print(f"\nResumen dias_laborables (rango {df['dias_laborables'].min()}-{df['dias_laborables'].max()}):")
    print(df["dias_laborables"].describe()[["mean", "std", "min", "max"]].to_string())

    # Correlacion exogena
    train = df.loc[TRAIN_INI:TRAIN_FIN]
    print(f"\nCorr dias_laborables vs demanda (train): {train['dias_laborables'].corr(train['demanda_MWh']):.4f}")

    # Modelo (a) baseline
    print("\n[1/2] Ajustando baseline (HDD18 + CDD22)...")
    modelo_base, pred_base = ajustar_y_predecir(df, ["HDD18", "CDD22"])
    y_test = df.loc[TEST_INI:TEST_FIN, "demanda_MWh"]
    mape_base = mape(y_test.values, pred_base.values)
    print(f"      AIC = {modelo_base.aic:.2f} | MAPE hold-out: {mape_base:.4f}%")

    # Modelo (b) extendido
    print("\n[2/2] Ajustando extendido (HDD18 + CDD22 + dias_laborables)...")
    modelo_ext, pred_ext = ajustar_y_predecir(df, ["HDD18", "CDD22", "dias_laborables"])
    mape_ext = mape(y_test.values, pred_ext.values)
    print(f"      AIC = {modelo_ext.aic:.2f} | MAPE hold-out: {mape_ext:.4f}%")

    # Significancia del coef de dias_laborables
    coef = modelo_ext.params.get("dias_laborables", None)
    pval = modelo_ext.pvalues.get("dias_laborables", None)

    mejora = mape_base - mape_ext
    print(f"\n{'='*60}")
    print(f"  Baseline (HDD+CDD)            : MAPE {mape_base:.4f}% | AIC {modelo_base.aic:.2f}")
    print(f"  Extendido (+dias_laborables)  : MAPE {mape_ext:.4f}% | AIC {modelo_ext.aic:.2f}")
    print(f"  Mejora absoluta               : {mejora:+.4f} pp")
    print(f"  Mejora relativa               : {mejora/mape_base*100:+.2f}%")
    print(f"  Coef dias_laborables          : {coef:.2f} (p = {pval:.4f})")
    print(f"{'='*60}")

    # Guardar predicciones
    resultados = pd.DataFrame({
        "fecha"      : y_test.index,
        "real_MWh"   : y_test.values,
        "pred_base"  : pred_base.values,
        "pred_ext"   : pred_ext.values,
        "error_base" : y_test.values - pred_base.values,
        "error_ext"  : y_test.values - pred_ext.values,
    }).set_index("fecha")

    out_csv = os.path.join(REPORTS, "sarimax_dias_laborables_predicciones.csv")
    resultados.to_csv(out_csv)
    print(f"\nPredicciones: {out_csv}")

    # Reporte markdown
    out_md = os.path.join(REPORTS, "sarimax_dias_laborables_resultado.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# SARIMAX extendido con dias laborables\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Configuracion\n\n")
        f.write(f"- Modelo: SARIMAX{ORDER}{SEASONAL_ORDER}\n")
        f.write(f"- Train: {TRAIN_INI} a {TRAIN_FIN}\n")
        f.write(f"- Hold-out: {TEST_INI} a {TEST_FIN} ({len(y_test)} meses)\n")
        f.write(f"- dias_laborables: L-V del mes menos festivos nacionales (libreria `holidays`, pais ES)\n\n")
        f.write("## Resultados\n\n")
        f.write("| Modelo | Exogenas | AIC | MAPE hold-out |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Baseline | HDD18, CDD22 | {modelo_base.aic:.2f} | {mape_base:.4f}% |\n")
        f.write(f"| Extendido | HDD18, CDD22, dias_laborables | {modelo_ext.aic:.2f} | {mape_ext:.4f}% |\n\n")
        f.write(f"**Mejora absoluta**: {mejora:+.4f} pp  \n")
        f.write(f"**Mejora relativa**: {mejora/mape_base*100:+.2f}%  \n")
        f.write(f"**Coef dias_laborables**: {coef:.2f} (p = {pval:.4f})\n\n")
        f.write("## Predicciones mensuales\n\n")
        f.write(resultados[["real_MWh", "pred_base", "pred_ext"]].to_markdown())
        f.write("\n")

    print(f"Reporte: {out_md}")


if __name__ == "__main__":
    main()
