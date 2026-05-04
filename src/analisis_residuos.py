"""
Analisis de autocorrelacion de residuos SARIMA/SARIMAX.

Test de Ljung-Box para decidir si un enfoque hibrido con foundation models
(Chronos/TimesFM) puede aportar mejora al modelo SARIMAX ganador.

Uso:
    python src/analisis_residuos.py

Salidas:
    reports/ljungbox.md        -- reporte ejecutivo con veredicto
    reports/residuos_sarimax.csv -- residuos in-sample del train completo
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

DATOS = os.path.join(ROOT_DIR, "data", "processed", "dataset_mensual.csv")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

ORDER = (0, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 12)
EXOG_COLS = ["HDD18", "CDD22"]

TRAIN_INI = "2015-01-01"
TRAIN_FIN = "2023-12-01"

# Train-end de cada fold CV (el test es el año siguiente, no se usa aqui)
FOLDS = {
    "fold_2019": "2019-12-01",
    "fold_2020": "2020-12-01",
    "fold_2021": "2021-12-01",
    "fold_2022": "2022-12-01",
}

LAGS = [12, 24, 36]


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATOS, parse_dates=["fecha"], index_col="fecha")
    df = df.asfreq("MS")
    return df


def ajustar_modelo(y: pd.Series, exog: pd.DataFrame | None) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            y,
            exog=exog,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
    return modelo


def ljung_box_pvalues(residuos: pd.Series, lags: list[int]) -> dict:
    resultado = acorr_ljungbox(residuos.dropna(), lags=lags, return_df=True)
    return {lag: float(resultado.loc[lag, "lb_pvalue"]) for lag in lags}


def estadisticas(residuos: pd.Series, demanda_media: float) -> dict:
    r = residuos.dropna()
    return {
        "std_MWh": float(r.std()),
        "std_pct": float(r.std() / demanda_media * 100),
        "n": int(r.count()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = cargar_datos()

    mask_full = (df.index >= TRAIN_INI) & (df.index <= TRAIN_FIN)
    demanda_media = float(df.loc[mask_full, "demanda_MWh"].mean())

    resultados: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Train completo (pre-holdout): 2015-01 a 2023-12
    # -----------------------------------------------------------------------
    y_full = df.loc[mask_full, "demanda_MWh"]
    exog_full = df.loc[mask_full, EXOG_COLS]

    print("Ajustando SARIMA  (train completo, 108 meses)...")
    fit_sarima = ajustar_modelo(y_full, exog=None)
    resid_sarima = fit_sarima.resid

    print("Ajustando SARIMAX (train completo, 108 meses)...")
    fit_sarimax = ajustar_modelo(y_full, exog=exog_full)
    resid_sarimax = fit_sarimax.resid

    for nombre, resid in [("SARIMA_full", resid_sarima), ("SARIMAX_full", resid_sarimax)]:
        lb = ljung_box_pvalues(resid, LAGS)
        stats = estadisticas(resid, demanda_media)
        resultados[nombre] = {**lb, **stats}
        print(f"  {nombre}: p-values LB = {lb}")

    # -----------------------------------------------------------------------
    # Folds CV (analisis de robustez)
    # -----------------------------------------------------------------------
    for fold_nombre, fin in FOLDS.items():
        mask_fold = (df.index >= TRAIN_INI) & (df.index <= fin)
        y_fold = df.loc[mask_fold, "demanda_MWh"]
        exog_fold = df.loc[mask_fold, EXOG_COLS]
        n_meses = mask_fold.sum()

        print(f"\nFold {fold_nombre} (hasta {fin[:7]}, {n_meses} meses):")

        print(f"  Ajustando SARIMA...")
        fit_s = ajustar_modelo(y_fold, exog=None)
        lb_s = ljung_box_pvalues(fit_s.resid, LAGS)
        stats_s = estadisticas(fit_s.resid, demanda_media)
        resultados[f"SARIMA_{fold_nombre}"] = {**lb_s, **stats_s}
        print(f"    p-values LB = {lb_s}")

        print(f"  Ajustando SARIMAX...")
        fit_sx = ajustar_modelo(y_fold, exog=exog_fold)
        lb_sx = ljung_box_pvalues(fit_sx.resid, LAGS)
        stats_sx = estadisticas(fit_sx.resid, demanda_media)
        resultados[f"SARIMAX_{fold_nombre}"] = {**lb_sx, **stats_sx}
        print(f"    p-values LB = {lb_sx}")

    # -----------------------------------------------------------------------
    # CSV de residuos (train completo, para uso en paso 2)
    # -----------------------------------------------------------------------
    df_resid = pd.DataFrame(
        {
            "residuo_sarima": resid_sarima,
            "residuo_sarimax": resid_sarimax,
        },
        index=y_full.index,
    )
    df_resid.index.name = "fecha"
    csv_path = os.path.join(REPORTS_DIR, "residuos_sarimax.csv")
    df_resid.to_csv(csv_path)
    print(f"\nCSV de residuos guardado: {csv_path}")

    # -----------------------------------------------------------------------
    # Reporte Markdown
    # -----------------------------------------------------------------------
    generar_reporte(resultados, demanda_media)
    md_path = os.path.join(REPORTS_DIR, "ljungbox.md")
    print(f"Reporte Markdown guardado: {md_path}")


# ---------------------------------------------------------------------------
# Generacion del reporte
# ---------------------------------------------------------------------------

def generar_reporte(resultados: dict, demanda_media: float):

    # Recopilar todos los p-values para el veredicto global
    todos_pvalues = [
        (config, lag, resultados[config][lag])
        for config in resultados
        for lag in LAGS
        if lag in resultados[config]
    ]
    hay_autocorrelacion = any(p < 0.05 for _, _, p in todos_pvalues)
    casos_significativos = [
        f"{config}, lag={lag} (p={p:.4f})"
        for config, lag, p in todos_pvalues
        if p < 0.05
    ]

    # ------------------------------------------------------------------
    # Tabla p-values
    # ------------------------------------------------------------------
    cab = "| Configuracion | lag=12 | lag=24 | lag=36 |"
    sep = "|---------------|--------|--------|--------|"
    filas_pv = [cab, sep]
    for config, vals in resultados.items():
        fila = f"| {config} |"
        for lag in LAGS:
            p = vals.get(lag)
            if p is None:
                fila += " — |"
            elif p < 0.05:
                fila += f" **{p:.4f}** |"
            else:
                fila += f" {p:.4f} |"
        filas_pv.append(fila)
    tabla_pv = "\n".join(filas_pv)

    # ------------------------------------------------------------------
    # Tabla estadisticas (solo modelos full)
    # ------------------------------------------------------------------
    cab2 = "| Configuracion | Desv. std (MWh) | Desv. std (% media) | N obs |"
    sep2 = "|---------------|-----------------|---------------------|-------|"
    filas_st = [cab2, sep2]
    for config in ("SARIMA_full", "SARIMAX_full"):
        v = resultados.get(config, {})
        filas_st.append(
            f"| {config} | {v.get('std_MWh', 0):,.0f} | "
            f"{v.get('std_pct', 0):.2f}% | {v.get('n', 0)} |"
        )
    tabla_st = "\n".join(filas_st)

    # ------------------------------------------------------------------
    # Veredicto y recomendacion
    # ------------------------------------------------------------------
    std_sarimax_pct = resultados.get("SARIMAX_full", {}).get("std_pct", 0)
    std_sarimax_mwh = resultados.get("SARIMAX_full", {}).get("std_MWh", 0)

    if not hay_autocorrelacion:
        veredicto = (
            "**Todos los p-values superan 0.05 en todas las configuraciones y folds.** "
            "Los residuos de ambos modelos se comportan como ruido blanco: no hay "
            "autocorrelacion residual estadisticamente significativa al nivel del 5%. "
            "Un modelo hibrido que prediga los residuos no puede explotar estructura inexistente."
        )
        recomendacion = (
            "**No procede implementar el enfoque hibrido SARIMAX+Chronos/TimesFM.** "
            "Los residuos son ruido blanco y la ganancia esperada sobre el MAPE es nula "
            "o marginal. Se recomienda cerrar el TFM con el SARIMAX como modelo final "
            "(MAPE 1.96% en hold-out 2024-2025) y dedicar el esfuerzo restante a la "
            "documentacion del pipeline y la redaccion de la memoria."
        )
    else:
        lista_casos = "\n".join(f"- {c}" for c in casos_significativos)
        veredicto = (
            f"**Al menos un p-value cae por debajo de 0.05.** Existe autocorrelacion "
            f"residual estadisticamente significativa: el SARIMAX no ha capturado toda "
            f"la estructura de la serie. Casos significativos:\n\n{lista_casos}\n\n"
            f"La desviacion estandar de los residuos del SARIMAX es "
            f"{std_sarimax_mwh:,.0f} MWh ({std_sarimax_pct:.2f}% de la demanda media). "
            f"Esa magnitud representa el techo teorico de mejora absoluta disponible para "
            f"el componente corrector del hibrido."
        )
        recomendacion = (
            f"**Procede avanzar al paso 2: implementar SARIMAX+Chronos / SARIMAX+TimesFM.** "
            f"Los residuos del SARIMAX contienen estructura autocorrelada que un foundation "
            f"model entrenado en series residuales puede aprender. La estrategia recomendada "
            f"es: (1) generar predicciones del SARIMAX para el hold-out 2024-2025, "
            f"(2) entrenar Chronos/TimesFM sobre los residuos in-sample del train completo "
            f"(fichero `reports/residuos_sarimax.csv`), (3) combinar ambas predicciones "
            f"y comparar MAPE con la linea base del 1.96%. La reduccion maxima alcanzable "
            f"equivale a capturar parte de los {std_sarimax_mwh:,.0f} MWh de desviacion "
            f"estandar residual."
        )

    # ------------------------------------------------------------------
    # Notas metodologicas por veredicto
    # ------------------------------------------------------------------
    if hay_autocorrelacion:
        nota_extra = (
            "- El fichero `reports/residuos_sarimax.csv` contiene los residuos del "
            "train completo listos para entrenar el componente corrector.\n"
            "- La columna `residuo_sarimax` es la entrada recomendada para Chronos/TimesFM."
        )
    else:
        nota_extra = (
            "- El fichero `reports/residuos_sarimax.csv` se genera de todos modos "
            "por si se desea una inspeccion visual adicional de los residuos."
        )

    reporte = f"""# Analisis de Autocorrelacion de Residuos SARIMA/SARIMAX

**TFM**: Prediccion de demanda electrica peninsular (REData, mensual)
**Fecha de analisis**: Mayo 2026
**Objetivo**: Determinar si los residuos del modelo SARIMAX ganador contienen
estructura autocorrelada aprovechable por un enfoque hibrido con foundation models
(Chronos / TimesFM).

---

## 1. Configuracion de modelos

| Parametro | Valor |
|-----------|-------|
| order (p,d,q) | (0, 1, 2) |
| seasonal_order (P,D,Q,m) | (1, 1, 1, 12) |
| Frecuencia | Mensual (MS) |
| Train pre-holdout | 2015-01 a 2023-12 (108 meses) |
| Hold-out (no usado en este analisis) | 2024-01 a 2025-12 (24 meses) |
| Variables exogenas SARIMAX | HDD18, CDD22 |
| MAPE hold-out SARIMAX (referencia) | 1.96% |

Los modelos se ajustan con `statsmodels.tsa.statespace.SARIMAX` sobre el train
pre-holdout completo y sobre cuatro folds de validacion cruzada walk-forward
(train terminando en 2019-12, 2020-12, 2021-12 y 2022-12) para verificar la
robustez del resultado.

---

## 2. Test de Ljung-Box (H0: ausencia de autocorrelacion)

Los residuos analizados son los errores de prediccion un paso adelante
(one-step-ahead) del filtro de Kalman (`model.resid` de statsmodels). El test
Q de Ljung-Box se aplica en los lags 12, 24 y 36, equivalentes a 1, 2 y 3
ciclos anuales completos.

**Negrita** = p < 0.05 (rechazo de H0; autocorrelacion significativa al 5%).

{tabla_pv}

---

## 3. Escala de los residuos

{tabla_st}

Demanda media mensual (train 2015-2023): {demanda_media:,.0f} MWh

---

## 4. Veredicto

{veredicto}

---

## 5. Recomendacion final

{recomendacion}

---

## Notas metodologicas

- Los residuos son errores one-step-ahead del filtro de Kalman, no errores de
  prediccion multi-paso. Son la magnitud correcta para diagnosticar si el modelo
  ha absorbido toda la dinamica lineal de la serie.
- El test de Ljung-Box se implementa con `acorr_ljungbox` de statsmodels con
  `return_df=True`. Los lags 12, 24 y 36 cubren hasta 3 ciclos estacionales.
- Los folds CV se usan exclusivamente para comprobar que el patron de
  autocorrelacion (o su ausencia) no es un artefacto del periodo concreto de
  entrenamiento.
- Un unico p-value < 0.05 es condicion suficiente para concluir que existe
  estructura residual explotable.
{nota_extra}
"""

    md_path = os.path.join(REPORTS_DIR, "ljungbox.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(reporte)


if __name__ == "__main__":
    main()
