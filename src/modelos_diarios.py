"""
Modelos diarios con doble estacionalidad y agregacion a mensual / anual.

Compara contra el baseline mensual SARIMAX:
  - rolling expanding mensual : MAPE 3.65 % +/- 1.66 (mensual, 6 iter)
  - rolling expanding anual   : MAPE 2.92 % +/- 1.97 (anual,   6 iter)

Modelos diarios:
  - naive_semanal : pred(t) = real(t - 364)
  - naive_trend   : naive_semanal ajustado por crecimiento medio del ultimo trimestre
  - sarimax_d     : SARIMAX(1,0,1)(1,1,1,7) + HDD18 + CDD22
  - prophet       : Prophet + regresores HDD/CDD + festivos ES nativos
  - lightgbm      : LightGBM con lags + HDD/CDD + calendario
  - ensemble      : media (naive_trend + prophet + lightgbm)
  - tbats_proper  : TBATS [7,365.25] con Box-Cox y ARMA errors (slow, opcional via flag --tbats)

Diseno: 6 iteraciones expanding (2020..2025), horizonte = 1 ano.

Salidas:
  - reports/diario_predicciones.csv
  - reports/diario_resumen.md
"""

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "data" / "processed" / "dataset_diario.csv"
REPORTS = ROOT / "reports"

ANOS_TEST = [2020, 2021, 2022, 2023, 2024, 2025]
EXOG_BASE = ["HDD18", "CDD22"]


def mape(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((real - pred) / real)) * 100)


# ---------- Modelos ----------

def naive_semanal_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    full = pd.concat([train, test])
    pred = full["demanda_MWh"].shift(364)
    return pred.loc[test.index].values


def naive_trend_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """naive_semanal multiplicado por el factor de crecimiento medio
    del ultimo trimestre vs el mismo trimestre del ano anterior.
    """
    full = pd.concat([train, test])
    base = full["demanda_MWh"].shift(364)

    last90 = train["demanda_MWh"].iloc[-90:]
    prev90 = train["demanda_MWh"].iloc[-90 - 364:-364] if len(train) > 454 else last90
    factor = last90.sum() / prev90.sum() if prev90.sum() > 0 else 1.0
    return (base.loc[test.index] * factor).values


def sarimax_d_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAX(
            train["demanda_MWh"],
            exog=train[EXOG_BASE],
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
        pred = m.forecast(steps=len(test), exog=test[EXOG_BASE])
    return np.asarray(pred)


def prophet_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from prophet import Prophet
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").setLevel(logging.WARNING)

    df_tr = pd.DataFrame({
        "ds": train.index,
        "y":  train["demanda_MWh"].values,
        "HDD18": train["HDD18"].values,
        "CDD22": train["CDD22"].values,
        "is_covid": _is_covid(train.index),
    })
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.add_country_holidays(country_name="ES")
    m.add_regressor("HDD18")
    m.add_regressor("CDD22")
    m.add_regressor("is_covid")
    m.fit(df_tr)

    df_te = pd.DataFrame({
        "ds": test.index,
        "HDD18": test["HDD18"].values,
        "CDD22": test["CDD22"].values,
        "is_covid": _is_covid(test.index),
    })
    fc = m.predict(df_te)
    return fc["yhat"].values


def _is_covid(idx: pd.DatetimeIndex) -> np.ndarray:
    """Confinamiento estricto en Espana: 14-mar-2020 .. 21-jun-2020 (estado de alarma)."""
    ini = pd.Timestamp("2020-03-14")
    fin = pd.Timestamp("2020-06-21")
    return ((idx >= ini) & (idx <= fin)).astype(int)


def _features_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["HDD18"] = df["HDD18"]
    out["CDD22"] = df["CDD22"]
    out["temp"]  = df["temp_media_C"]
    out["dow"]   = df["dow"]
    out["mes"]   = df["mes"]
    out["doy"]   = df["doy"]
    out["is_weekend"] = df["is_weekend"]
    out["is_holiday"] = df["is_holiday"]
    out["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    out["is_covid"] = _is_covid(df.index)
    # Rolling means de temperatura (causales: hasta el dia anterior)
    s_temp = df["temp_media_C"]
    out["temp_ma7"]  = s_temp.shift(1).rolling(7,  min_periods=1).mean()
    out["temp_ma30"] = s_temp.shift(1).rolling(30, min_periods=1).mean()
    out["HDD_ma7"]   = df["HDD18"].shift(1).rolling(7, min_periods=1).mean()
    out["CDD_ma7"]   = df["CDD22"].shift(1).rolling(7, min_periods=1).mean()
    return out


def lightgbm_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    # Features sobre el full (train+test): los rolling de T y HDD/CDD usan obs.
    # de temperatura (conocida ex-ante). lag_364 esta siempre dentro del train.
    full = pd.concat([train, test]).copy()
    feats_full = _features_lgbm(full)
    feats_full["lag_364"] = full["demanda_MWh"].shift(364)

    feats_train = feats_full.loc[train.index]
    feats_test  = feats_full.loc[test.index]
    y_train = train["demanda_MWh"].values

    mask_tr = feats_train.notna().all(axis=1)
    Xtr = feats_train[mask_tr]
    ytr = y_train[mask_tr.values]

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(feats_test)
    return pred


def tbats_proper_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from tbats import TBATS
    train_recent = train.iloc[-365 * 3:]
    e = TBATS(
        seasonal_periods=[7, 365.25],
        use_box_cox=True,
        use_arma_errors=True,
        n_jobs=1,
    )
    m = e.fit(train_recent["demanda_MWh"].values)
    return m.forecast(steps=len(test))


# ---------- Pipeline ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbats", action="store_true", help="incluir TBATS proper (lento)")
    args = ap.parse_args()

    modelos = {
        "naive_semanal": naive_semanal_pred,
        "naive_trend":   naive_trend_pred,
        "sarimax_d":     sarimax_d_pred,
        "prophet":       prophet_pred,
        "lightgbm":      lightgbm_pred,
    }
    if args.tbats:
        modelos["tbats_proper"] = tbats_proper_pred

    df = pd.read_csv(DATOS, parse_dates=["fecha"], index_col="fecha").asfreq("D")
    print(f"Datos: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} dias)\n")

    rows = []
    for ano in ANOS_TEST:
        train_ini = pd.Timestamp("2015-01-01")
        train_fin = pd.Timestamp(f"{ano-1}-12-31")
        test_ini  = pd.Timestamp(f"{ano}-01-01")
        test_fin  = pd.Timestamp(f"{ano}-12-31")
        train = df.loc[train_ini:train_fin]
        test  = df.loc[test_ini:test_fin]
        print(f"\n=== Test {ano}: train n={len(train)}, test n={len(test)} ===")

        preds_iter: dict[str, np.ndarray] = {}
        for nombre, fn in modelos.items():
            t0 = time.time()
            try:
                pred = fn(train, test)
            except Exception as e:
                print(f"  [{nombre:14}] FAILED: {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0
            real = test["demanda_MWh"].values
            mape_d = mape(real, pred)
            print(f"  [{nombre:14}] {dt:6.1f}s  MAPE diario {mape_d:.4f}%")
            preds_iter[nombre] = pred
            for f, r, p in zip(test.index, real, pred):
                rows.append({"modelo": nombre, "ano_test": ano,
                             "fecha": f, "real": float(r), "pred": float(p)})

        # Ensemble (media de naive_trend + prophet + lightgbm si los 3 estan)
        ens_keys = [k for k in ("naive_trend", "prophet", "lightgbm") if k in preds_iter]
        if len(ens_keys) >= 2:
            ens = np.mean([preds_iter[k] for k in ens_keys], axis=0)
            real = test["demanda_MWh"].values
            mape_d = mape(real, ens)
            print(f"  [{'ensemble':14}] {0.0:6.1f}s  MAPE diario {mape_d:.4f}%  ({'+'.join(ens_keys)})")
            for f, r, p in zip(test.index, real, ens):
                rows.append({"modelo": "ensemble", "ano_test": ano,
                             "fecha": f, "real": float(r), "pred": float(p)})

    df_pred = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True, parents=True)
    df_pred.to_csv(REPORTS / "diario_predicciones.csv", index=False)

    # --- agregaciones ---
    df_pred["fecha"] = pd.to_datetime(df_pred["fecha"])

    # mensual
    df_pred["ym"] = df_pred["fecha"].dt.to_period("M").dt.to_timestamp()
    mens = df_pred.groupby(["modelo", "ano_test", "ym"]).agg(
        real=("real", "sum"),
        pred=("pred", "sum"),
    ).reset_index()
    mens["ape"] = (mens["real"] - mens["pred"]).abs() / mens["real"] * 100
    res_iter_m = mens.groupby(["modelo", "ano_test"]).agg(MAPE_mes=("ape", "mean")).reset_index()
    res_agg_m  = res_iter_m.groupby("modelo").agg(
        MAPE_media=("MAPE_mes", "mean"),
        MAPE_std  =("MAPE_mes", "std"),
        MAPE_min  =("MAPE_mes", "min"),
        MAPE_max  =("MAPE_mes", "max"),
    ).round(4).reset_index().sort_values("MAPE_media")

    # anual
    anual = df_pred.groupby(["modelo", "ano_test"]).agg(
        real=("real", "sum"),
        pred=("pred", "sum"),
    ).reset_index()
    anual["ape"] = (anual["real"] - anual["pred"]).abs() / anual["real"] * 100
    res_anual = anual.groupby("modelo").agg(
        MAPE_anual_media=("ape", "mean"),
        MAPE_anual_std  =("ape", "std"),
    ).round(4).reset_index().sort_values("MAPE_anual_media")

    print("\n=== MAPE MENSUAL (agregando pred diaria) ===")
    print(res_agg_m.to_string(index=False))
    print("\n=== MAPE ANUAL ===")
    print(res_anual.to_string(index=False))

    # --- baseline mensual SARIMAX agregado a anual desde rolling_predicciones.csv ---
    base_csv = REPORTS / "rolling_predicciones.csv"
    if base_csv.exists():
        bdf = pd.read_csv(base_csv, parse_dates=["fecha"])
        bdf = bdf[bdf["modalidad"] == "expanding"].copy()
        bdf["ano"] = bdf["fecha"].dt.year
        ban = bdf.groupby("ano").agg(real=("real", "sum"), pred=("pred", "sum")).reset_index()
        ban["ape"] = (ban["real"] - ban["pred"]).abs() / ban["real"] * 100
        base_anual_media = ban["ape"].mean()
        base_anual_std   = ban["ape"].std()
        print(f"\nBaseline mensual SARIMAX -> anual: {base_anual_media:.4f}% +/- {base_anual_std:.4f}")
    else:
        base_anual_media = base_anual_std = None

    out_md = REPORTS / "diario_resumen.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Modelos diarios con doble estacionalidad - rolling expanding\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Diseno\n\n")
        f.write("- Train: expanding desde 2015. Test: cada ano completo 2020..2025 (6 iter).\n")
        f.write("- Pred diaria; agregada a mensual (sum) y anual (sum) para comparar con baseline.\n\n")
        f.write("## MAPE MENSUAL (media 12 meses por iteracion, luego media de las 6)\n\n")
        f.write(res_agg_m.to_markdown(index=False))
        f.write("\n\n## MAPE ANUAL (sum de los 365 dias real vs pred, MAPE entre los 6 anos)\n\n")
        f.write(res_anual.to_markdown(index=False))
        if base_anual_media is not None:
            f.write(f"\n\n**Referencia baseline mensual SARIMAX -> anual**: "
                    f"{base_anual_media:.4f} % +/- {base_anual_std:.4f}\n")
        f.write("\n## Detalle por iteracion (MAPE mensual)\n\n")
        f.write(res_iter_m.round(4).to_markdown(index=False))
        f.write("\n\n## Baseline mensual (referencia)\n\n")
        f.write("- SARIMAX(0,1,2)(1,1,1,12) + HDD/CDD, rolling expanding mensual: **3.65 %** +/- 1.66 (6 iter).\n")
        if base_anual_media is not None:
            f.write(f"- SARIMAX(0,1,2)(1,1,1,12) + HDD/CDD agregado anual: **{base_anual_media:.2f} %** "
                    f"+/- {base_anual_std:.2f} (6 iter).\n")
        f.write("- Hold-out fijo 2024-2025 mensual: 1.96 %.\n")

    print(f"\nGuardado: {REPORTS / 'diario_predicciones.csv'}")
    print(f"Guardado: {out_md}")


if __name__ == "__main__":
    main()
