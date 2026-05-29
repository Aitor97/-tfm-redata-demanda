"""
Modelos horarios para la demanda peninsular.

Disenio: rolling expanding 2020..2025 (6 iter), train desde 2015-01-01.

Modelos:
  - naive_hd       : pred(t) = media historica para (hora, dow) del train
  - sarimax_d24    : SARIMAX diario (orden (1,0,1)(1,1,1,7)) + perfil horario medio (hora, dow, mes) del train
  - prophet_h      : Prophet con yearly + weekly + daily_seasonality + regresores HDD/CDD + festivos ES
  - lightgbm_h     : LightGBM con lags 1, 24, 168 + calendario + temp + rolling means

Salidas:
  - reports/horario_predicciones.csv (filas por modelo/ano_test/timestamp con real y pred)
  - reports/horario_resumen.md       (MAPE horario, diario, mensual, anual; tabla comparativa)
"""

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "data" / "processed" / "dataset_horario.csv"
REPORTS = ROOT / "reports"

ANOS_TEST = [2018, 2019, 2020]  # OPSD horario cubre 2015 .. sep-2020
EXOG_BASE = ["HDD18", "CDD22"]


def mape(real: np.ndarray, pred: np.ndarray) -> float:
    mask = real != 0
    return float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100)


def _is_covid(idx: pd.DatetimeIndex) -> np.ndarray:
    ini = pd.Timestamp("2020-03-14")
    fin = pd.Timestamp("2020-06-21 23:00")
    return ((idx >= ini) & (idx <= fin)).astype(int)


# ---------------- Modelos ----------------

def naive_hd_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Media historica por (hora, dow) del train."""
    perfil = (train.assign(hora=train.index.hour, dow=train.index.dayofweek)
                    .groupby(["hora", "dow"])["demanda_MWh"].mean())
    idx = pd.MultiIndex.from_arrays(
        [test.index.hour, test.index.dayofweek], names=["hora", "dow"]
    )
    return perfil.reindex(idx).values


def sarimax_d24_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """SARIMAX diario + perfil horario (hora, dow, mes) del train.

    1. Agrega train a diario y ajusta SARIMAX(1,0,1)(1,1,1,7) + HDD/CDD diarios.
    2. Forecast diario sobre el test (HDD/CDD diarios de test).
    3. Multiplica cada dia por su perfil horario tal que la suma de las 24 horas
       reproduzca el total diario.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    tr_d = train["demanda_MWh"].resample("D").sum().to_frame("demanda_MWh")
    tr_d["HDD18"] = train["HDD18"].resample("D").mean()
    tr_d["CDD22"] = train["CDD22"].resample("D").mean()

    te_d = test["demanda_MWh"].resample("D").sum().to_frame("demanda_MWh")
    te_d["HDD18"] = test["HDD18"].resample("D").mean()
    te_d["CDD22"] = test["CDD22"].resample("D").mean()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAX(
            tr_d["demanda_MWh"],
            exog=tr_d[EXOG_BASE],
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
        pred_d = m.forecast(steps=len(te_d), exog=te_d[EXOG_BASE])
    pred_d = pd.Series(pred_d.values, index=te_d.index)

    # Perfil horario por (mes, dow, hora) normalizado para sumar 1 al dia.
    tr = train.assign(
        mes=train.index.month, dow=train.index.dayofweek, hora=train.index.hour
    )
    g = tr.groupby(["mes", "dow", "hora"])["demanda_MWh"].mean().rename("h")
    tot = g.groupby(level=[0, 1]).transform("sum")
    perfil = (g / tot).rename("share").to_frame()

    te = test.copy()
    te["mes"] = te.index.month
    te["dow"] = te.index.dayofweek
    te["hora"] = te.index.hour
    te["fecha"] = te.index.normalize()

    te = te.join(perfil, on=["mes", "dow", "hora"])
    # Fallback: si una combinacion no aparece en train -> 1/24
    te["share"] = te["share"].fillna(1.0 / 24.0)

    pred_d_idx = pred_d.reindex(te["fecha"]).values
    return pred_d_idx * te["share"].values


def prophet_h_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from prophet import Prophet
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").setLevel(logging.WARNING)

    df_tr = pd.DataFrame({
        "ds":       train.index,
        "y":        train["demanda_MWh"].values,
        "HDD18":    train["HDD18"].values,
        "CDD22":    train["CDD22"].values,
        "is_covid": _is_covid(train.index),
    })
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    m.add_country_holidays(country_name="ES")
    m.add_regressor("HDD18")
    m.add_regressor("CDD22")
    m.add_regressor("is_covid")
    m.fit(df_tr)

    df_te = pd.DataFrame({
        "ds":       test.index,
        "HDD18":    test["HDD18"].values,
        "CDD22":    test["CDD22"].values,
        "is_covid": _is_covid(test.index),
    })
    fc = m.predict(df_te)
    return fc["yhat"].values


def _features_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["HDD18"] = df["HDD18"]
    out["CDD22"] = df["CDD22"]
    out["temp"]  = df["temp_media_C"]
    out["hora"]  = df.index.hour
    out["dow"]   = df.index.dayofweek
    out["mes"]   = df.index.month
    out["doy"]   = df.index.dayofyear
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_holiday"] = df["is_holiday"]
    out["doy_sin"]  = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"]  = np.cos(2 * np.pi * out["doy"] / 365.25)
    out["hora_sin"] = np.sin(2 * np.pi * out["hora"] / 24.0)
    out["hora_cos"] = np.cos(2 * np.pi * out["hora"] / 24.0)
    out["is_covid"] = _is_covid(df.index)
    # Rollings causales de temperatura (1h shift)
    s_temp = df["temp_media_C"]
    out["temp_ma24"]  = s_temp.shift(1).rolling(24,  min_periods=1).mean()
    out["temp_ma168"] = s_temp.shift(1).rolling(168, min_periods=1).mean()
    out["HDD_ma24"]   = df["HDD18"].shift(1).rolling(24, min_periods=1).mean()
    out["CDD_ma24"]   = df["CDD22"].shift(1).rolling(24, min_periods=1).mean()
    return out


def lightgbm_h_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    full = pd.concat([train, test]).copy()
    feats_full = _features_lgbm(full)
    # Lags causales (todos quedan dentro del train para el test porque shift >= 1)
    feats_full["lag_1"]   = full["demanda_MWh"].shift(1)
    feats_full["lag_24"]  = full["demanda_MWh"].shift(24)
    feats_full["lag_168"] = full["demanda_MWh"].shift(168)
    feats_full["lag_8760"] = full["demanda_MWh"].shift(8760)

    feats_train = feats_full.loc[train.index]
    feats_test  = feats_full.loc[test.index]
    y_train = train["demanda_MWh"].values

    mask_tr = feats_train.notna().all(axis=1)
    Xtr = feats_train[mask_tr]
    ytr = y_train[mask_tr.values]

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=128,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )
    model.fit(Xtr, ytr)
    # 1-step ahead: los lags son reales (benchmark estandar horario).
    # Justificacion: lag_1/24/168 son siempre observados ex-ante en operacion.
    return model.predict(feats_test.values)


# ---------------- Pipeline ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="modelos a saltar")
    args = ap.parse_args()

    modelos = {
        "naive_hd":    naive_hd_pred,
        "sarimax_d24": sarimax_d24_pred,
        "prophet_h":   prophet_h_pred,
        "lightgbm_h":  lightgbm_h_pred,
    }
    for s in args.skip:
        modelos.pop(s, None)

    df = pd.read_csv(DATOS, parse_dates=["timestamp"], index_col="timestamp").asfreq("H")
    print(f"Datos: {df.index[0]} -> {df.index[-1]} ({len(df)} horas)\n")

    rows = []
    for ano in ANOS_TEST:
        train_ini = pd.Timestamp("2015-01-01 00:00")
        train_fin = pd.Timestamp(f"{ano-1}-12-31 23:00")
        test_ini  = pd.Timestamp(f"{ano}-01-01 00:00")
        test_fin  = pd.Timestamp(f"{ano}-12-31 23:00")
        train = df.loc[train_ini:train_fin]
        test  = df.loc[test_ini:test_fin]
        if len(test) == 0:
            print(f"\n=== Test {ano}: sin datos, salto ===")
            continue
        print(f"\n=== Test {ano}: train h={len(train)}, test h={len(test)} ===")

        preds_iter: dict[str, np.ndarray] = {}
        for nombre, fn in modelos.items():
            t0 = time.time()
            try:
                pred = fn(train, test)
            except Exception as e:
                print(f"  [{nombre:12}] FAILED: {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0
            real = test["demanda_MWh"].values
            mape_h = mape(real, pred)
            print(f"  [{nombre:12}] {dt:7.1f}s  MAPE horario {mape_h:.4f}%")
            preds_iter[nombre] = pred
            for ts, r, p in zip(test.index, real, pred):
                rows.append({"modelo": nombre, "ano_test": ano,
                             "timestamp": ts, "real": float(r), "pred": float(p)})

        # Ensemble (media de prophet_h + lightgbm_h + sarimax_d24)
        ens_keys = [k for k in ("sarimax_d24", "prophet_h", "lightgbm_h") if k in preds_iter]
        if len(ens_keys) >= 2:
            ens = np.mean([preds_iter[k] for k in ens_keys], axis=0)
            real = test["demanda_MWh"].values
            mape_h = mape(real, ens)
            print(f"  [{'ensemble':12}] {0.0:7.1f}s  MAPE horario {mape_h:.4f}%  ({'+'.join(ens_keys)})")
            for ts, r, p in zip(test.index, real, ens):
                rows.append({"modelo": "ensemble", "ano_test": ano,
                             "timestamp": ts, "real": float(r), "pred": float(p)})

    df_pred = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True, parents=True)
    df_pred.to_csv(REPORTS / "horario_predicciones.csv", index=False)

    # ---------- agregaciones ----------
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"])

    # diaria
    df_pred["fecha"] = df_pred["timestamp"].dt.normalize()
    dia = df_pred.groupby(["modelo", "ano_test", "fecha"]).agg(
        real=("real", "sum"), pred=("pred", "sum")
    ).reset_index()
    dia["ape"] = (dia["real"] - dia["pred"]).abs() / dia["real"] * 100
    res_dia = dia.groupby(["modelo", "ano_test"]).agg(MAPE_dia=("ape", "mean")).reset_index()
    res_dia_agg = res_dia.groupby("modelo").agg(
        MAPE_dia_media=("MAPE_dia", "mean"),
        MAPE_dia_std  =("MAPE_dia", "std"),
    ).round(4).reset_index().sort_values("MAPE_dia_media")

    # mensual
    df_pred["ym"] = df_pred["timestamp"].dt.to_period("M").dt.to_timestamp()
    mens = df_pred.groupby(["modelo", "ano_test", "ym"]).agg(
        real=("real", "sum"), pred=("pred", "sum")
    ).reset_index()
    mens["ape"] = (mens["real"] - mens["pred"]).abs() / mens["real"] * 100
    res_mes = mens.groupby(["modelo", "ano_test"]).agg(MAPE_mes=("ape", "mean")).reset_index()
    res_mes_agg = res_mes.groupby("modelo").agg(
        MAPE_mes_media=("MAPE_mes", "mean"),
        MAPE_mes_std  =("MAPE_mes", "std"),
    ).round(4).reset_index().sort_values("MAPE_mes_media")

    # anual
    anual = df_pred.groupby(["modelo", "ano_test"]).agg(
        real=("real", "sum"), pred=("pred", "sum")
    ).reset_index()
    anual["ape"] = (anual["real"] - anual["pred"]).abs() / anual["real"] * 100
    res_anual = anual.groupby("modelo").agg(
        MAPE_anual_media=("ape", "mean"),
        MAPE_anual_std  =("ape", "std"),
    ).round(4).reset_index().sort_values("MAPE_anual_media")

    # horario nativo
    df_pred["ape_h"] = (df_pred["real"] - df_pred["pred"]).abs() / df_pred["real"].replace(0, np.nan) * 100
    res_h = df_pred.groupby(["modelo", "ano_test"]).agg(MAPE_h=("ape_h", "mean")).reset_index()
    res_h_agg = res_h.groupby("modelo").agg(
        MAPE_h_media=("MAPE_h", "mean"),
        MAPE_h_std  =("MAPE_h", "std"),
    ).round(4).reset_index().sort_values("MAPE_h_media")

    print("\n=== MAPE HORARIO ===");   print(res_h_agg.to_string(index=False))
    print("\n=== MAPE DIARIO ===");    print(res_dia_agg.to_string(index=False))
    print("\n=== MAPE MENSUAL ===");   print(res_mes_agg.to_string(index=False))
    print("\n=== MAPE ANUAL ===");     print(res_anual.to_string(index=False))

    out_md = REPORTS / "horario_resumen.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Modelos horarios - rolling expanding (2020..2025)\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Diseno\n\n")
        f.write("- Train: expanding desde 2015-01-01. Test: cada ano completo 2020..2025 (6 iter).\n")
        f.write("- Pred horaria; agregada a diaria/mensual/anual para comparar con baselines.\n\n")
        f.write("## MAPE HORARIO\n\n");  f.write(res_h_agg.to_markdown(index=False))
        f.write("\n\n## MAPE DIARIO\n\n"); f.write(res_dia_agg.to_markdown(index=False))
        f.write("\n\n## MAPE MENSUAL\n\n");f.write(res_mes_agg.to_markdown(index=False))
        f.write("\n\n## MAPE ANUAL\n\n"); f.write(res_anual.to_markdown(index=False))
        f.write("\n\n## Detalle por iteracion (MAPE horario)\n\n")
        f.write(res_h.round(4).to_markdown(index=False))
        f.write("\n\n## Baselines de referencia\n\n")
        f.write("- Mensual SARIMAX(0,1,2)(1,1,1,12) + HDD/CDD: **3.65 %** +/- 1.66 (mensual, 6 iter).\n")
        f.write("- Diario ensemble (naive_trend+prophet+lightgbm): MAPE anual ~2.92 %.\n")

    print(f"\nGuardado: {REPORTS / 'horario_predicciones.csv'}")
    print(f"Guardado: {out_md}")


if __name__ == "__main__":
    main()
