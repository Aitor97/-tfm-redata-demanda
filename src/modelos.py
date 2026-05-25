"""
Funciones de prediccion: SARIMA(X), ETS, Prophet.

Cada funcion ajusta sobre `y_train` y devuelve una pd.Series de predicciones
indexada por `y_test_index`.

Granularidad: datos diarios.
  - SARIMA/SARIMAX: estacionalidad semanal (m=7)
  - ETS: estacionalidad semanal (seasonal_periods=7)
  - Prophet: estacionalidad semanal + anual automatica
"""

import warnings
import logging
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Silenciar logs de Prophet/cmdstanpy que ensucian la salida
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# SARIMA / SARIMAX
# ---------------------------------------------------------------------------

def predecir_sarima(
    y_train: pd.Series,
    y_test_index: pd.Index,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    exog_train: pd.DataFrame | None = None,
    exog_test:  pd.DataFrame | None = None,
) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
        pred = modelo.forecast(steps=len(y_test_index), exog=exog_test)
    pred.index = y_test_index
    return pred


def buscar_orden_sarima(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None = None,
    p_range=(0, 1, 2),
    q_range=(0, 1, 2),
    P_range=(0, 1),
    Q_range=(0, 1),
    d=1, D=1, m=7,
) -> tuple[tuple, tuple, float]:
    """Selecciona (order, seasonal_order) por menor AIC en una grid pequena.

    m=7 captura la estacionalidad semanal de los datos diarios.
    """
    mejor = (None, None, np.inf)
    for p, q, P, Q in product(p_range, q_range, P_range, Q_range):
        order = (p, d, q)
        sorder = (P, D, Q, m)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=sorder,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            if fit.aic < mejor[2]:
                mejor = (order, sorder, fit.aic)
        except Exception:
            continue
    return mejor


# ---------------------------------------------------------------------------
# ETS / Holt-Winters
# ---------------------------------------------------------------------------

def predecir_ets(
    y_train: pd.Series,
    y_test_index: pd.Index,
    trend: str | None = "add",
    seasonal: str | None = "add",
    damped_trend: bool = False,
) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            y_train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=7,
            damped_trend=damped_trend,
            initialization_method="estimated",
        ).fit()
        pred = modelo.forecast(steps=len(y_test_index))
    pred.index = y_test_index
    return pred


def buscar_config_ets(y_train: pd.Series) -> tuple[dict, float]:
    """Selecciona (trend, seasonal, damped_trend) por menor AIC.

    seasonal_periods=7 captura la estacionalidad semanal de los datos diarios.
    """
    mejor_cfg, mejor_aic = None, np.inf
    for trend, seasonal, damped in product(["add", "mul", None], ["add", "mul"], [False, True]):
        if trend is None and damped:
            continue  # damped sin trend no aplica
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(
                    y_train,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=7,
                    damped_trend=damped,
                    initialization_method="estimated",
                ).fit()
            if fit.aic < mejor_aic:
                mejor_cfg = {"trend": trend, "seasonal": seasonal, "damped_trend": damped}
                mejor_aic = fit.aic
        except Exception:
            continue
    return mejor_cfg, mejor_aic


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def predecir_prophet(
    y_train: pd.Series,
    y_test_index: pd.Index,
    exog_train: pd.DataFrame | None = None,
    exog_test:  pd.DataFrame | None = None,
    festivos_es: bool = True,
    seasonality_mode: str = "multiplicative",
) -> pd.Series:
    from prophet import Prophet

    df_train = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
    if exog_train is not None:
        for col in exog_train.columns:
            df_train[col] = exog_train[col].values

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,   # activado para datos diarios (patron dia de semana)
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
    )
    if festivos_es:
        m.add_country_holidays(country_name="ES")
    if exog_train is not None:
        for col in exog_train.columns:
            m.add_regressor(col)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df_train)

    df_future = pd.DataFrame({"ds": y_test_index})
    if exog_test is not None:
        for col in exog_test.columns:
            df_future[col] = exog_test[col].values

    forecast = m.predict(df_future)
    return pd.Series(forecast["yhat"].values, index=y_test_index)
