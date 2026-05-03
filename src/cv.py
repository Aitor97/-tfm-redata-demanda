"""
Walk-forward cross-validation y metricas de evaluacion.
"""

import numpy as np
import pandas as pd


def folds_walk_forward(df: pd.DataFrame, fold_starts: list, horizon: int = 12):
    """Genera (train, test, etiqueta) por cada fecha de inicio de test.

    Ventana expansiva: train = todo lo anterior; test = `horizon` meses desde
    `fold_start`. Si no hay suficientes meses para el test, se omite el fold.
    """
    for start in fold_starts:
        start = pd.Timestamp(start)
        end_train = start - pd.offsets.MonthBegin(1)
        end_test  = start + pd.offsets.MonthBegin(horizon - 1)
        train = df.loc[:end_train]
        test  = df.loc[start:end_test]
        if len(test) < horizon:
            continue
        yield train, test, str(start.year)


def metricas(y_true, y_pred) -> dict:
    """MAE, RMSE, MAPE a nivel mensual + error de la suma anual (en %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err    = y_true - y_pred
    mae    = float(np.abs(err).mean())
    rmse   = float(np.sqrt((err ** 2).mean()))
    mape   = float((np.abs(err / y_true)).mean() * 100)
    s_true, s_pred = float(y_true.sum()), float(y_pred.sum())
    return {
        "MAE_MWh":        mae,
        "RMSE_MWh":       rmse,
        "MAPE_%":         mape,
        "anual_real_TWh": s_true / 1e6,
        "anual_pred_TWh": s_pred / 1e6,
        "error_anual_%": (s_pred - s_true) / s_true * 100,
    }
