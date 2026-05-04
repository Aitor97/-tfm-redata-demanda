# Analisis de Autocorrelacion de Residuos SARIMA/SARIMAX

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

| Configuracion | lag=12 | lag=24 | lag=36 |
|---------------|--------|--------|--------|
| SARIMA_full | **0.0030** | 0.1404 | 0.5085 |
| SARIMAX_full | **0.0029** | 0.1563 | 0.6560 |
| SARIMA_fold_2019 | 0.0522 | 0.5847 | 0.9373 |
| SARIMAX_fold_2019 | 0.0672 | 0.6603 | 0.9758 |
| SARIMA_fold_2020 | **0.0309** | 0.4741 | 0.8882 |
| SARIMAX_fold_2020 | **0.0331** | 0.5072 | 0.9395 |
| SARIMA_fold_2021 | **0.0166** | 0.3486 | 0.7763 |
| SARIMAX_fold_2021 | **0.0191** | 0.4016 | 0.8906 |
| SARIMA_fold_2022 | **0.0078** | 0.2426 | 0.6553 |
| SARIMAX_fold_2022 | **0.0090** | 0.2839 | 0.8093 |

---

## 3. Escala de los residuos

| Configuracion | Desv. std (MWh) | Desv. std (% media) | N obs |
|---------------|-----------------|---------------------|-------|
| SARIMA_full | 3,058,385 | 14.16% | 108 |
| SARIMAX_full | 2,544,190 | 11.78% | 108 |

Demanda media mensual (train 2015-2023): 21,598,368 MWh

---

## 4. Veredicto

**Al menos un p-value cae por debajo de 0.05.** Existe autocorrelacion residual estadisticamente significativa: el SARIMAX no ha capturado toda la estructura de la serie. Casos significativos:

- SARIMA_full, lag=12 (p=0.0030)
- SARIMAX_full, lag=12 (p=0.0029)
- SARIMA_fold_2020, lag=12 (p=0.0309)
- SARIMAX_fold_2020, lag=12 (p=0.0331)
- SARIMA_fold_2021, lag=12 (p=0.0166)
- SARIMAX_fold_2021, lag=12 (p=0.0191)
- SARIMA_fold_2022, lag=12 (p=0.0078)
- SARIMAX_fold_2022, lag=12 (p=0.0090)

La desviacion estandar de los residuos del SARIMAX es 2,544,190 MWh (11.78% de la demanda media). Esa magnitud representa el techo teorico de mejora absoluta disponible para el componente corrector del hibrido.

---

## 5. Recomendacion final

**Procede avanzar al paso 2: implementar SARIMAX+Chronos / SARIMAX+TimesFM.** Los residuos del SARIMAX contienen estructura autocorrelada que un foundation model entrenado en series residuales puede aprender. La estrategia recomendada es: (1) generar predicciones del SARIMAX para el hold-out 2024-2025, (2) entrenar Chronos/TimesFM sobre los residuos in-sample del train completo (fichero `reports/residuos_sarimax.csv`), (3) combinar ambas predicciones y comparar MAPE con la linea base del 1.96%. La reduccion maxima alcanzable equivale a capturar parte de los 2,544,190 MWh de desviacion estandar residual.

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
- El fichero `reports/residuos_sarimax.csv` contiene los residuos del train completo listos para entrenar el componente corrector.
- La columna `residuo_sarimax` es la entrada recomendada para Chronos/TimesFM.
