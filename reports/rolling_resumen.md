# Validacion rolling-origin: SARIMAX(0,1,2)(1,1,1,12) + HDD18 + CDD22

**Fecha**: 2026-05-08 15:38

## Diseno

- Modelo: `SARIMAX(0, 1, 2)(1, 1, 1, 12)` con exogenas `['HDD18', 'CDD22']`.
- Horizonte por iteracion: 12 meses (1 ano).
- Anos de test: [2020, 2021, 2022, 2023, 2024, 2025] (6 iteraciones por modalidad).
- Expanding: train acumulativo desde 2015.
- Sliding  : train de 5 anos justo previos al test.

## Resumen agregado

| modalidad   |   MAPE_media |   MAPE_std |   MAPE_min |   MAPE_max |   MAE_media |   RMSE_media |
|:------------|-------------:|-----------:|-----------:|-----------:|------------:|-------------:|
| expanding   |       3.6549 |     1.6593 |     1.1092 |     5.3352 |      744337 |       899703 |
| sliding     |       3.7551 |     1.0767 |     2.6017 |     5.3352 |      767538 |       927473 |

## Detalle por iteracion

| modalidad   |   iter |   ano_test |   n_train |   MAPE_% |              MAE |             RMSE |
|:------------|-------:|-----------:|----------:|---------:|-----------------:|-----------------:|
| expanding   |      1 |       2020 |        60 |   5.3352 |      1.03057e+06 |      1.32576e+06 |
| expanding   |      2 |       2021 |        72 |   3.4919 | 725038           | 996047           |
| expanding   |      3 |       2022 |        84 |   4.9109 |      1.02637e+06 |      1.09063e+06 |
| expanding   |      4 |       2023 |        96 |   4.7311 | 964869           |      1.05022e+06 |
| expanding   |      5 |       2024 |       108 |   1.1092 | 230458           | 311937           |
| expanding   |      6 |       2025 |       120 |   2.3509 | 488722           | 623616           |
| sliding     |      1 |       2020 |        60 |   5.3352 |      1.03057e+06 |      1.32576e+06 |
| sliding     |      2 |       2021 |        60 |   4.626  | 961455           |      1.24827e+06 |
| sliding     |      3 |       2022 |        60 |   3.4369 | 724759           | 827492           |
| sliding     |      4 |       2023 |        60 |   3.827  | 782272           | 898101           |
| sliding     |      5 |       2024 |        60 |   2.7041 | 565019           | 625801           |
| sliding     |      6 |       2025 |        60 |   2.6017 | 541154           | 639410           |

## Comentarios

- 2020 sufre el shock COVID; 2022-2023 la crisis energetica. Ambos cambios de regimen son los mejores predictores de iteraciones con MAPE elevado.
- Diferencia expanding vs sliding revela si datos antiguos (2015-2019) ayudan o estorban a la prediccion del regimen actual.
- Comparacion con baseline hold-out fijo 2024-2025 (MAPE 1.957%) en la fila correspondiente.
