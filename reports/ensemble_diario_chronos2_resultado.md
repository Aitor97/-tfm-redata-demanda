# Ensemble diario 50/50: Chronos-2 + ensemble actual

**Fecha**: 2026-06-24 00:13

## Configuracion

- A: ensemble actual = mean(naive_trend + Prophet + LightGBM)
- B: Chronos-2 zero-shot + HDD18/CDD22/is_holiday (future covariates)
- Combinacion: 0.5 A + 0.5 B (sin tuning)
- Rolling expanding 6 ventanas, anos test 2020..2025

## Resumen global (media 6 iter)

| Modelo | MAPE mensual | MAPE anual |
|---|---|---|
| Ensemble actual (A) | 2.9564% +/- 1.7192 | 2.4237% +/- 1.8536 |
| Chronos-2 (B) | 2.8594% +/- 1.9782 | 2.3298% +/- 2.1214 |
| **Ensemble 50/50 (FINAL)** | **2.6986% +/- 1.9870** | **2.1295% +/- 2.1519** |

**Mejora vs ensemble actual**: mensual +0.2577 pp (+8.72%), anual +0.2942 pp (+12.14%)

## Detalle por ano (MAPE)

|   ano |   MAPE_mes_ens_actual |   MAPE_mes_chronos2 |   MAPE_mes_50_50 |   MAPE_anual_ens_actual |   MAPE_anual_chronos2 |   MAPE_anual_50_50 |   MAPE_dia_50_50 |
|------:|----------------------:|--------------------:|-----------------:|------------------------:|----------------------:|-------------------:|-----------------:|
|  2020 |               6.16936 |            6.02005  |          6.07469 |                5.66055  |              5.47525  |           5.5679   |          6.55126 |
|  2021 |               2.02343 |            1.72338  |          1.13387 |                1.98844  |              1.48365  |           0.252395 |          1.9909  |
|  2022 |               3.63152 |            4.58659  |          4.09934 |                3.51573  |              4.50677  |           4.01125  |          4.35983 |
|  2023 |               2.22468 |            1.80398  |          1.73463 |                1.50985  |              1.37093  |           1.44039  |          2.85932 |
|  2024 |               1.59881 |            0.985156 |          1.08629 |                0.785117 |              0.727282 |           0.756199 |          2.39784 |
|  2025 |               2.09054 |            2.0371   |          2.06305 |                1.08243  |              0.415215 |           0.748822 |          2.86138 |
