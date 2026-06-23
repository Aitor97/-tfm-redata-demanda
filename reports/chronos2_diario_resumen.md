# Chronos-2 diario (rolling expanding 6 ventanas)

**Fecha**: 2026-06-24 00:08

## Configuracion

- Modelo: amazon/chronos-2 zero-shot (120 M params)
- Train expanding desde 2015-01-01; test cada ano 2020..2025
- Future covariates: HDD18, CDD22, is_holiday
- Punto: cuantil 0.5 (mediana)

## Resumen global (media 6 iter)

| Metric | MAPE | std |
|---|---|---|
| Diario | 3.6813% | 1.6802 |
| Mensual | 2.8594% | 1.9782 |
| Anual | 2.3298% | 2.1214 |

## Detalle por iteracion

|   ano_test |   MAPE_diario |   MAPE_mensual |   MAPE_anual |
|-----------:|--------------:|---------------:|-------------:|
|       2020 |       6.54027 |       6.02005  |     5.47526  |
|       2021 |       2.29244 |       1.72338  |     1.48366  |
|       2022 |       4.85245 |       4.58659  |     4.50676  |
|       2023 |       2.99302 |       1.80398  |     1.37093  |
|       2024 |       2.3444  |       0.985157 |     0.727287 |
|       2025 |       3.06534 |       2.0371   |     0.415213 |

## Comparacion con ensemble diario actual

Referencia (memoria, 6 iter):

- Ensemble (naive_trend + Prophet + LightGBM): MAPE mensual 2.9564 % +/- 1.72, anual 2.4237 % +/- 1.85
- Baseline mensual SARIMAX -> anual: 2.9168 % +/- 1.97
