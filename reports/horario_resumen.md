# Modelos horarios - rolling expanding (2020..2025)

**Fecha**: 2026-05-29 12:39

## Diseno

- Train: expanding desde 2015-01-01. Test: cada ano completo 2020..2025 (6 iter).
- Pred horaria; agregada a diaria/mensual/anual para comparar con baselines.

## MAPE HORARIO

| modelo      |   MAPE_h_media |   MAPE_h_std |
|:------------|---------------:|-------------:|
| lightgbm_h  |         0.8648 |       0.3567 |
| ensemble    |         3.7487 |       0.9687 |
| prophet_h   |         6.35   |       0.6344 |
| sarimax_d24 |         6.9369 |       1.4864 |
| naive_hd    |         7.5384 |       2.7958 |

## MAPE DIARIO

| modelo      |   MAPE_dia_media |   MAPE_dia_std |
|:------------|-----------------:|---------------:|
| lightgbm_h  |           0.4851 |         0.257  |
| ensemble    |           3.3993 |         1.1009 |
| prophet_h   |           4.9174 |         0.7988 |
| sarimax_d24 |           6.7826 |         1.4762 |
| naive_hd    |           7.2252 |         2.6581 |

## MAPE MENSUAL

| modelo      |   MAPE_mes_media |   MAPE_mes_std |
|:------------|-----------------:|---------------:|
| lightgbm_h  |           0.3213 |         0.3669 |
| ensemble    |           2.7518 |         1.4843 |
| prophet_h   |           4.0471 |         1.1352 |
| sarimax_d24 |           5.6931 |         1.8533 |
| naive_hd    |           5.7775 |         2.9935 |

## MAPE ANUAL

| modelo      |   MAPE_anual_media |   MAPE_anual_std |
|:------------|-------------------:|-----------------:|
| lightgbm_h  |             0.2294 |           0.3001 |
| ensemble    |             1.4269 |           2.0993 |
| naive_hd    |             2.889  |           3.1037 |
| prophet_h   |             3.6951 |           1.2924 |
| sarimax_d24 |             4.4244 |           1.7983 |

## Detalle por iteracion (MAPE horario)

| modelo      |   ano_test |   MAPE_h |
|:------------|-----------:|---------:|
| ensemble    |       2018 |   2.6312 |
| ensemble    |       2019 |   4.2651 |
| ensemble    |       2020 |   4.3498 |
| lightgbm_h  |       2018 |   0.6774 |
| lightgbm_h  |       2019 |   0.6407 |
| lightgbm_h  |       2020 |   1.2761 |
| naive_hd    |       2018 |   5.8789 |
| naive_hd    |       2019 |   5.97   |
| naive_hd    |       2020 |  10.7663 |
| prophet_h   |       2018 |   5.6654 |
| prophet_h   |       2019 |   6.4669 |
| prophet_h   |       2020 |   6.9178 |
| sarimax_d24 |       2018 |   5.2524 |
| sarimax_d24 |       2019 |   7.4942 |
| sarimax_d24 |       2020 |   8.0641 |

## Baselines de referencia

- Mensual SARIMAX(0,1,2)(1,1,1,12) + HDD/CDD: **3.65 %** +/- 1.66 (mensual, 6 iter).
- Diario ensemble (naive_trend+prophet+lightgbm): MAPE anual ~2.92 %.
