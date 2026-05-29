# Modelos diarios con doble estacionalidad - rolling expanding

**Fecha**: 2026-05-18 23:42

## Diseno

- Train: expanding desde 2015. Test: cada ano completo 2020..2025 (6 iter).
- Pred diaria; agregada a mensual (sum) y anual (sum) para comparar con baseline.

## MAPE MENSUAL (media 12 meses por iteracion, luego media de las 6)

| modelo        |   MAPE_media |   MAPE_std |   MAPE_min |   MAPE_max |
|:--------------|-------------:|-----------:|-----------:|-----------:|
| ensemble      |       2.9564 |     1.7192 |     1.5988 |     6.1694 |
| prophet       |       3.366  |     1.3455 |     1.7244 |     5.6875 |
| lightgbm      |       3.7047 |     1.8729 |     1.6712 |     6.6886 |
| naive_semanal |       3.9671 |     1.569  |     2.2242 |     6.9167 |
| naive_trend   |       4.6212 |     1.4    |     2.9892 |     6.5088 |
| sarimax_d     |       5.499  |     1.6182 |     3.3842 |     7.8611 |
| tbats_proper  |       9.2336 |     3.7794 |     4.1168 |    14.7066 |

## MAPE ANUAL (sum de los 365 dias real vs pred, MAPE entre los 6 anos)

| modelo        |   MAPE_anual_media |   MAPE_anual_std |
|:--------------|-------------------:|-----------------:|
| ensemble      |             2.4237 |           1.8536 |
| naive_semanal |             2.7539 |           1.8755 |
| prophet       |             3.0588 |           1.4961 |
| lightgbm      |             3.2661 |           2.1076 |
| naive_trend   |             4.1057 |           1.6379 |
| sarimax_d     |             4.9689 |           1.7661 |
| tbats_proper  |             8.1806 |           5.2411 |

**Referencia baseline mensual SARIMAX -> anual**: 2.9168 % +/- 1.9673

## Detalle por iteracion (MAPE mensual)

| modelo        |   ano_test |   MAPE_mes |
|:--------------|-----------:|-----------:|
| ensemble      |       2020 |     6.1694 |
| ensemble      |       2021 |     2.0234 |
| ensemble      |       2022 |     3.6315 |
| ensemble      |       2023 |     2.2247 |
| ensemble      |       2024 |     1.5988 |
| ensemble      |       2025 |     2.0905 |
| lightgbm      |       2020 |     6.6886 |
| lightgbm      |       2021 |     2.8143 |
| lightgbm      |       2022 |     4.819  |
| lightgbm      |       2023 |     4.0597 |
| lightgbm      |       2024 |     2.1753 |
| lightgbm      |       2025 |     1.6712 |
| naive_semanal |       2020 |     6.9167 |
| naive_semanal |       2021 |     3.7109 |
| naive_semanal |       2022 |     3.557  |
| naive_semanal |       2023 |     4.0056 |
| naive_semanal |       2024 |     2.2242 |
| naive_semanal |       2025 |     3.3879 |
| naive_trend   |       2020 |     6.5088 |
| naive_trend   |       2021 |     5.2251 |
| naive_trend   |       2022 |     3.4632 |
| naive_trend   |       2023 |     5.7251 |
| naive_trend   |       2024 |     3.8155 |
| naive_trend   |       2025 |     2.9892 |
| prophet       |       2020 |     5.6875 |
| prophet       |       2021 |     3.6594 |
| prophet       |       2022 |     3.6395 |
| prophet       |       2023 |     2.6223 |
| prophet       |       2024 |     2.8628 |
| prophet       |       2025 |     1.7244 |
| sarimax_d     |       2020 |     6.3486 |
| sarimax_d     |       2021 |     4.7118 |
| sarimax_d     |       2022 |     4.4355 |
| sarimax_d     |       2023 |     7.8611 |
| sarimax_d     |       2024 |     3.3842 |
| sarimax_d     |       2025 |     6.2528 |
| tbats_proper  |       2020 |     8.5823 |
| tbats_proper  |       2021 |     7.6488 |
| tbats_proper  |       2022 |    14.7066 |
| tbats_proper  |       2023 |    12.4736 |
| tbats_proper  |       2024 |     4.1168 |
| tbats_proper  |       2025 |     7.8732 |

## Baseline mensual (referencia)

- SARIMAX(0,1,2)(1,1,1,12) + HDD/CDD, rolling expanding mensual: **3.65 %** +/- 1.66 (6 iter).
- SARIMAX(0,1,2)(1,1,1,12) + HDD/CDD agregado anual: **2.92 %** +/- 1.97 (6 iter).
- Hold-out fijo 2024-2025 mensual: 1.96 %.
