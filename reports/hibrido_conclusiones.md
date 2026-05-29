# Conclusiones del enfoque híbrido SARIMAX + foundation models

**TFM**: Predicción de demanda eléctrica peninsular de España
**Pista**: mensual (132 obs, 2015–2025), hold-out fijo 2024–2025 (24 meses)
**Fecha**: 2026-05-18

---

## 1. Hipótesis de partida

Esquema clásico de descomposición (Zhang, 2003):

```
demanda = SARIMAX (parte estructural lineal) + residuos (parte no capturada)
            ↑ modelo base robusto                 ↑ corrector: Chronos / Lag-Llama
```

El **modelo base robusto** es `SARIMAX(0,1,2)(1,1,1,12) + HDD18/CDD22`
(MAPE hold-out = 1.96 %): parsimonioso, interpretable y captura el grueso
de la señal (tendencia + estacionalidad anual + temperatura vía grados-día).

La hipótesis: si los residuos del SARIMAX contienen estructura
autocorrelada, un corrector flexible aplicado **solo sobre los residuos**
(sin tocar la base) podría reducir el error.

## 2. Diagnóstico previo (Ljung-Box)

Test de Ljung-Box sobre los residuos one-step-ahead del SARIMAX
(`reports/ljungbox.md`):

| Lag | p-value (SARIMAX_full) | Interpretación |
|-----|------------------------|----------------|
| 12  | **0.0029**             | Autocorrelación significativa (1 ciclo anual) |
| 24  | 0.1563                 | No significativa |
| 36  | 0.6560                 | No significativa |

- Desv. típica residual: 2 544 190 MWh = **11.78 %** de la demanda media.
- Veredicto: existe autocorrelación residual estadísticamente
  significativa, pero **solo en lag 12 y de forma marginal**. Esto fue
  suficiente para justificar probar el paso 2 (el híbrido).

## 3. Implementación

Dos correctores zero-shot de última generación, mismo procedimiento:

1. Ajustar SARIMAX en train 2015–2023; predecir hold-out 2024–2025.
2. Usar la serie histórica de residuos in-sample como contexto.
3. Pedir al foundation model los 24 residuos del hold-out.
4. Híbrido = `pred_SARIMAX + pred_residuos`.

- **Chronos**: `amazon/chronos-t5-small`, 100 muestras, media.
- **Lag-Llama**: zero-shot, 100 muestras probabilísticas.

## 4. Resultado: ambos híbridos EMPEORAN el baseline

| Modelo | MAPE hold-out | Δ vs baseline |
|--------|---------------|---------------|
| SARIMAX puro (baseline) | **1.96 %** | — |
| SARIMAX + Chronos | 2.05 % | −0.09 pp (**−4.57 %**) |
| SARIMAX + Lag-Llama | 2.22 % | −0.26 pp (**−13.27 %**) |

El componente corrector **añade error neto** en lugar de quitarlo.

## 5. Por qué fracasó

1. **Significación estadística ≠ estructura predictible.** La
   autocorrelación de Ljung-Box es marginal, solo en lag 12, estimada
   sobre 108 puntos mensuales: señal débil detectable in-sample pero no
   extrapolable out-of-sample.
2. **Los residuos de un buen SARIMAX son casi ruido blanco.** Un
   foundation model zero-shot, sin conocimiento del dominio eléctrico
   español, no puede recuperar estructura donde predomina el ruido. En la
   tabla de predicciones de Chronos se observa que añade un *offset*
   positivo casi constante (~+0.4–0.6 %) a todos los meses: aprendió un
   sesgo plano, no la dinámica, y degrada justo donde el SARIMAX ya
   acertaba.
3. **El corrector aporta su propia varianza.** Predecir residuos ≈
   predecir ruido. Lag-Llama, más libre y probabilístico, lo hace peor
   (−13.27 %) que Chronos (−4.57 %).

## 6. Conclusión defendible

Se planteó una hipótesis razonable, se validó con un diagnóstico
estadístico (Ljung-Box, p = 0.0029 en lag 12), se implementaron dos
correctores zero-shot del estado del arte y **ninguno mejoró el
baseline**. Un resultado negativo bien ejecutado es contenido válido del
TFM:

> La autocorrelación residual del SARIMAX, aunque estadísticamente
> significativa, es demasiado débil y la muestra mensual demasiado corta
> (132 obs) para ser explotada por foundation models genéricos. El
> SARIMAX opera cerca del límite de lo aprovechable a escala mensual. Las
> mejoras reales no provienen de corregir residuos, sino de **aumentar la
> resolución temporal**: la pista diaria con ensemble alcanza MAPE anual
> 2.42 %, batiendo al baseline mensual (2.92 %).

Esto cierra el círculo del trabajo: justifica por qué la **pista diaria
es la línea ganadora** y por qué el híbrido mensual se descarta.

## 7. Limitación metodológica reconocida

El híbrido se evaluó únicamente en **hold-out fijo 2024–2025**, no en
validación rolling de 6 ventanas como las pistas mensual y diaria. Dado
que ya fracasa en hold-out, se reporta explícitamente como **prueba de
concepto**; extenderlo a rolling no se justifica (no cambiaría la
conclusión y consumiría recursos).
