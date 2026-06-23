# Diario del TFM

Registro de sesiones de trabajo. Cada entrada agrupa cambios, decisiones y resultados del día.

---

## 2026-05-08 — Híbridos zero-shot, exógenas calendario y validación rolling

### Lo que hice

1. **Pull + branch**: traída la rama `feature/hibrido-sarimax-chronos` desde el remoto, que incluía el script de análisis de residuos y el reporte Ljung-Box del paso anterior.

2. **Híbrido SARIMAX + Chronos** (`src/hibrido_sarimax_chronos.py`):
   - Tres parches de compatibilidad:
     - API actual de `ChronosPipeline.predict` (parámetro posicional `inputs`, retorna tensor único en vez de tupla).
     - `tabulate` añadido a las dependencias auto-instalables.
     - `PYTHONIOENCODING=utf-8` para evitar el `UnicodeEncodeError` de Windows con `→`.
   - Resultado: MAPE híbrido **2.05 %** vs SARIMAX puro **1.96 %** → **empeora 0.09 pp**.

3. **Híbrido SARIMAX + Lag-Llama** (`src/hibrido_sarimax_lagllama.py`):
   - Tres parches:
     - `freq="M"` con `pd.Period` en `ListDataset` (gluonts no acepta `MS`).
     - `weights_only=False` en `torch.load` (PyTorch 2.6+ por defecto rechaza checkpoints con clases custom).
     - Lectura de hiperparámetros del checkpoint (`hidden=144`) para construir el `LagLlamaEstimator` con la arquitectura correcta.
   - Resultado: MAPE híbrido **2.22 %** → **empeora 0.26 pp**.

4. **Diagnóstico del fracaso de los híbridos zero-shot**: los residuos del SARIMAX ya son ruido blanco (Ljung-Box pasa). Cualquier corrector zero-shot que prediga ≠ 0 sobre una serie sin señal residual añade error. El "fracaso" en realidad **valida** que el SARIMAX está bien especificado.

5. **SARIMAX + días laborables** (`src/sarimax_dias_laborables.py`):
   - Calculé `dias_laborables` por mes (L-V menos festivos nacionales ES vía librería `holidays`).
   - Resultado: AIC mejora ligeramente (2803.39 → 2801.35), pero **MAPE hold-out empeora de 1.96 % a 2.81 %** (+43 % peor).
   - Diagnóstico: redundancia con la estacionalidad mensual (S=12) ya capta ese patrón medio. Coef marginal (p=0.09) y sobreajuste fuera de muestra.

6. **Validación rolling-origin** (`src/validacion_rolling.py`):
   - 6 iteraciones expanding (2020..2025) + 6 iteraciones sliding (ventana 5 años).
   - **Hallazgo clave**: el MAPE medio sobre 6 ventanas es **3.65 %** (expanding) y **3.76 %** (sliding). El 1.96 % del hold-out fijo era una muestra afortunada (2024 dio 1.11 %).
   - Sliding tiene menor varianza (σ=1.08 vs σ=1.66) → descartar datos antiguos pre-COVID estabiliza el modelo.

### Tabla resumen del día

| Modelo / setup | MAPE | Notas |
|---|---|---|
| SARIMAX baseline (hold-out 2024-25) | 1.96 % | Punto de comparación inicial |
| SARIMAX baseline (rolling expanding, media 6 años) | 3.65 % ± 1.66 | Estimación más realista |
| SARIMAX baseline (rolling sliding 5 años, media 6 años) | 3.76 % ± 1.08 | Más estable |
| SARIMAX + Chronos (residuos, hold-out) | 2.05 % | Empeora 0.09 pp |
| SARIMAX + Lag-Llama (residuos, hold-out) | 2.22 % | Empeora 0.26 pp |
| SARIMAX + dias_laborables (hold-out) | 2.81 % | Empeora 0.85 pp |

### Decisiones / aprendizajes

- El hold-out fijo 2024-2025 **subestima el MAPE real del modelo**. Hay que reportar el rolling como métrica principal en el TFM.
- 2020 (COVID) y 2022 (crisis energética) son las iteraciones con mayor MAPE → cambios de régimen que ningún modelo puramente estadístico maneja bien.
- Los foundation models zero-shot (Chronos, Lag-Llama) **no aportan** sobre residuos blancos. Documentado y descartado.
- Añadir `dias_laborables` en niveles **redundante** con la estacionalidad SARIMAX. Si se prueba de nuevo, usar **anomalía** (desviación respecto a la media del mes-calendario) y dummy COVID.

### Próximos pasos candidatos

- M3/M4: SARIMAX con anomalía días laborables + dummy COVID + dummy crisis energética 2022.
- Probar IPI (Índice de Producción Industrial INE) como exógena.
- Comparativa con Prophet y ETS (ensemble simple).
- Foundation model con exógenas nativas: Moirai, TimesFM-2.5 o AutoGluon-TS con `covariate_regressor`.
- Decidir si la métrica del TFM es MAPE mensual rolling o MAPE anual (probablemente este último, que es el objetivo final del scope).

### Archivos generados hoy

- `src/hibrido_sarimax_chronos.py` (modificado)
- `src/hibrido_sarimax_lagllama.py` (modificado)
- `src/sarimax_dias_laborables.py` (nuevo)
- `src/validacion_rolling.py` (nuevo)
- `reports/hibrido_predicciones.csv` + `hibrido_resultado.md`
- `reports/hibrido_lagllama_predicciones.csv` + `hibrido_lagllama_resultado.md`
- `reports/sarimax_dias_laborables_predicciones.csv` + `sarimax_dias_laborables_resultado.md`
- `reports/rolling_predicciones.csv` + `rolling_resumen.md` + `rolling_mape.png`

### Commits

- `9e0b590` Adaptar híbridos Chronos y Lag-Llama al entorno actual.
- `8176697` Validación rolling, SARIMAX con días laborables y diario del día.

---

### Discusión posterior (sin código ejecutado)

Después del commit `8176697` la sesión continuó en modo exploratorio sin nuevos experimentos. Lo conversado:

**Diagnóstico de calidad de datos** del pipeline actual:
- Lo que **ya está bien**: demanda peninsular oficial REData, temperatura ponderada por población (10 ciudades, pesos INE 2024), HDD/CDD calculados a nivel diario y agregados al mes (no desde la T media mensual), fuente ERA5 vía Open-Meteo.
- Lo que **se puede mejorar**, ordenado por impacto:
  1. **Cambio de régimen por autoconsumo FV**: REE publica demanda en barras, no real. Post-2022 el autoconsumo solar (>20 GW instalados a finales 2024) hace que la serie subestime cada vez más la demanda real → cambio de régimen artificial que confunde al SARIMAX. Solución candidata: usar **IRE General** (índice REE ya corregido por laboralidad y temperatura) o sumar generación FV autoconsumo como exógena.
  2. **Frecuencia diaria** (en vez de mensual): pasar de 132 a ~3650 obs, doble estacionalidad (S=7, S=365.25). Modelos canónicos: Prophet, TBATS. Para el TFM (target anual) seguiríamos agregando, pero la literatura indica que predicción diaria → agregada suele batir a mensual directa.
  3. **Tratamiento formal de eventos** (intervención SARIMAX): pulso COVID (2020-03..2020-06), escalón crisis energética (≥2022-03), heatwaves específicas.
  4. **Bases HDD/CDD óptimas** vía grid search (ahora 18/22 por convención).
  5. **Más zonas climáticas** (añadir 4-5 ciudades a las 10 actuales).

**Más observaciones**: 132 meses son **pocas** para SARIMAX estacional (S=12 consume muchos grados de libertad). Tres caminos no excluyentes:
- (a) Extender hacia atrás → riesgo de contaminar con régimen pre-2010.
- (b) Subir frecuencia (diaria/horaria) → multiplicador limpio, mismo régimen.
- (c) Panel CCAA (15 × 132 = 1.980 obs) → buen segundo capítulo del TFM, especialmente con jerárquica + foundation models.

**Decisiones tomadas, no ejecutadas**:
- Acordado tirar por **(1) IRE General** y luego **(2) frecuencia diaria con Prophet/TBATS**.
- Empezada la Fase 1 (escribir `src/rolling_ire_general.py`): **rechazada por el usuario antes de escribirla** → "vamos a dejarlo así de momento".
- Reabierta la conversación: el usuario optó por **bajar a frecuencia diaria** y se intentó programar la tarea con `/schedule` para que la ejecutara un agente remoto.
- **`/schedule` falló reiteradamente** con error de conexión a claude.ai (varios reintentos a lo largo de la sesión). No fue posible delegar la tarea a un agente.
- Se aceptó ejecutarlo localmente en chat ("opción 3"): se creó **`src/descargar_diaria.py`** (descarga demanda peninsular diaria 2015-2025 desde REData) y antes de ejecutarlo el usuario interrumpió → "dejamos así".

**Estado al final del día**:
- `src/descargar_diaria.py` queda **huérfano** en working tree, no ejecutado, no commiteado, sin datos diarios descargados.
- Ningún experimento nuevo desde `8176697`.

### Próximos pasos al retomar (orden sugerido)

1. **Descargar demanda peninsular diaria** (ejecutar `src/descargar_diaria.py` o descartar).
2. **Adaptar `src/temperatura.py`** para guardar también la serie diaria peninsular (ahora solo guarda mensual; la función `temperatura_diaria_peninsular` ya devuelve diaria internamente).
3. **Construir `data/processed/dataset_diario.csv`** con demanda + HDD/CDD diarios + features de calendario (día semana, festivo nacional, mes, día año).
4. **Modelos diarios** con doble estacionalidad: Prophet (regresores HDD/CDD + festivos), TBATS (`seasonal_periods=[7, 365.25]`), SARIMAX(1,0,1)(1,1,1,7), naive estacional semanal como referencia.
5. **Agregar predicciones diarias** a mensual y anual; comparar MAPE rolling expanding contra el baseline mensual (3.65 %).
6. Alternativa o complemento: **IRE General** como target con SARIMA sin exógenas (la fase 1 que se quedó sin ejecutar).
7. **Panel CCAA** como tercer capítulo (jerárquica peninsular = suma de 15 CCAAs).

---

## 2026-05-10 — Pipeline diario y comparativa con baseline mensual

### Lo que hice

1. **Descarga de la serie diaria peninsular** (`src/descargar_diaria.py`): 4018 días 2015-01-01..2025-12-31 vía REData (`time_trunc=day`, `geo_limit=peninsular`). Salida `data/processed/demanda_peninsular_diaria.csv`.

2. **Temperatura peninsular diaria** (`src/temperatura.py`): añadida función `serie_diaria` y segundo CSV de salida `data/raw/temperatura_peninsular_diaria.csv` con T media + HDD18 + CDD22 a nivel diario, ponderada por población de las 10 ciudades existentes.

3. **`src/preprocesado_diario.py`** (nuevo): join demanda + temperatura + features de calendario (`dow`, `is_weekend`, `is_holiday` con `holidays.country_holidays("ES")`, `mes`, `doy`). Salida `data/processed/dataset_diario.csv` (4018 × 10 columnas).

4. **`src/modelos_diarios.py`** (nuevo): pipeline rolling expanding con 6 iteraciones (test = año completo 2020..2025). Modelos:
   - `naive_semanal`: pred(t) = real(t − 364), preserva DOW.
   - `naive_trend`: naive escalado por crecimiento del último trimestre.
   - `sarimax_d`: SARIMAX(1,0,1)(1,1,1,7) + HDD18 + CDD22.
   - `prophet`: Prophet con festivos ES nativos (`add_country_holidays`), regresores HDD/CDD y `is_covid` (lockdown 2020-03-14..2020-06-21).
   - `lightgbm`: LGBMRegressor con `lag_364`, HDD/CDD/T diarias, dow/mes/doy/sin/cos, `is_holiday`, `is_covid`, rolling-means causales (T7, T30, HDD7, CDD7).
   - `ensemble`: media de naive_trend + prophet + lightgbm.
   - `tbats_proper`: opcional vía `--tbats` (Box-Cox + ARMA errors, sólo últimos 3 años de train).
   Las predicciones diarias se agregan a mensual y anual para comparar con el baseline mensual (rolling expanding 3.65 %, agregado a anual 2.92 %).

### Hallazgos

| Modelo | MAPE mensual | MAPE anual |
|---|---|---|
| **ensemble (naive_trend + prophet + lightgbm)** | **2.96 %** ± 1.72 | **2.42 %** ± 1.85 |
| prophet (con COVID dummy + festivos ES) | 3.37 % ± 1.35 | 3.06 % ± 1.50 |
| lightgbm | 3.70 % ± 1.87 | 3.27 % ± 2.11 |
| naive_semanal (lag 364) | 3.97 % ± 1.57 | 2.75 % ± 1.88 |
| naive_trend | 4.62 % ± 1.40 | 4.11 % ± 1.64 |
| sarimax_d (1,0,1)(1,1,1,7) | 5.50 % ± 1.62 | 4.97 % ± 1.77 |
| tbats_proper [7, 365.25] + Box-Cox + ARMA (3 años train) | 9.23 % ± 3.78 | 8.18 % ± 5.24 |
| **baseline SARIMAX mensual** (referencia) | **3.65 % ± 1.66** | **2.92 % ± 1.97** |

- **El ensemble bate al baseline en ambas escalas**: −0.69 pp mensual (~19 % relativo) y −0.50 pp anual (~17 % relativo).
- **Aporte del COVID dummy**: Prophet pasa de 4.13 % → 3.37 % mensual; el ensemble de 3.20 % → 2.96 %. El estado de alarma marzo-junio 2020 explica la mayor parte del error en 2020.
- **2020 sigue siendo el año peor** del ensemble (6.17 % mensual). Sin contar 2020 el ensemble queda en 2.31 % mensual.
- **2024 es el mejor año** del ensemble (1.60 % mensual): año "normal" reciente, sin shocks, con autoconsumo FV ya estabilizado.
- **TBATS descartado definitivamente (ejecutado en rolling 6 ventanas)**. La versión rápida previa (sin Box-Cox/ARMA) dio ~30 %. La versión *proper* `--tbats` (doble estacionalidad [7, 365.25] + Box-Cox + ARMA errors) se corrió el 2026-05-18: **9.23 % mensual / 8.18 % anual**, con diferencia el peor modelo (vs ensemble 2.96 % / 2.42 %, vs sarimax_d 5.50 % / 4.97 %) y además el más lento (~100–220 s por ventana, ~13 min total frente a segundos del resto). Empeora en los años de cambio de régimen (2022: 14.71 %, 2023: 12.47 %). **No entra en el ensemble.** Caveat metodológico: TBATS entrena solo sobre los **últimos ~3 años** (`train.iloc[-365*3:]`), no expanding desde 2015 como el resto; dado el resultado catastrófico, alinear la ventana no cambiaría la conclusión (la familia TBATS queda cubierta conceptualmente y descartada empíricamente).

### Lecciones metodológicas

- **Trampa de lags cortos en LightGBM**: la versión inicial usaba `lag_7`/`lag_14` calculados sobre `concat(train, test)` y daba 2.06 % mensual — irreal porque el test estaba mirando los últimos 7-14 días verdaderos durante un forecast a 365 días. Eliminados, el modelo cae a 3.70 %, número honesto. Mantengo solo `lag_364`, que está siempre dentro del train por construcción del split.
- **Las features rolling de temperatura no tienen ese problema**: la T diaria es conocida ex-ante para el horizonte de forecast, así que `temp.shift(1).rolling(N).mean()` es lícito.
- **Compatibilidad con baseline mensual**: el SARIMAX mensual tiene MAPE anual agregado de 2.92 % (expanding) y 3.29 % (sliding). Necesario calcularlo aparte desde `reports/rolling_predicciones.csv` para que la comparación con los modelos diarios sea apples-to-apples.

### Decisiones / aprendizajes

- **Bajar a frecuencia diaria SÍ aporta** una vez se hace ensemble y se mete intervención COVID. La mejora es modesta pero consistente en todas las ventanas excepto 2020.
- **Naive semanal puro** es sorprendentemente competitivo (2.75 % anual vs 2.92 % del baseline). Para reportar en TFM, el naive es un baseline de referencia obligatorio.
- **Prophet brilla en años recientes y estables** (2025: 1.72 % mensual, 2024: 2.86 %), pero en COVID y crisis energética sin dummies se desboca. Con dummy mejora drásticamente.
- **LightGBM es el motor del ensemble**: aporta complementariedad al naive y prophet por usar features no-lineales (HDD/CDD interactuando con DOW y mes).

### Próximos pasos candidatos

- **Ensemble ponderado** por inverso del MAPE histórico de cada modelo (en lugar de media simple); puede arañar 0.1-0.3 pp.
- ~~TBATS proper~~ — hecho (2026-05-18): descartado, ver Hallazgos.
- **Tuning de LightGBM**: grid search sobre `num_leaves`, `learning_rate`, `n_estimators` con CV expanding interno.
- **Intervención COVID más rica**: en lugar de un dummy binario lockdown, modelar también el "post-lockdown" con efecto que decae (~3 meses) o usar variables de movilidad si hay disponibles.
- **Modelo jerárquico**: usar las 15 CCAA + reconciliación (MinT/OLS) para predicción peninsular como suma.
- **Hiperparámetro días/años de train sliding**: probar LightGBM con 5 años de sliding en lugar de expanding desde 2015 (igual ayuda a mitigar el cambio de régimen FV).

### Archivos generados / modificados

- `src/temperatura.py` (modificado: nueva función `serie_diaria`, segundo CSV de salida).
- `src/descargar_diaria.py` (nuevo).
- `src/preprocesado_diario.py` (nuevo).
- `src/modelos_diarios.py` (nuevo).
- `data/processed/demanda_peninsular_diaria.csv` (nuevo).
- `data/raw/temperatura_peninsular_diaria.csv` (nuevo).
- `data/processed/dataset_diario.csv` (nuevo).
- `reports/diario_predicciones.csv` (nuevo).
- `reports/diario_resumen.md` (nuevo).

---

## 2026-06-24 — Chronos-2 directo y ensembles 50/50 (mensual y diario)

### Lo que hice

1. **Cambio de variante Chronos**: descartado `amazon/chronos-t5-small` como apuesta principal. Tras revisar el estado del arte (Chronos-Bolt es ~250× más rápido y SOTA en GIFT-Eval / fev-bench; **Chronos-2** sale Oct-2025, encoder-only 120 M, soporta **known future covariates** nativamente), se elige **`amazon/chronos-2`**.

2. **Chronos-2 directo mensual** (`src/chronos2_directo.py`): predicción de demanda mensual usando `HDD18` y `CDD22` como future covariates conocidas (no como corrector de residuos SARIMAX, sino sustituyendo el modelo entero). Train 2015-01..2023-12, hold-out 2024-01..2025-12. API: `BaseChronosPipeline.from_pretrained("amazon/chronos-2") → predict_df(context_df, future_df, …)`, punto = cuantil 0.5.

3. **Ensemble SARIMAX + Chronos-2 mensual 50/50** (`src/ensemble_sarimax_chronos2.py`): promedio simple de las dos predicciones del hold-out. Versión preliminar con ponderación validada en 2023 (inverse-MAPE) — descartada porque 2023 fue patológico para SARIMAX y los pesos sobre-ponderaban Chronos-2; el oracle (w_S=0.45) coincide con 50/50, así que no hay margen tunear.

4. **Chronos-2 diario rolling expanding** (`src/chronos2_diario.py`): replicado el setup de `src/modelos_diarios.py` (6 ventanas, test = año completo 2020..2025). Future covariates: `HDD18`, `CDD22`, `is_holiday` (las que no son deducibles del timestamp). Sorpresa: **1-2 s por ventana en CPU** (esperaba 1-3 min); el modelo procesa 365 d de predicción con contextos de 1.8-3.6 k d casi instantáneamente.

5. **Ensemble diario 50/50** (`src/ensemble_diario_chronos2.py`): cruce de `chronos2_diario_predicciones.csv` con la fila `ensemble` de `reports/diario_predicciones.csv` (= mean naive_trend + Prophet + LightGBM). Promedio simple día a día, agregado a mensual y anual.

### Hallazgos

**Mensual (hold-out 2024-2025):**

| Modelo | MAPE |
|---|---|
| SARIMAX puro | 1.957 % |
| Chronos-2 directo | 1.874 % |
| **Ensemble SARIMAX + Chronos-2 50/50** | **1.363 %** |
| Oracle (w_S=0.45, upper bound) | 1.355 % |

Mejora vs SARIMAX puro: **−0.59 pp absoluta, −30 % relativa**. El 50/50 está a 0.008 pp del oracle, no hay margen de tuning. Los errores de SARIMAX y Chronos-2 cancelan especialmente bien en los meses de régimen post-autoconsumo FV (2025-04, 2025-05, 2025-08).

**Diario (rolling expanding 6 iter):**

| Modelo | MAPE diario | MAPE mensual | MAPE anual |
|---|---|---|---|
| Ensemble actual (naive_trend + Prophet + LightGBM) | — | 2.956 % ± 1.72 | 2.424 % ± 1.85 |
| Chronos-2 zero-shot + HDD/CDD/holidays | 3.681 % ± 1.68 | 2.859 % ± 1.98 | 2.330 % ± 2.12 |
| **Ensemble diario 50/50 (Chronos-2 + ensemble actual)** | — | **2.699 % ± 1.99** | **2.130 % ± 2.15** |

Mejora vs ensemble actual: **−0.26 pp mensual (−8.7 %)**, **−0.29 pp anual (−12.1 %)**. Gana 5/6 años; pierde solo 2022 (shock energético post-Ucrania). En 2021 el 50/50 da **MAPE anual de 0.25 %** — el mejor número del TFM.

### Decisiones / aprendizajes

- **Chronos-2 sustituye a Chronos-T5 como foundation model del pipeline.** Soporta future covariates de fábrica, así que ya no necesita el esquema "paso 1 SARIMAX, paso 2 corrector de residuos"; entra directo a competir con el modelo estructural.
- **El ensemble 50/50 es la receta ganadora en ambas pistas.** Cero tuning, cero leakage, defendible al tribunal en una línea. La validación interna (inverse-MAPE de 2023) fue peor por sobreponderar al ganador puntual de un año atípico; **lección metodológica**: con cambios de régimen, los pesos por validación de un solo año son frágiles.
- **El 1.36 % mensual del hold-out 2024-2025 NO es directamente comparable** con el 2.96 % mensual del ensemble diario (rolling expanding 6 ventanas). Para defenderlo en el TFM hay que llevarlo también a rolling mensual.
- **2022 sigue siendo el año patológico** para todos los modelos: shock energético, cambio de patrones de consumo industrial. Es el único año donde Chronos-2 pierde claramente; el SARIMAX/ensemble clásico aporta robustez ahí.
- **Coste computacional**: Chronos-2 diario en CPU corre en ~10 s para las 6 ventanas. Iterar es trivialmente barato — se pueden hacer experimentos con covariates alternativas (añadir `is_weekend`, `dow`, FV instalada acumulada, etc.) sin presupuesto de cómputo.

### Próximos pasos candidatos

- **Ensemble diario 4-vías**: `(naive_trend + Prophet + LightGBM + Chronos-2) / 4`, pesos iguales. Apuesto a que baja del 2.10 % anual.
- **Ensemble mensual rolling expanding 6 ventanas** (versión del 1.36 % comparable apples-to-apples con el resto).
- **Chronos-2 con más covariates futuros**: añadir `is_weekend`, `dow`, dummy de cambio de régimen FV post-2022.
- **Chronos-2 en horario** (`src/modelos_horarios.py` ya tiene la infra): contexto ~50 k h, horizonte ~8.7 k h. Verificar si Chronos-2 mantiene el tirón con doble estacionalidad diaria-semanal en niveles horarios.
- **Comparativa formal del TFM**: tabla maestra con baseline mensual / ensemble diario clásico / ensemble diario + Chronos-2 / ensemble mensual + Chronos-2, todos en rolling expanding.

### Archivos generados

- `src/chronos2_directo.py` (nuevo) — Chronos-2 mensual single hold-out.
- `src/ensemble_sarimax_chronos2.py` (nuevo) — ensemble 50/50 mensual.
- `src/chronos2_diario.py` (nuevo) — Chronos-2 diario rolling 6 ventanas.
- `src/ensemble_diario_chronos2.py` (nuevo) — ensemble 50/50 diario.
- `reports/chronos2_directo_{predicciones.csv,resultado.md}` (nuevos).
- `reports/ensemble_sarimax_chronos2_{predicciones.csv,resultado.md}` (nuevos).
- `reports/chronos2_diario_{predicciones.csv,resumen.md}` (nuevos).
- `reports/ensemble_diario_chronos2_{predicciones.csv,resumen_por_ano.csv,resultado.md}` (nuevos).
