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
| **baseline SARIMAX mensual** (referencia) | **3.65 % ± 1.66** | **2.92 % ± 1.97** |

- **El ensemble bate al baseline en ambas escalas**: −0.69 pp mensual (~19 % relativo) y −0.50 pp anual (~17 % relativo).
- **Aporte del COVID dummy**: Prophet pasa de 4.13 % → 3.37 % mensual; el ensemble de 3.20 % → 2.96 %. El estado de alarma marzo-junio 2020 explica la mayor parte del error en 2020.
- **2020 sigue siendo el año peor** del ensemble (6.17 % mensual). Sin contar 2020 el ensemble queda en 2.31 % mensual.
- **2024 es el mejor año** del ensemble (1.60 % mensual): año "normal" reciente, sin shocks, con autoconsumo FV ya estabilizado.
- **TBATS rápido (sin Box-Cox, sin ARMA, en una iteración previa) dio MAPE catastrófico ~30 %**. Excluido del ensemble; la versión `--tbats` opcional con Box-Cox + ARMA queda como follow-up.

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
- **TBATS proper** (`python src/modelos_diarios.py --tbats`): correr con Box-Cox + ARMA errors y comparar.
- **Tuning de LightGBM**: grid search sobre `num_leaves`, `learning_rate`, `n_estimators` con CV expanding interno.
- **Intervención COVID más rica**: en lugar de un dummy binario lockdown, modelar también el "post-lockdown" con efecto que decae (~3 meses) o usar variables de movilidad si hay disponibles.
- **Modelo jerárquico**: usar las 15 CCAA + reconciliación (MinT/OLS) para predicción peninsular como suma.
- **Hiperparámetro días/años de train sliding**: probar TBATS y LightGBM con 5 años de sliding en lugar de expanding desde 2015 (igual ayuda a mitigar el cambio de régimen FV).

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
