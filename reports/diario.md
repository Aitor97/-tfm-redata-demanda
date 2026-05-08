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
