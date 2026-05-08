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
- (pendiente) Validación rolling y SARIMAX con días laborables.
