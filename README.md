# MODELO_RNN_NYSE
Repositorio para los scripts de la primera versión del modelo de redes neuronales para la predicción de tendencias en el mercado de valores.

## Estructura de datos
Al ejecutar los scripts se crean carpetas locales bajo `data/` para almacenar cada fase del proceso:

- `data/raw/stocks`: descargas originales de Yahoo Finance.
- `data/raw/indices`: ficheros brutos de índices (por ejemplo, S&P 500).
- `data/processed/stocks`: datos con indicadores técnicos generados.
- `data/processed/with_target`: datos de acciones con la variable objetivo `Trend`.
- `data/processed/indices`: índices procesados con indicadores técnicos.
- `data/processed/with_index`: unión de acciones con el índice.
- `data/models`: artefactos del entrenamiento (modelo y reportes).

## Ejecución de la canalización
Se añadió `pipeline.py` para orquestar los scripts de forma secuencial. Ejecuta los pasos de preparación de datos con:

```bash
python pipeline.py
```

Por defecto se ejecutan los pasos de datos (`download-stocks`, `preprocess-index`, `build-features`, `add-target`, `merge-index`).
Para incluir el entrenamiento añade `--include-train`:

```bash
python pipeline.py --include-train
```

También puedes seleccionar pasos concretos:

```bash
python pipeline.py --steps build-features add-target
```

Cada script puede ejecutarse de manera independiente si se desea.
