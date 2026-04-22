# Crypto AI (Binance) — Predicción Probabilística + Logging + Evaluación

Este proyecto construye un pipeline completo para:
1) **Descargar datos OHLCV (velas) de Binance**.
2) **Entrenar un modelo** que devuelve una probabilidad `p_up` = P(subida en H minutos).
3) Ejecutar un **logger autónomo** que registra `p_up` por cada vela cerrada en un CSV.
4) Ejecutar un **evaluador** que compara el CSV de predicciones con velas reales y mide aciertos/métricas.

**Importante**: en esta fase NO hay bot de trading (no se envían órdenes). Primero se valida que:
- el sistema genera probabilidades de forma autónoma,
- el log es consistente,
- la evaluación funciona y cuantifica el rendimiento real.

---

## Estructura del proyecto
C:\crypto_ai
    README.md
    src
        01_download_klines.py
        02_build_dataset.py
        03_train_model.py
        04_predict_latest.py
        07_log_probabilities.py
        08_evaluate_predictions.py
    data
        BTCUSDT_1m_raw.parquet
        BTCUSDT_1m_dataset_h5_thr0.002.parquet
    models
        model_BTCUSDT_1m_h5_thr0.002.joblib
    logs
        preds_BTCUSDT_1m.csv
        preds_eval_BTCUSDT_1m.csv

### `src/`
Contiene todos los scripts del pipeline (descarga, dataset, entrenamiento, predicción, logging, evaluación).

### `data/`
Datos generados:
- `*_raw.parquet`: histórico de velas descargado de Binance.
- `*_dataset_*.parquet`: dataset con features + labels para entrenamiento.

### `models/`
Modelos entrenados exportados en `.joblib`, incluyendo:
- modelo base (XGBoost),
- calibrador de probabilidad (Platt scaling con LogisticRegression),
- lista exacta de columnas/features usadas (para inferencia consistente),
- configuración (símbolo, intervalo, horizonte, umbral).

### `logs/`
Logs operativos:
- `preds_*.csv`: predicciones `p_up` por vela cerrada (logger autónomo).
- `preds_eval_*.csv`: predicciones enriquecidas con verdad terreno (close futuro, retorno futuro, etiquetas reales y predicción).

---

## Requisitos / entorno

Se recomienda usar Miniconda y un entorno con Python 3.11:

```bat
conda create -n cryptoai python=3.11 -y
conda activate cryptoai
pip install pandas numpy pyarrow scikit-learn xgboost joblib matplotlib python-dotenv requests websockets


```markdown
# Crypto AI (Binance) — Proyecto completo (pipeline + logging autónomo + evaluación)

Este repositorio/proyecto monta un sistema **end-to-end** para:
1) Descargar velas (OHLCV) desde Binance (API pública).
2) Construir un dataset de entrenamiento con features.
3) Entrenar un modelo que produce una probabilidad `p_up = P(sube en H minutos)`.
4) Ejecutar un **logger 100% autónomo** que, por cada vela cerrada de 1 minuto, calcula `p_up` y lo guarda en un CSV.
5) Ejecutar un evaluador que compara esas probabilidades contra la realidad (velas futuras) y calcula métricas.

**Fase actual:** NO hay bot que opere (sin órdenes). Primero se valida que el sistema:
- genera probabilidades de forma estable,
- registra el histórico correctamente,
- puede evaluarse contra la realidad.

---

## 1) Requisitos

### 1.1 Sistema
- Windows 11
- Conexión a internet

### 1.2 Software
- Miniconda (recomendado) o Anaconda
- Anaconda Prompt / Miniconda Prompt
- (Opcional) VS Code + Git

---

## 2) Estructura del proyecto (carpetas y propósito)

```

C:\crypto_ai
README.md
src
01_download_klines.py
02_build_dataset.py
03_train_model.py
04_predict_latest.py
07_log_probabilities.py
08_evaluate_predictions.py
data
BTCUSDT_1m_raw.parquet
BTCUSDT_1m_dataset_h5_thr0.002.parquet
models
model_BTCUSDT_1m_h5_thr0.002.joblib
logs
preds_BTCUSDT_1m.csv
preds_eval_BTCUSDT_1m.csv

````

### `src/`
Scripts del pipeline: descarga, dataset, entrenamiento, predicción puntual, logger autónomo, evaluación.

### `data/`
Datos:
- `*_raw.parquet`: velas descargadas (OHLCV + campos extra de kline).
- `*_dataset_*.parquet`: dataset con features + etiqueta (target) para entrenar.

### `models/`
Modelos exportados:
- Fichero `.joblib` con:
  - modelo base (XGBoost),
  - calibrador de probabilidad (Platt scaling),
  - lista de features exactas usadas,
  - parámetros de configuración (símbolo, horizonte, thr).

### `logs/`
Logs operativos:
- `preds_*.csv`: probabilidades generadas por el logger por vela cerrada.
- `preds_eval_*.csv`: CSV enriquecido con evaluación (retorno futuro real, etiquetas reales, aciertos, etc.).

---

## 3) Entorno (Conda) — Instalación y configuración

> IMPORTANTE: ejecutar siempre en una consola que muestre `(cryptoai)` antes del prompt.  
> Si pone `(base)`, no estás en el entorno y faltarán librerías.

### 3.1 Crear entorno
Abrir **Anaconda Prompt (Miniconda3)** y ejecutar:

```bat
cd C:\crypto_ai
conda create -n cryptoai python=3.11 -y
conda activate cryptoai
````

### 3.2 Instalar dependencias

```bat
pip install pandas numpy pyarrow scikit-learn xgboost joblib matplotlib python-dotenv requests websockets
```

### 3.3 Verificar instalación

```bat
python --version
python -c "import pandas, numpy, sklearn, xgboost; print('OK')"
```

### 3.4 Problema típico: “Python desde Microsoft Store”

Si en PowerShell aparece un mensaje tipo “instalar Python desde Microsoft Store”, Windows está interceptando el comando `python`.

Soluciones:

* Usar **Anaconda Prompt** (recomendado).
* Desactivar aliases:

  * Configuración → Aplicaciones → Configuración avanzada → Alias de ejecución → desactivar `python.exe` / `python3.exe`.
* O ejecutar python por ruta:

  * `C:\Users\<USER>\miniconda3\envs\cryptoai\python.exe ...`

---

## 4) Definición del problema (MVP actual)

### 4.1 Parámetros actuales

* Símbolo: `BTCUSDT`
* Timeframe: `1m`
* Horizonte `H`: 5 minutos
* Umbral neutralidad `thr`: `0.002` (0.20%)

### 4.2 Target (etiqueta)

Para cada vela en tiempo `t` se calcula:

* `future_ret = close(t+H) / close(t) - 1`

Durante entrenamiento se define zona neutra:

* Si `abs(future_ret) <= thr` → se descarta (ruido).
* Si `future_ret > thr` → `y = 1` (sube).
* Si `future_ret < -thr` → `y = 0` (baja).

Durante evaluación se usan dos modos:

* **SIGNO:** evalúa la dirección (sube/baja) para todas las filas con futuro disponible.
* **FILTRADA:** evalúa solo las filas fuera de zona neutra (misma lógica del entrenamiento).

---

## 5) Scripts — Qué hace cada uno y qué genera

### 5.1 `src/01_download_klines.py` — Descargar velas

* Descarga velas (OHLCV) desde API pública de Binance.
* Guarda el histórico en Parquet.

Salida:

* `data/BTCUSDT_1m_raw.parquet`

Ejecutar:

```bat
conda activate cryptoai
cd C:\crypto_ai
python src\01_download_klines.py
```

Nota técnica (pandas):

* Para evitar errores de timezone en pandas 3.x se usa `pd.Timestamp.now(tz="UTC")`.

---

### 5.2 `src/02_build_dataset.py` — Features + etiqueta (dataset)

* Lee `data/BTCUSDT_1m_raw.parquet`.
* Calcula features (retornos, geometría de vela, volumen, EMAs, volatilidad).
* Crea etiquetas `y` con horizonte `H` y umbral `thr`.
* Descarta filas neutrales (dentro de `thr`) y filas con NaN por ventanas rolling.

Salida:

* `data/BTCUSDT_1m_dataset_h5_thr0.002.parquet`

Ejecutar:

```bat
python src\02_build_dataset.py
```

---

### 5.3 `src/03_train_model.py` — Entrenar modelo + calibración de probabilidades

* Entrena un `XGBClassifier` como modelo base.
* Calibra probabilidades con **Platt scaling**:

  * Entrena una `LogisticRegression` sobre la probabilidad del modelo base en validación (`p_raw`).
* Evalúa en test con:

  * AUC
  * Brier score

Motivo de calibración manual:

* En scikit-learn reciente, `CalibratedClassifierCV(cv="prefit")` ya no funciona como antes.

Salida:

* `models/model_BTCUSDT_1m_h5_thr0.002.joblib`

Ejecutar:

```bat
python src\03_train_model.py
```

---

### 5.4 `src/04_predict_latest.py` — Predicción puntual (manual)

* Descarga últimas velas (limit=500).
* Calcula features.
* Carga el modelo entrenado.
* Devuelve `p_up` de la última vela cerrada.

Ejecutar:

```bat
python src\04_predict_latest.py
```

Nota importante:

* Las columnas en la descarga deben coincidir con las del entrenamiento:

  * `quote_asset_volume`, `num_trades`, `taker_buy_base`, `taker_buy_quote`
  * Si se renombran (qav/trades/tbb/tbq), `feature_cols` no encaja y falla.

---

## 6) Logger 100% autónomo — `src/07_log_probabilities.py`

### 6.1 Objetivo

Este script se deja corriendo y:

* Detecta velas ya cerradas.
* Calcula `p_up` por cada vela cerrada.
* Guarda una fila por minuto en `logs/preds_BTCUSDT_1m.csv`.

### 6.2 Autonomía real (backfill)

El logger está preparado para:

* Si el proceso se queda pausado (por consola) o pierde tiempo, al volver:

  * recupera velas cerradas desde la última registrada (backfill)
  * escribe todas las predicciones faltantes en el CSV
* Evita duplicados leyendo el último `open_time_ms` guardado.

### 6.3 Problema típico en Windows: “se queda congelado”

En Windows, la consola puede PAUSAR el proceso si se selecciona texto con el ratón (QuickEdit Mode).
Síntoma:

* deja de imprimir
* al pulsar Enter reaparece y parece “saltarse minutos”

Solución:

* En la ventana: Properties → Options → desmarcar **QuickEdit Mode**
* Usar una consola dedicada y evitar seleccionar texto.

### 6.4 Salidas del logger

Genera/actualiza:

* `logs/preds_BTCUSDT_1m.csv`

Cada fila incluye:

* `open_time_utc`, `close_time_utc`
* `open_time_ms`, `close_time_ms` (alineación exacta por minuto)
* `symbol`, `interval`, `close`
* `p_raw` (probabilidad base XGB)
* `p_up` (probabilidad final calibrada)
* `horizon_min`, `thr`, `model_path`

### 6.5 Ejecución

```bat
conda activate cryptoai
cd C:\crypto_ai
mkdir logs
python src\07_log_probabilities.py
```

---

## 7) Evaluación — `src/08_evaluate_predictions.py`

### 7.1 Objetivo

* Leer `logs/preds_BTCUSDT_1m.csv`
* Para cada predicción, consultar velas reales y obtener el `close` a `t+H`.
* Calcular:

  * `future_ret`
  * etiquetas reales (SIGNO y FILTRADA por thr)
  * métricas (Accuracy y Brier, por modo)

### 7.2 Salidas de evaluación

* Imprime un resumen con:

  * Filas en log
  * Filas con futuro disponible
  * Filas en zona neutra (thr)
  * Accuracy + Brier en SIGNO
  * Accuracy + Brier filtrada por thr
* Genera CSV enriquecido:

  * `logs/preds_eval_BTCUSDT_1m.csv`

Campos extra en `preds_eval`:

* `close_h` (close real a +H)
* `future_ret`
* `y_sign` (etiqueta real dirección)
* `y_thr` (etiqueta real filtrada por thr)
* `y_pred` (predicción binaria por p_up>=0.5)

### 7.3 Ejecución correcta

Ejecutar desde el entorno:

```bat
conda activate cryptoai
cd C:\crypto_ai
python src\08_evaluate_predictions.py
```

### 7.4 Por qué a veces sale “No hay filas evaluables”

Puede ocurrir si:

* aún no han pasado `H` minutos desde las primeras filas del log (no hay futuro disponible), o
* el filtro `thr` hace que todas queden en zona neutra.

Ejemplo real observado:

* 19 filas log, 17 con futuro disponible, pero 17/17 en neutra para `thr=0.002` a `H=5`.
* En ese caso, la evaluación FILTRADA no tiene datos, pero SIGNO sí.

---

## 8) Flujo de trabajo completo (cómo se usa el proyecto)

### 8.1 Entrenamiento (offline)

```bat
conda activate cryptoai
cd C:\crypto_ai
python src\01_download_klines.py
python src\02_build_dataset.py
python src\03_train_model.py
```

### 8.2 Predicción puntual (sanity check)

```bat
python src\04_predict_latest.py
```

### 8.3 Logging autónomo (online)

En una consola dedicada:

```bat
python src\07_log_probabilities.py
```

### 8.4 Evaluación (online)

En otra consola:

```bat
python src\08_evaluate_predictions.py
```

Recomendación:

* Dejar correr el logger hasta tener **200–500 filas evaluables** para métricas más estables.

---

## 9) Estado actual del proyecto

✅ Funciona:

* pipeline de datos → entrenamiento → predicción
* logger autónomo con backfill
* evaluación con CSV enriquecido y métricas

🚫 Aún NO implementado:

* Bot conectado a Binance testnet para operar (se hará más adelante, tras validar edge).

---

## 10) Próximos pasos (se harán más adelante)

* Mejorar accuracy/AUC:

  * más histórico (1–2 años),
  * grid de parámetros `H` y `thr`,
  * nuevas features (order-flow proxies: taker ratio, num_trades),
  * validación walk-forward.
* Ejecutar logger/evaluador como servicio (Task Scheduler).
* Bot en testnet (paper trading → órdenes testnet → real si hay edge).

---

```
```
