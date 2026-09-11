# henry-dev-modelo-riesgo

Modelo predictivo mediante técnicas de aprendizaje automático, utilizando información histórica de créditos, con el objetivo de anticipar el comportamiento de nuevos usuarios. La empresa opera bajo un esquema estructurado de proyectos, en el cual cada iniciativa debe seguir una arquitectura de carpetas estrictamente definida.

---

## 💼 Caso de negocio

La entidad financiera aprueba créditos todos los días y necesita saber, **antes de
desembolsar**, qué tan probable es que el cliente pague a tiempo. El modelo predice la
variable `Pago_atiempo` a partir del perfil del solicitante: capital pedido, plazo, edad,
salario, puntaje interno, puntaje de Datacrédito, obligaciones vigentes, entre otras.

El impacto es directo en dos frentes:

- **Pérdida esperada:** aprobar clientes que no van a pagar se traduce en cartera vencida
  y en provisiones que golpean el estado de resultados.
- **Costo de oportunidad:** negar clientes buenos por un modelo mal calibrado significa
  dejar de colocar créditos rentables frente a la competencia.

El dataset tiene 10.763 registros y 23 variables, y está desbalanceado (10.252 pagos a
tiempo contra 511 incumplimientos, cerca del 4,7% de mora). Por eso el desempeño se
evalúa con precision, recall, F1 y ROC-AUC, nunca con accuracy sola.

---

## 🔍 ¿Por qué monitorear data drift?

Un modelo de riesgo se entrena sobre una fotografía del pasado, pero la realidad no se
queda quieta: cambia la política de originación, entra un segmento de clientes distinto,
se ajusta el scoring del buró, o simplemente el ciclo económico se mueve. Cuando la
distribución de las variables de entrada se aleja de la que el modelo vio en
entrenamiento, hablamos de **data drift**.

Lo peligroso del drift es que **no lanza un error**. El pipeline sigue corriendo, la API
sigue respondiendo, y el modelo sigue entregando probabilidades con toda confianza —
solo que cada vez peor. En un modelo de crédito eso se descubre meses después, cuando la
mora real ya subió y el daño está hecho.

Monitorear drift es el sistema de alarma temprana: nos avisa que las entradas cambiaron
**antes** de que el negocio sienta el deterioro, y nos permite decidir a tiempo si hay
que investigar la fuente de datos o reentrenar el modelo.

---

## 📁 Estructura del proyecto

Esta arquitectura de carpetas es **fija** y no se debe reorganizar: las validaciones
automáticas de la compañía dependen de ella.

```
henry-dev-modelo-riesgo/
│
├── mlops_pipeline/
│   └── src/
│       ├── cargar_datos.ipynb              # 1️⃣ Cargar datos desde CSV
│       ├── comprension_eda.ipynb           # 2️⃣ Análisis exploratorio de datos
│       ├── ft_engineering.py               # 3️⃣ Feature engineering
│       ├── model_training_evaluation.py    # 4️⃣ Entrenar y evaluar modelos
│       ├── model_monitoring.py             # 5️⃣ Monitoreo de data drift
│       └── app_monitoreo.py                # 6️⃣ App de Streamlit para visualizar el drift
│
├── Base_de_datos.csv                       # 📊 Dataset principal (10.763 registros)
├── Base_de_datos.xlsx                      # 📑 Datos originales en Excel
├── requirements.txt                        # 📦 Dependencias de Python
├── .gitignore                              # 🚫 Archivos a ignorar en Git
└── README.md                               # 📖 Este archivo
```

Al ejecutar el monitoreo se generan además dos archivos en la raíz:
`drift_report.json` (última corrida) y `drift_history.csv` (historial acumulado).

---

## 🏗️ Arquitectura del monitoreo

Todo el monitoreo vive en `mlops_pipeline/src/model_monitoring.py`. El script compara
dos distribuciones —el **histórico** (`Base_de_datos.csv`, lo que el modelo conoce)
contra un **batch actual** (lo que está llegando ahora)— variable por variable.

### Métricas y umbrales

Los umbrales están definidos como constantes en mayúsculas al inicio del script:

| Métrica | Constante | Umbral | Aplica a | Se marca drift si… |
|---------|-----------|--------|----------|--------------------|
| PSI (Population Stability Index) | `PSI_UMBRAL` | `0.25` | Numéricas | PSI > 0.25 |
| KS (Kolmogorov-Smirnov) | `KS_UMBRAL` | `0.25` | Numéricas | estadístico > 0.25 |
| Jensen-Shannon | `JS_UMBRAL` | `0.3` | Numéricas y categóricas | distancia > 0.3 |
| Chi-cuadrado | `CHI2_PVALUE_UMBRAL` | `0.05` | Categóricas | p-value < 0.05 |

Una variable numérica se evalúa con PSI, KS y Jensen-Shannon; una categórica con
chi-cuadrado y Jensen-Shannon. Basta con que **una** métrica supere su umbral para marcar
la variable con `drift_detectado = True`.

### Variables monitoreadas

- **Numéricas (11):** `capital_prestado`, `plazo_meses`, `edad_cliente`, `salario_cliente`,
  `total_otros_prestamos`, `cuota_pactada`, `puntaje`, `puntaje_datacredito`,
  `cant_creditosvigentes`, `saldo_total`, `promedio_ingresos_datacredito`.
- **Categóricas (3):** `tipo_laboral`, `tendencia_ingresos`, `tipo_credito`.

Quedan fuera `fecha_prestamo` (es una marca de tiempo, no una feature) y `Pago_atiempo`
(es el target, no un input del modelo).

### Funciones del script

| Función | Qué hace |
|---------|----------|
| `cargar_datos_historicos(path)` | Carga el CSV de referencia |
| `generar_batch_actual(df, frac, seed)` | Simula el lote actual con perturbación controlada |
| `calcular_ks(...)` | Prueba de Kolmogorov-Smirnov, devuelve (estadístico, p-value) |
| `calcular_psi(...)` | PSI manual con bins por cuantiles del histórico |
| `calcular_jensen_shannon(...)` | Distancia JS sobre distribuciones discretizadas |
| `calcular_chi_cuadrado(...)` | Chi-cuadrado sobre tabla de contingencia |
| `evaluar_variable(...)` | Aplica las métricas que le corresponden al tipo de variable |
| `ejecutar_monitoreo(...)` | Orquesta todo y escribe los archivos de salida |

### Salidas que produce cada corrida

- **`drift_report.json`** — fotografía de la última corrida (se sobrescribe cada vez).
- **`drift_history.csv`** — una fila por variable y por corrida; se crea con encabezados
  la primera vez y luego crece en modo *append*. Es el insumo del análisis temporal.

En producción, `ejecutar_monitoreo()` no se corre a mano: se programa con periodicidad
definida (por ejemplo, **semanal** vía cron, Airflow o un job de Jenkins) para que el
historial se llene solo y la tendencia sea visible sin esperar a que el modelo falle.

---

## ▶️ Cómo ejecutar

### 1. Preparar el entorno

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Script de monitoreo

```bash
python mlops_pipeline/src/model_monitoring.py
```

Imprime un resumen en consola y deja `drift_report.json` y `drift_history.csv` en la
raíz del proyecto.

### 3. App de visualización (Streamlit)

```bash
streamlit run mlops_pipeline/src/app_monitoreo.py
```

Se abre en `http://localhost:8501` con tres pestañas:

- **📈 Visualización de métricas** — selector de variable, comparación gráfica de histórico
  vs actual (histograma para numéricas, barras de proporción para categóricas), tabla
  completa de métricas y semáforo por variable (🟢 estable · 🟡 en vigilancia ·
  🔴 drift detectado).
- **🕒 Análisis temporal** — evolución de la métrica principal de cada variable a lo largo
  de las corridas registradas en `drift_history.csv`, con la tendencia calculada
  comparando el promedio de la primera mitad de las corridas contra el de la segunda.
- **🚨 Recomendaciones y alertas** — diagnóstico automático de la última corrida, qué
  variables revisar, si conviene reentrenar, y el listado de alertas con severidad
  (alta / media / ninguna).

> La app necesita que `model_monitoring.py` se haya ejecutado al menos una vez. Si no
> encuentra `drift_report.json`, te lo dice explícitamente en pantalla.

---

## 📉 Hallazgos de la última corrida

> ⏳ **Pendiente:** esta sección se completa después de ejecutar
> `python mlops_pipeline/src/model_monitoring.py` en este repositorio. Los valores
> reales quedan en `drift_report.json` y se resumen aquí: variables con drift, sus
> métricas, y la recomendación resultante.

Qué esperar al leer los resultados:

- **PSI y KS no siempre coinciden.** Las variables discretas concentradas en pocos
  valores (como `plazo_meses`) pueden disparar PSI alto con un KS moderado, porque el PSI
  compara proporciones bin por bin y castiga fuerte cualquier redistribución.
- **El chi-cuadrado es muy sensible con muestras grandes.** Con más de diez mil registros
  detecta diferencias mínimas y devuelve p-values cercanos a cero. Por eso se lee junto a
  Jensen-Shannon: el chi-cuadrado dice si el cambio es *real*, JS dice si es *grande*.
- **El semáforo amarillo importa.** Una variable que no cruzó el umbral pero se le acercó
  es justamente la que hay que vigilar en la próxima corrida.

---

## ⚠️ Nota importante: el batch actual es simulado

Todavía **no existe un log real de predicciones en producción**, así que el "batch actual"
se construye tomando una muestra del 30% del histórico y aplicándole una perturbación
controlada y reproducible (`seed=42`):

- **Numéricas:** ruido gaussiano con desviación igual al 10% de la desviación estándar de
  cada variable, más un desplazamiento intencional de media en `puntaje` (−0.8 σ) y
  `salario_cliente` (+0.6 σ) para que el drift sea visible.
- **Categóricas:** se reponderan las proporciones (la categoría dominante pierde peso y
  las demás lo ganan).

La función `generar_batch_actual()` lleva un `# TODO` explícito arriba: debe reemplazarse
por el log real de predicciones (inputs + predicción del modelo) en cuanto ese pipeline
exista. **La lógica de las métricas, los umbrales y toda la infraestructura de reporte son
las definitivas** — lo único provisional es de dónde salen los datos del batch actual.

---

## 🔀 Flujo de ramas y estado de este avance

```
developer  ──PR #1──►  certification  ──PR #2──►  main
(desarrollo)           (pruebas/QA)              (producción)
```

- **`developer`** — donde se escribe el código. Aquí se hace el commit de este avance.
- **`certification`** — ambiente de pruebas. Nada llega aquí sin pasar por un Pull Request
  revisado desde `developer`.
- **`main`** — producción. Solo recibe lo que ya fue certificado.

**Este avance corresponde al primer Pull Request: `developer` → `certification`.**
No se abre PR hacia `main` en este punto; la promoción a producción es un paso posterior
y separado, que ocurre solo después de que el código haya sido validado en certification.

---

## 🛠️ Tecnologías utilizadas

| Área | Herramientas |
|------|--------------|
| **Lenguaje** | Python 3.10 |
| **Datos** | pandas, numpy |
| **ML** | scikit-learn, xgboost |
| **Experimentos** | MLflow |
| **Monitoreo** | scipy (KS, chi², Jensen-Shannon), Streamlit |
| **Visualización** | seaborn, matplotlib |
| **API** | FastAPI, uvicorn |
