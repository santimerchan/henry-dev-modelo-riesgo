"""Monitoreo de data drift para el modelo de riesgo crediticio.

Compara la distribución histórica de cada variable (con la que se entrenó el modelo)
contra la de un lote reciente, y reporta si alguna se movió lo suficiente como para
revisar el modelo.

Cada corrida genera:
    drift_report.json  -> foto de la última corrida (se sobrescribe)
    drift_history.csv  -> historial acumulado (se agrega una fila por variable)

Uso:
    python mlops_pipeline/src/model_monitoring.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

# ===========================================================================
# 1. CONFIGURACIÓN
# ===========================================================================

# Umbrales de drift. Están aquí, juntos y en mayúsculas, para cambiarlos en un
# solo lugar el día que el área de riesgo pida otro nivel de sensibilidad.
PSI_UMBRAL = 0.25          # PSI > 0.25 -> cambio significativo
KS_UMBRAL = 0.25           # estadístico KS > 0.25 -> cambio significativo
JS_UMBRAL = 0.3            # Jensen-Shannon > 0.3 -> cambio significativo
CHI2_PVALUE_UMBRAL = 0.05  # chi-cuadrado con p-value < 0.05 -> señal de cambio

# Este archivo vive en mlops_pipeline/src/, así que la raíz está dos niveles arriba.
RAIZ = Path(__file__).resolve().parents[2]

# Se dejan fuera 'fecha_prestamo' (marca de tiempo) y 'Pago_atiempo' (es el target).
NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
    "cant_creditosvigentes", "saldo_total", "promedio_ingresos_datacredito",
]

CATEGORICAS = ["tipo_laboral", "tendencia_ingresos", "tipo_credito"]


# ===========================================================================
# 2. DATOS
# ===========================================================================

def cargar_datos_historicos(path: str = "Base_de_datos.csv") -> pd.DataFrame:
    """Carga el CSV histórico, que es la distribución de referencia del modelo."""
    df = pd.read_csv(RAIZ / path)
    print(f"Histórico cargado: {len(df):,} filas x {df.shape[1]} columnas")
    return df


# ---------------------------------------------------------------------------
# TODO: reemplazar esta simulación por el log real de predicciones en
# producción (inputs + predicción del modelo) en cuanto ese pipeline exista.
# ---------------------------------------------------------------------------
def generar_batch_actual(
    df_historico: pd.DataFrame, frac: float = 0.3, seed: int = 42
) -> pd.DataFrame:
    """Simula el lote 'actual' que llegaría de producción.

    Toma una muestra del histórico y le aplica una perturbación controlada y
    reproducible: a las numéricas ruido gaussiano del 10% de su desviación estándar,
    más un desplazamiento deliberado de la media en 'puntaje' y 'salario_cliente'
    para que el drift sea visible; a las categóricas, proporciones distintas a las
    del histórico.
    """
    rng = np.random.default_rng(seed)
    df_actual = df_historico.sample(frac=frac, random_state=seed).copy()

    desplazamientos = {"puntaje": -0.8, "salario_cliente": 0.6}  # en desviaciones estándar
    for columna in NUMERICAS:
        sigma = df_historico[columna].std()
        ruido = rng.normal(0, 0.10 * sigma, len(df_actual))
        df_actual[columna] += ruido + desplazamientos.get(columna, 0) * sigma

    for columna in CATEGORICAS:
        proporciones = df_historico[columna].value_counts(normalize=True, dropna=False)
        pesos = proporciones.to_numpy().copy()
        pesos[0] *= 0.55   # la categoría dominante pierde peso
        pesos[1:] *= 1.80  # las demás lo ganan
        df_actual[columna] = rng.choice(
            proporciones.index, size=len(df_actual), p=pesos / pesos.sum()
        )

    print(f"Batch actual simulado: {len(df_actual):,} filas")
    return df_actual


# ===========================================================================
# 3. MÉTRICAS DE DRIFT
# ===========================================================================

def _proporciones_numericas(
    serie_historica: pd.Series, serie_actual: pd.Series, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reparte ambas series en bins por cuantiles DEL HISTÓRICO y devuelve proporciones.

    Los cortes se calculan sobre el histórico a propósito: es la referencia fija.
    Si se recalcularan con los datos nuevos, cada corrida se compararía consigo misma.
    """
    historica = serie_historica.dropna()
    actual = serie_actual.dropna()
    cortes = np.unique(np.quantile(historica, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf  # extremos abiertos: nada se queda por fuera

    prop_historica = np.histogram(historica, bins=cortes)[0] / len(historica)
    prop_actual = np.histogram(actual, bins=cortes)[0] / len(actual)
    return prop_historica, prop_actual


def _proporciones_categoricas(
    serie_historica: pd.Series, serie_actual: pd.Series
) -> pd.DataFrame:
    """Arma una tabla de frecuencias alineada: una fila por categoría, dos columnas."""
    tabla = pd.concat(
        [
            serie_historica.astype(str).value_counts(),
            serie_actual.astype(str).value_counts(),
        ],
        axis=1,
        keys=["historico", "actual"],
    ).fillna(0)
    return tabla


def calcular_ks(serie_historica: pd.Series, serie_actual: pd.Series) -> tuple[float, float]:
    """Prueba de Kolmogorov-Smirnov: máxima distancia entre las dos acumuladas."""
    resultado = ks_2samp(serie_historica.dropna(), serie_actual.dropna())
    return float(resultado.statistic), float(resultado.pvalue)


def calcular_psi(
    serie_historica: pd.Series, serie_actual: pd.Series, bins: int = 10
) -> float:
    """Population Stability Index: suma de (%actual - %hist) * ln(%actual / %hist)."""
    prop_historica, prop_actual = _proporciones_numericas(serie_historica, serie_actual, bins)

    # Los bins vacíos se reemplazan por un valor mínimo: sin esto habría log(0).
    prop_historica = np.where(prop_historica == 0, 1e-6, prop_historica)
    prop_actual = np.where(prop_actual == 0, 1e-6, prop_actual)

    return float(np.sum((prop_actual - prop_historica) * np.log(prop_actual / prop_historica)))


def calcular_jensen_shannon(
    serie_historica: pd.Series, serie_actual: pd.Series, bins: int = 10
) -> float:
    """Distancia de Jensen-Shannon (0 a 1). Sirve para numéricas y para categóricas."""
    if pd.api.types.is_numeric_dtype(serie_historica):
        prop_historica, prop_actual = _proporciones_numericas(serie_historica, serie_actual, bins)
    else:
        tabla = _proporciones_categoricas(serie_historica, serie_actual)
        prop_historica = tabla["historico"] / tabla["historico"].sum()
        prop_actual = tabla["actual"] / tabla["actual"].sum()

    return float(jensenshannon(prop_historica, prop_actual, base=2))


def calcular_chi_cuadrado(
    serie_historica: pd.Series, serie_actual: pd.Series
) -> tuple[float, float]:
    """Chi-cuadrado sobre la tabla de frecuencias: ¿cambiaron las proporciones?"""
    tabla = _proporciones_categoricas(serie_historica, serie_actual)
    estadistico, p_value, _, _ = chi2_contingency(tabla.T)
    return float(estadistico), float(p_value)


# ===========================================================================
# 4. EVALUACIÓN POR VARIABLE
# ===========================================================================

def evaluar_variable(
    nombre_columna: str,
    serie_historica: pd.Series,
    serie_actual: pd.Series,
    es_categorica: bool,
) -> dict:
    """Aplica las métricas que corresponden al tipo de variable y decide si hay drift.

    Basta con que UNA métrica cruce su umbral para marcar la variable. Preferimos una
    falsa alarma que se revisa en cinco minutos, a un modelo degradado sin que nadie
    se entere.
    """
    resultado = {
        "variable": nombre_columna,
        "tipo": "categorica" if es_categorica else "numerica",
        "psi": None, "ks": None, "jensen_shannon": None, "chi2_pvalue": None,
    }

    if es_categorica:
        _, resultado["chi2_pvalue"] = calcular_chi_cuadrado(serie_historica, serie_actual)
        resultado["jensen_shannon"] = calcular_jensen_shannon(serie_historica, serie_actual)
        hay_drift = (
            resultado["chi2_pvalue"] < CHI2_PVALUE_UMBRAL
            or resultado["jensen_shannon"] > JS_UMBRAL
        )
    else:
        resultado["ks"], _ = calcular_ks(serie_historica, serie_actual)
        resultado["psi"] = calcular_psi(serie_historica, serie_actual)
        resultado["jensen_shannon"] = calcular_jensen_shannon(serie_historica, serie_actual)
        hay_drift = (
            resultado["ks"] > KS_UMBRAL
            or resultado["psi"] > PSI_UMBRAL
            or resultado["jensen_shannon"] > JS_UMBRAL
        )

    resultado = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in resultado.items()}
    resultado["drift_detectado"] = bool(hay_drift)
    return resultado


# ===========================================================================
# 5. ORQUESTADOR
# ===========================================================================

def ejecutar_monitoreo(
    path_datos: str = "Base_de_datos.csv",
    columnas_categoricas: list[str] | None = None,
    columnas_numericas: list[str] | None = None,
) -> dict:
    """Corre el ciclo completo de monitoreo y guarda los resultados en disco.

    En producción esto no se corre a mano: se programa con una periodicidad definida
    (por ejemplo semanal, con cron, Airflow o un job de Jenkins) para que el historial
    se llene solo y la tendencia sea visible antes de que el modelo falle.
    """
    columnas_numericas = columnas_numericas or NUMERICAS
    columnas_categoricas = columnas_categoricas or CATEGORICAS

    df_historico = cargar_datos_historicos(path_datos)
    df_actual = generar_batch_actual(df_historico)

    resultados = [
        evaluar_variable(c, df_historico[c], df_actual[c], es_categorica=False)
        for c in columnas_numericas
    ] + [
        evaluar_variable(c, df_historico[c], df_actual[c], es_categorica=True)
        for c in columnas_categoricas
    ]

    con_drift = [r["variable"] for r in resultados if r["drift_detectado"]]
    reporte = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_historico": len(df_historico),
        "n_actual": len(df_actual),
        "batch_actual_simulado": True,
        "umbrales": {
            "PSI_UMBRAL": PSI_UMBRAL, "KS_UMBRAL": KS_UMBRAL,
            "JS_UMBRAL": JS_UMBRAL, "CHI2_PVALUE_UMBRAL": CHI2_PVALUE_UMBRAL,
        },
        "variables_evaluadas": len(resultados),
        "variables_con_drift": con_drift,
        "drift_detectado_global": bool(con_drift),
        "resultados": resultados,
    }

    # El JSON se sobrescribe (es la foto de hoy); el CSV se acumula (es la película).
    (RAIZ / "drift_report.json").write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    historial = RAIZ / "drift_history.csv"
    pd.DataFrame(resultados).assign(timestamp=reporte["timestamp"]).to_csv(
        historial, mode="a" if historial.exists() else "w",
        header=not historial.exists(), index=False,
    )

    print(f"Reporte guardado en {RAIZ / 'drift_report.json'}")
    print(f"Historial actualizado en {historial}")
    return reporte


# ===========================================================================
# 6. PUNTO DE ENTRADA
# ===========================================================================

if __name__ == "__main__":
    reporte = ejecutar_monitoreo()

    tabla = pd.DataFrame(reporte["resultados"]).set_index("variable")
    print("\n" + tabla.to_string() + "\n")

    if reporte["drift_detectado_global"]:
        print(f"Drift en {len(reporte['variables_con_drift'])} de {reporte['variables_evaluadas']} variables:")
        print("  " + ", ".join(reporte["variables_con_drift"]))
        print("Revisa esas variables y evalúa si hay que reentrenar el modelo.")
    else:
        print("Sin drift: las distribuciones se mantienen estables.")
