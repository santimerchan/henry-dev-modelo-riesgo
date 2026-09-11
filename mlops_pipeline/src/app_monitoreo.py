"""Tablero de Streamlit para leer los resultados del monitoreo de data drift.

Lee lo que genera model_monitoring.py (drift_report.json y drift_history.csv) y lo
presenta en tres pestañas: métricas de la última corrida, evolución en el tiempo y
recomendaciones automáticas.

Uso:
    streamlit run mlops_pipeline/src/app_monitoreo.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from model_monitoring import (  # noqa: E402
    CHI2_PVALUE_UMBRAL, JS_UMBRAL, KS_UMBRAL, PSI_UMBRAL, RAIZ,
    generar_batch_actual,
)

st.set_page_config(page_title="Monitoreo de Data Drift", page_icon="📊", layout="wide")


# ===========================================================================
# 1. CARGA DE DATOS
# ===========================================================================

@st.cache_data
def cargar_todo():
    """Carga el reporte, el historial y las dos distribuciones que se comparan."""
    reporte = json.loads((RAIZ / "drift_report.json").read_text(encoding="utf-8"))

    historial = pd.read_csv(RAIZ / "drift_history.csv")
    historial["timestamp"] = pd.to_datetime(historial["timestamp"])

    df_historico = pd.read_csv(RAIZ / "Base_de_datos.csv")
    df_actual = generar_batch_actual(df_historico)  # misma semilla: mismas distribuciones

    return reporte, historial, df_historico, df_actual


def metrica_principal(tipo: str) -> str:
    """PSI para las numéricas, Jensen-Shannon para las categóricas."""
    return "psi" if tipo == "numerica" else "jensen_shannon"


def semaforo(resultado: dict) -> str:
    """Rojo si cruzó un umbral, amarillo si va por encima del 60%, verde si está sano."""
    if resultado["drift_detectado"]:
        return "🔴"
    valor = resultado[metrica_principal(resultado["tipo"])]
    umbral = PSI_UMBRAL if resultado["tipo"] == "numerica" else JS_UMBRAL
    return "🟡" if valor > umbral * 0.6 else "🟢"


if not (RAIZ / "drift_report.json").exists():
    st.error("Todavía no hay reporte. Corre primero: `python mlops_pipeline/src/model_monitoring.py`")
    st.stop()

reporte, historial, df_historico, df_actual = cargar_todo()
resultados = {r["variable"]: r for r in reporte["resultados"]}
n_con_drift = len(reporte["variables_con_drift"])


# ===========================================================================
# 2. ENCABEZADO
# ===========================================================================

st.title("📊 Monitoreo de Data Drift — Modelo de Riesgo Crediticio")
st.caption(
    f"Última corrida: {reporte['timestamp']} · "
    f"{reporte['n_historico']:,} filas históricas vs {reporte['n_actual']:,} del batch actual"
)

col1, col2, col3 = st.columns(3)
col1.metric("Variables evaluadas", reporte["variables_evaluadas"])
col2.metric("Variables con drift", n_con_drift)
col3.metric("Corridas registradas", historial["timestamp"].nunique())

st.info("El batch 'actual' es **simulado** mientras no exista un log real de producción.")

tab1, tab2, tab3 = st.tabs(
    ["📈 Visualización de métricas", "🕒 Análisis temporal", "🚨 Recomendaciones y alertas"]
)


# ===========================================================================
# TAB 1 — VISUALIZACIÓN DE MÉTRICAS
# ===========================================================================

with tab1:
    variable = st.selectbox("Variable:", list(resultados.keys()))
    resultado = resultados[variable]
    st.markdown(f"### {semaforo(resultado)} {variable}")

    figura, eje = plt.subplots(figsize=(9, 4))
    if resultado["tipo"] == "numerica":
        eje.hist(df_historico[variable].dropna(), bins=40, alpha=0.6, density=True, label="Histórico")
        eje.hist(df_actual[variable].dropna(), bins=40, alpha=0.6, density=True, label="Batch actual")
    else:
        pd.DataFrame({
            "Histórico": df_historico[variable].astype(str).value_counts(normalize=True),
            "Batch actual": df_actual[variable].astype(str).value_counts(normalize=True),
        }).fillna(0).head(8).plot(kind="bar", ax=eje)

    eje.set_title(f"{variable}: histórico vs batch actual")
    eje.legend()
    st.pyplot(figura)

    st.subheader("Métricas por variable")
    st.dataframe(
        pd.DataFrame([
            {
                "": semaforo(r), "Variable": r["variable"], "Tipo": r["tipo"],
                "PSI": r["psi"], "KS": r["ks"], "Jensen-Shannon": r["jensen_shannon"],
                "Chi² p-value": r["chi2_pvalue"], "Drift": "Sí" if r["drift_detectado"] else "No",
            }
            for r in reporte["resultados"]
        ]),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Umbrales: PSI > {PSI_UMBRAL} · KS > {KS_UMBRAL} · JS > {JS_UMBRAL} · "
        f"Chi² p < {CHI2_PVALUE_UMBRAL}   |   🟢 estable · 🟡 en vigilancia · 🔴 drift"
    )


# ===========================================================================
# TAB 2 — ANÁLISIS TEMPORAL
# ===========================================================================

with tab2:
    if historial["timestamp"].nunique() < 2:
        st.warning("Con una sola corrida no hay tendencia. Corre el monitoreo otra vez.")

    seleccionadas = st.multiselect(
        "Variables a graficar:", list(resultados.keys()), default=list(resultados.keys())[:5]
    )

    figura, eje = plt.subplots(figsize=(10, 4))
    for nombre in seleccionadas:
        serie = historial[historial["variable"] == nombre].sort_values("timestamp")
        eje.plot(serie["timestamp"], serie[metrica_principal(resultados[nombre]["tipo"])],
                 marker="o", label=nombre)
    eje.axhline(PSI_UMBRAL, linestyle="--", color="red", alpha=0.5)
    eje.set_ylabel("Métrica principal (PSI o JS)")
    eje.legend(fontsize=8)
    figura.autofmt_xdate()
    st.pyplot(figura)

    st.subheader("Tendencia")
    st.caption("Compara el promedio de la primera mitad de las corridas contra el de la segunda.")

    filas = []
    for nombre, resultado in resultados.items():
        serie = historial[historial["variable"] == nombre].sort_values("timestamp")
        valores = serie[metrica_principal(resultado["tipo"])].to_numpy()

        if len(valores) < 2:
            tendencia, detalle = "Sin datos", "Se necesita más de una corrida"
        else:
            mitad = len(valores) // 2
            inicio, final = valores[:mitad].mean(), valores[mitad:].mean()
            variacion = (final - inicio) / abs(inicio) if inicio else 0
            tendencia = "📈 Sube" if variacion > 0.1 else "📉 Baja" if variacion < -0.1 else "➡️ Estable"
            detalle = f"{inicio:.4f} → {final:.4f} ({variacion:+.1%})"

        filas.append({"Variable": nombre, "Corridas": len(valores),
                      "Tendencia": tendencia, "Detalle": detalle})

    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)


# ===========================================================================
# TAB 3 — RECOMENDACIONES Y ALERTAS
# ===========================================================================

with tab3:
    proporcion = n_con_drift / reporte["variables_evaluadas"]
    severidad = "Ninguna" if n_con_drift == 0 else "Alta" if proporcion >= 0.3 else "Media"

    st.metric("Severidad de la alerta", severidad)

    if n_con_drift == 0:
        st.success("No se detectó drift. Las distribuciones están dentro de los umbrales.")
        st.markdown(
            "- ✅ No necesitas reentrenar el modelo por ahora.\n"
            "- 🔁 Mantén la corrida programada para seguir acumulando historial."
        )
    else:
        st.error(f"Drift en {n_con_drift} de {reporte['variables_evaluadas']} variables ({proporcion:.0%}).")

        st.subheader("Variables a revisar")
        for nombre in reporte["variables_con_drift"]:
            resultado = resultados[nombre]
            metrica = metrica_principal(resultado["tipo"])
            st.markdown(f"- 🔴 `{nombre}` ({resultado['tipo']}) — {metrica.upper()} = {resultado[metrica]}")

        st.subheader("Qué hacer")
        st.markdown(
            "- 🔍 Confirma con el equipo de datos si hubo un cambio en la fuente o en el ETL.\n"
            "- 🏷️ Verifica que no haya categorías nuevas o renombradas que rompan el encoding.\n"
            + ("- ♻️ **Reentrena el modelo** con datos recientes: con este nivel de drift el "
               "desempeño en producción probablemente ya se degradó.\n" if severidad == "Alta"
               else "- 👀 Aún no es obligatorio reentrenar; deja estas variables en vigilancia.\n")
            + "- 📊 Contrasta con la mora observada antes de decidir el reentrenamiento."
        )

    st.subheader("Alertas")
    alertas = [
        {"Severidad": "Alta" if r["drift_detectado"] else "Media", "Variable": r["variable"],
         "Métrica": metrica_principal(r["tipo"]).upper(),
         "Valor": r[metrica_principal(r["tipo"])], "Estado": semaforo(r)}
        for r in reporte["resultados"] if semaforo(r) != "🟢"
    ]
    if alertas:
        st.dataframe(pd.DataFrame(alertas), width="stretch", hide_index=True)
    else:
        st.success("Sin alertas activas. 🟢")
