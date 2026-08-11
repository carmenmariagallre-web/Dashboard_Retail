"""
PMV - Dashboard de Segmentacion Inteligente de Clientes (Retail Online)

Como correrlo localmente:
    pip install -r requirements.txt
    streamlit run app.py

Como usarlo:
    1. Sube tu archivo "Online Retail.xlsx" (o uno con las mismas columnas) en la barra lateral,
       o activa "Usar dataset de ejemplo (UCI)" para descargarlo automaticamente.
    2. Elige el numero de segmentos (k) para el clustering.
    3. Explora los KPIs, graficos, la tabla de clientes y las reglas de negocio del arbol.
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from rfm_utils import compute_rfm, load_and_clean, resumen_por_segmento, run_clustering, train_explain_tree

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

st.set_page_config(page_title="Segmentacion de Clientes - Retail Online", layout="wide")


@st.cache_data(show_spinner=False)
def _descargar_dataset_demo() -> bytes:
    """Descarga el dataset de ejemplo desde el repositorio UCI y devuelve el xlsx en bytes."""
    import zipfile

    resp = requests.get(UCI_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        nombre = z.namelist()[0]
        return z.read(nombre)


@st.cache_data(show_spinner=False)
def _procesar(archivo_bytes: bytes, k: int):
    df_clean = load_and_clean(io.BytesIO(archivo_bytes))
    rfm = compute_rfm(df_clean)
    rfm, kmeans, scaler = run_clustering(rfm, k=k)
    resumen = resumen_por_segmento(rfm)
    arbol, accuracy, reglas, importancias = train_explain_tree(rfm, max_depth=4)
    return df_clean, rfm, resumen, accuracy, reglas, importancias


# --- Barra lateral: carga de datos y parametros ---
st.sidebar.title("Configuracion")
st.sidebar.markdown("**1. Datos**")

usar_demo = st.sidebar.checkbox("Usar dataset de ejemplo (UCI Online Retail)", value=False)
archivo_subido = None
if not usar_demo:
    archivo_subido = st.sidebar.file_uploader(
        "Sube tu archivo de datos (mismas columnas que Online Retail.xlsx)",
        type=["xlsx"],
    )

st.sidebar.markdown("**2. Clustering**")
k = st.sidebar.slider("Numero de segmentos (k)", min_value=2, max_value=5, value=4)

st.sidebar.markdown("---")
st.sidebar.caption(
    "El pipeline hace todo el proceso automaticamente: limpieza de datos, calculo de RFM "
    "(Dias desde ultima compra, Frecuencia, Gasto Total), clustering con K-Means y un arbol "
    "de decision que explica los segmentos con reglas de negocio."
)

# --- Titulo ---
st.title("Segmentacion Inteligente de Clientes - Retail Online")
st.markdown(
    "Dashboard interactivo para explorar los segmentos de clientes generados a partir de su "
    "comportamiento de compra (modelo RFM + K-Means)."
)

# --- Obtener los bytes del archivo a procesar ---
archivo_bytes = None
if usar_demo:
    with st.spinner("Descargando dataset de ejemplo desde UCI..."):
        archivo_bytes = _descargar_dataset_demo()
elif archivo_subido is not None:
    archivo_bytes = archivo_subido.getvalue()

if archivo_bytes is None:
    st.info(
        "Sube un archivo .xlsx en la barra lateral, o activa el dataset de ejemplo, para "
        "generar el dashboard."
    )
    st.stop()

# --- Procesar (con manejo de errores simple) ---
try:
    with st.spinner("Procesando datos y entrenando modelos..."):
        df_clean, rfm, resumen, accuracy, reglas, importancias = _procesar(archivo_bytes, k)
except Exception as e:
    st.error(f"No se pudo procesar el archivo: {e}")
    st.stop()

# --- KPIs ---
total_clientes = rfm.shape[0]
ingresos_totales = rfm["Gasto_Total"].sum()
ticket_promedio = rfm["Gasto_Total"].mean()
clientes_inactivos = int((rfm["Segmento"] == "Inactivo").sum()) if "Inactivo" in rfm["Segmento"].values else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de clientes", f"{total_clientes:,}")
col2.metric("Ingresos totales", f"£{ingresos_totales:,.0f}")
col3.metric("Gasto promedio por cliente", f"£{ticket_promedio:,.0f}")
col4.metric("Clientes inactivos", f"{clientes_inactivos:,}")

st.markdown("---")

# --- Distribucion de clusters e ingresos por segmento ---
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Clientes por segmento")
    fig_pie = px.pie(
        resumen, names="Segmento", values="Clientes", hole=0.4,
        color="Segmento",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_der:
    st.subheader("Ingresos totales por segmento")
    fig_bar = px.bar(
        resumen.sort_values("Gasto_Total_suma", ascending=False),
        x="Segmento", y="Gasto_Total_suma", color="Segmento",
        labels={"Gasto_Total_suma": "Ingresos (£)"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Scatter RFM ---
st.subheader("Mapa de clientes (Dias desde ultima compra vs. Frecuencia)")
fig_scatter = px.scatter(
    rfm, x="Dias_Ultima_Compra", y="Frecuencia", size="Gasto_Total", color="Segmento",
    hover_data=["CustomerID", "Gasto_Total"],
    labels={"Dias_Ultima_Compra": "Dias desde ultima compra", "Frecuencia": "Frecuencia de compra"},
    opacity=0.7,
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Tabla resumen ---
st.subheader("Tabla resumen por segmento")
st.dataframe(
    resumen.rename(columns={
        "Dias_Ultima_Compra_prom": "Dias ultima compra (prom.)",
        "Frecuencia_prom": "Frecuencia (prom.)",
        "Gasto_Total_prom": "Gasto total (prom.)",
        "Gasto_Total_suma": "Gasto total (suma)",
    }),
    use_container_width=True,
    hide_index=True,
)

# --- Explicabilidad: reglas del arbol de decision ---
with st.expander("Como se explican los segmentos (reglas del arbol de decision)"):
    st.markdown(f"**Precision del arbol al predecir el segmento:** {accuracy:.1%}")
    st.markdown("**Importancia de cada variable:**")
    st.dataframe(importancias, use_container_width=True, hide_index=True)
    st.markdown("**Reglas de negocio (formato texto):**")
    st.code(reglas, language="text")

# --- Tabla de clientes filtrable + descarga ---
st.subheader("Clientes por segmento")
segmentos_disponibles = sorted(rfm["Segmento"].unique().tolist())
segmento_filtro = st.multiselect("Filtrar por segmento", segmentos_disponibles, default=segmentos_disponibles)

rfm_filtrado = rfm[rfm["Segmento"].isin(segmento_filtro)][
    ["CustomerID", "Dias_Ultima_Compra", "Frecuencia", "Gasto_Total", "Segmento"]
].sort_values("Gasto_Total", ascending=False)

st.dataframe(rfm_filtrado, use_container_width=True, hide_index=True)

csv = rfm_filtrado.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar tabla de clientes segmentados (CSV)",
    data=csv,
    file_name="clientes_segmentados.csv",
    mime="text/csv",
)
