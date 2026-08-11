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

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from rfm_utils import compute_rfm, load_and_clean, resumen_por_segmento, run_clustering, train_explain_tree

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

# Orden de negocio (de mejor a peor cliente) y color asociado a cada segmento posible.
# Los colores estan pensados para que el nombre y el color coincidan intuitivamente
# (Oro = dorado, Plata = plateado, Bronce = cobre, Inactivo = gris apagado).
SEGMENT_ORDER = ["Oro", "Plata", "Bronce", "Nuevo", "Activo", "Inactivo"]
SEGMENT_COLORS = {
    "Oro": "#E8B84B",
    "Plata": "#AEB4BD",
    "Bronce": "#C97D4B",
    "Nuevo": "#4C9AFF",
    "Activo": "#4C9AFF",
    "Inactivo": "#6B7280",
}

st.set_page_config(page_title="Segmentación de Clientes - Retail Online", layout="wide", page_icon="🛍️")


def _orden_presente(valores) -> list:
    """Devuelve SEGMENT_ORDER filtrado a los segmentos que realmente existen en los datos."""
    return [s for s in SEGMENT_ORDER if s in set(valores)]


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
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("**1. Datos**")

usar_demo = st.sidebar.checkbox("Usar dataset de ejemplo (UCI Online Retail)", value=True)
archivo_subido = None
if not usar_demo:
    archivo_subido = st.sidebar.file_uploader(
        "Sube tu archivo de datos (mismas columnas que Online Retail.xlsx)",
        type=["xlsx"],
    )

st.sidebar.markdown("**2. Segmentación**")
k = st.sidebar.slider("Número de segmentos", min_value=2, max_value=5, value=4)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Esta app hace todo el proceso automáticamente: limpia los datos, calcula el historial "
    "de compra de cada cliente y los agrupa en segmentos con Machine Learning."
)

# --- Titulo ---
st.title("🛍️ Segmentación Inteligente de Clientes")
st.caption("Retail Online — resultados listos para tomar decisiones de negocio, sin necesidad de leer código.")

# --- Obtener los bytes del archivo a procesar ---
archivo_bytes = None
if usar_demo:
    with st.spinner("Descargando dataset de ejemplo..."):
        archivo_bytes = _descargar_dataset_demo()
elif archivo_subido is not None:
    archivo_bytes = archivo_subido.getvalue()

if archivo_bytes is None:
    st.info("Sube un archivo .xlsx en la barra lateral, o activa el dataset de ejemplo, para generar el dashboard.")
    st.stop()

# --- Procesar (con manejo de errores simple) ---
try:
    with st.spinner("Procesando datos y generando segmentos..."):
        df_clean, rfm, resumen, accuracy, reglas, importancias = _procesar(archivo_bytes, k)
except Exception as e:
    st.error(f"No se pudo procesar el archivo: {e}")
    st.stop()

orden = _orden_presente(resumen["Segmento"])
paleta = {s: SEGMENT_COLORS.get(s, "#4C9AFF") for s in orden}
resumen = resumen.set_index("Segmento").loc[orden].reset_index()

# ============================================================
# KPIs principales (los 4 que pide el reto)
# ============================================================
total_clientes = int(rfm.shape[0])
num_segmentos = int(rfm["Segmento"].nunique())
ingreso_total = rfm["Gasto_Total"].sum()
ingreso_promedio_segmento = resumen["Gasto_Total_suma"].mean()

k1, k2, k3, k4 = st.columns(4)
with k1, st.container(border=True):
    st.metric("👥 Total de clientes", f"{total_clientes:,}")
with k2, st.container(border=True):
    st.metric("🧩 Número de segmentos", f"{num_segmentos}")
with k3, st.container(border=True):
    st.metric("💰 Ingreso total", f"£{ingreso_total:,.0f}")
with k4, st.container(border=True):
    st.metric("📊 Ingreso promedio por segmento", f"£{ingreso_promedio_segmento:,.0f}")

st.markdown("")

# ============================================================
# Resumen ejecutivo (interpretación en lenguaje simple)
# ============================================================
st.subheader("📝 Lo que dicen los datos")

frases = []
if "Oro" in resumen["Segmento"].values:
    fila = resumen[resumen["Segmento"] == "Oro"].iloc[0]
    pct_clientes = fila["Clientes"] / total_clientes
    pct_ingreso = fila["Gasto_Total_suma"] / ingreso_total
    frases.append(
        f"El segmento **Oro** son solo el **{pct_clientes:.0%}** de tus clientes, pero generan el "
        f"**{pct_ingreso:.0%}** de tus ingresos. Son tu prioridad para fidelizar."
    )
if "Inactivo" in resumen["Segmento"].values:
    fila = resumen[resumen["Segmento"] == "Inactivo"].iloc[0]
    pct_clientes = fila["Clientes"] / total_clientes
    frases.append(
        f"**{fila['Clientes']:,.0f} clientes ({pct_clientes:.0%})** llevan en promedio "
        f"**{fila['Dias_Ultima_Compra_prom']:.0f} días** sin comprar. En el pasado gastaron un total de "
        f"£{fila['Gasto_Total_suma']:,.0f} — son candidatos para una campaña de reactivación."
    )
if "Plata" in resumen["Segmento"].values:
    fila = resumen[resumen["Segmento"] == "Plata"].iloc[0]
    frases.append(
        f"El segmento **Plata** ({fila['Clientes']:,.0f} clientes) compra con cierta regularidad "
        f"(cada ~{fila['Dias_Ultima_Compra_prom']:.0f} días desde su última compra); con el incentivo correcto "
        "podrían pasar a Oro."
    )

for f in frases:
    st.markdown(f"- {f}")

st.markdown("---")

# ============================================================
# Graficos: distribucion de clientes y comparacion de gasto
# ============================================================
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Clientes por segmento")
    fig_pie = px.pie(
        resumen, names="Segmento", values="Clientes", hole=0.45,
        color="Segmento", color_discrete_map=paleta,
        category_orders={"Segmento": orden},
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(legend_title_text="", margin=dict(t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Qué proporción de tu base de clientes cae en cada segmento.")

with col_der:
    st.subheader("Gasto total por segmento")
    fig_bar = px.bar(
        resumen, x="Segmento", y="Gasto_Total_suma", color="Segmento",
        color_discrete_map=paleta, category_orders={"Segmento": orden},
        labels={"Gasto_Total_suma": "Ingresos (£)"}, text_auto=".2s",
    )
    fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Cuánto dinero ha generado cada segmento en total.")

# ============================================================
# Representacion visual de los clusters
# ============================================================
st.subheader("Mapa de clientes")
st.caption(
    "Cada punto es un cliente. Más a la izquierda = compró más recientemente. Más arriba = compra más seguido. "
    "El tamaño del punto es cuánto ha gastado en total."
)
fig_scatter = px.scatter(
    rfm, x="Dias_Ultima_Compra", y="Frecuencia", size="Gasto_Total", color="Segmento",
    color_discrete_map=paleta, category_orders={"Segmento": orden},
    hover_data={"CustomerID": True, "Gasto_Total": ":,.0f", "Dias_Ultima_Compra": True, "Frecuencia": True},
    labels={"Dias_Ultima_Compra": "Días desde última compra", "Frecuencia": "Frecuencia de compra"},
    opacity=0.65, log_y=True,
)
fig_scatter.update_layout(legend_title_text="", margin=dict(t=10, b=10))
st.plotly_chart(fig_scatter, use_container_width=True)
st.caption(
    "El eje de frecuencia usa una escala ajustada (logarítmica) para poder ver bien tanto a los clientes "
    "que compran poco como a los que compran mucho, sin que estos últimos aplasten la gráfica."
)

# ============================================================
# Tabla resumen con metricas RFM por segmento
# ============================================================
st.subheader("Tabla resumen por segmento")
st.dataframe(
    resumen.rename(columns={
        "Clientes": "Clientes",
        "Dias_Ultima_Compra_prom": "Días desde última compra (prom.)",
        "Frecuencia_prom": "Frecuencia de compra (prom.)",
        "Gasto_Total_prom": "Gasto por cliente (prom.)",
        "Gasto_Total_suma": "Gasto total del segmento",
    }),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Gasto por cliente (prom.)": st.column_config.NumberColumn(format="£%.0f"),
        "Gasto total del segmento": st.column_config.NumberColumn(format="£%.0f"),
    },
)

# ============================================================
# Detalle tecnico (colapsado, no es el foco del dashboard)
# ============================================================
with st.expander("🔍 Detalle técnico: cómo el modelo distingue cada segmento"):
    st.markdown(
        "Un árbol de decisión entrenado para reconocer estos segmentos logra "
        f"**{accuracy:.1%} de precisión**, usando estas 3 señales del cliente:"
    )
    st.dataframe(
        importancias.rename(columns={"Variable": "Variable", "Importancia": "Peso en la decisión"}),
        use_container_width=True, hide_index=True,
        column_config={"Peso en la decisión": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
    )
    st.markdown("**Reglas que sigue el modelo (formato texto):**")
    st.code(reglas, language="text")

# ============================================================
# Tabla de clientes filtrable + descarga
# ============================================================
st.subheader("Listado de clientes")
segmento_filtro = st.multiselect("Filtrar por segmento", orden, default=orden)

rfm_filtrado = rfm[rfm["Segmento"].isin(segmento_filtro)][
    ["CustomerID", "Dias_Ultima_Compra", "Frecuencia", "Gasto_Total", "Segmento"]
].rename(columns={
    "CustomerID": "Cliente",
    "Dias_Ultima_Compra": "Días desde última compra",
    "Frecuencia": "Frecuencia de compra",
    "Gasto_Total": "Gasto total",
}).sort_values("Gasto total", ascending=False)

st.dataframe(
    rfm_filtrado, use_container_width=True, hide_index=True,
    column_config={"Gasto total": st.column_config.NumberColumn(format="£%.0f")},
)

csv = rfm_filtrado.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar esta tabla (CSV)",
    data=csv,
    file_name="clientes_segmentados.csv",
    mime="text/csv",
)
