"""
Funciones del pipeline de segmentacion de clientes (RFM + K-Means + Arbol explicativo).
Separadas del archivo de la app de Streamlit para poder probarlas y reutilizarlas.
"""

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# Nombres de negocio para los segmentos, ordenados de mejor a peor cliente.
# Se asignan segun el ranking de Gasto_Total promedio de cada cluster (de mayor a menor).
SEGMENT_NAMES = {
    2: ["Activo", "Inactivo"],
    3: ["Oro", "Plata", "Inactivo"],
    4: ["Oro", "Plata", "Bronce", "Inactivo"],
    5: ["Oro", "Plata", "Bronce", "Nuevo", "Inactivo"],
}

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def load_and_clean(file_or_path) -> pd.DataFrame:
    """Carga el Excel de Online Retail y aplica la limpieza estandar del proyecto.

    Reglas de limpieza (mismas que se validaron en el notebook de analisis):
    - Se descartan filas sin CustomerID (no se puede segmentar un cliente desconocido).
    - Se descartan cancelaciones (InvoiceNo que empieza con 'C').
    - Se descartan Quantity <= 0 y UnitPrice <= 0 (errores / devoluciones).
    - Se descartan filas duplicadas exactas.
    """
    df = pd.read_excel(file_or_path)

    faltantes = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(
            "El archivo no tiene las columnas esperadas del dataset Online Retail. "
            f"Faltan: {faltantes}"
        )

    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=["CustomerID"])
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    df_clean = df_clean.drop_duplicates()

    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

    return df_clean


def compute_rfm(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Calcula Dias_Ultima_Compra (Recencia), Frecuencia y Gasto_Total (Monetario) por cliente."""
    fecha_referencia = df_clean["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = (
        df_clean.groupby("CustomerID")
        .agg(
            Dias_Ultima_Compra=("InvoiceDate", lambda x: (fecha_referencia - x.max()).days),
            Frecuencia=("InvoiceNo", "nunique"),
            Gasto_Total=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    return rfm


def run_clustering(rfm: pd.DataFrame, k: int = 4, random_state: int = 42):
    """Transforma (log), escala, corre K-Means y asigna nombres de segmento de negocio.

    Los nombres se asignan ordenando los clusters por Gasto_Total promedio (de mayor a
    menor) y repartiendo SEGMENT_NAMES[k] en ese orden. Esto hace que el nombre "Oro"
    siempre caiga en el cluster que mas gasta, sin importar el numero interno que le
    puso K-Means (que es arbitrario).
    """
    rfm = rfm.copy()
    rfm["Dias_Ultima_Compra_log"] = np.log1p(rfm["Dias_Ultima_Compra"])
    rfm["Frecuencia_log"] = np.log1p(rfm["Frecuencia"])
    rfm["Gasto_Total_log"] = np.log1p(rfm["Gasto_Total"])

    cols_log = ["Dias_Ultima_Compra_log", "Frecuencia_log", "Gasto_Total_log"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[cols_log])

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    orden_clusters = (
        rfm.groupby("Cluster")["Gasto_Total"].mean().sort_values(ascending=False).index.tolist()
    )
    nombres = SEGMENT_NAMES.get(k, [f"Segmento {i + 1}" for i in range(k)])
    mapeo = {cluster_id: nombres[i] for i, cluster_id in enumerate(orden_clusters)}
    rfm["Segmento"] = rfm["Cluster"].map(mapeo)

    return rfm, kmeans, scaler


def resumen_por_segmento(rfm: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen: clientes, promedios RFM e ingresos totales por segmento."""
    resumen = (
        rfm.groupby("Segmento")
        .agg(
            Clientes=("CustomerID", "count"),
            Dias_Ultima_Compra_prom=("Dias_Ultima_Compra", "mean"),
            Frecuencia_prom=("Frecuencia", "mean"),
            Gasto_Total_prom=("Gasto_Total", "mean"),
            Gasto_Total_suma=("Gasto_Total", "sum"),
        )
        .round(1)
        .reset_index()
        .sort_values("Gasto_Total_prom", ascending=False)
    )
    return resumen


def train_explain_tree(rfm: pd.DataFrame, max_depth: int = 4):
    """Entrena un arbol de decision para explicar los segmentos con reglas legibles."""
    X = rfm[["Dias_Ultima_Compra", "Frecuencia", "Gasto_Total"]]
    y = rfm["Segmento"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    arbol = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    arbol.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, arbol.predict(X_test))
    reglas_texto = export_text(arbol, feature_names=list(X.columns))
    importancias = (
        pd.DataFrame({"Variable": X.columns, "Importancia": arbol.feature_importances_})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )

    return arbol, accuracy, reglas_texto, importancias
