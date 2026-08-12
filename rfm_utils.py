"""
Pipeline de segmentacion de clientes.

IMPORTANTE: este archivo NO entrena ningun modelo. El modelo (K-Means), el escalador
(StandardScaler) y el arbol de decision explicativo fueron entrenados por el usuario en su
propio notebook de Google Colab y exportados como archivos .pkl (con joblib). Este modulo
solo carga esos archivos y los aplica sobre datos nuevos:

    - load_and_clean / compute_rfm  -> preparan los datos (limpieza + calculo de RFM),
      con las mismas reglas usadas en el notebook.
    - cargar_modelo                 -> carga los 5 .pkl entrenados en Colab.
    - aplicar_modelo                -> transforma y predice usando esos objetos ya
      entrenados (scaler.transform + kmeans.predict), sin volver a ajustarlos.
    - explicar_con_arbol            -> usa el arbol ya entrenado para generar las reglas
      de negocio, sin reentrenarlo.
"""

import datetime as dt

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text

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

# Nombres de los 5 archivos exportados desde Colab (deben vivir junto a este archivo).
ARCHIVOS_MODELO = {
    "scaler": "scaler_clientes.pkl",
    "kmeans": "modelo_clientes.pkl",
    "columnas": "columnas_clientes.pkl",
    "mapeo": "mapeo_segmentos.pkl",
    "arbol": "arbol_clientes.pkl",
}


def load_and_clean(file_or_path) -> pd.DataFrame:
    """Carga el Excel de Online Retail y aplica la limpieza estandar del proyecto.

    Reglas de limpieza (las mismas que se usaron en el notebook de Colab):
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
    """Calcula Recencia, Frecuencia y Monetario por cliente (mismos nombres que en Colab)."""
    fecha_referencia = df_clean["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = (
        df_clean.groupby("CustomerID")
        .agg(
            Recencia=("InvoiceDate", lambda x: (fecha_referencia - x.max()).days),
            Frecuencia=("InvoiceNo", "nunique"),
            Monetario=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    return rfm


def cargar_modelo(carpeta: str = "."):
    """Carga los 5 artefactos entrenados en Colab (no entrena nada nuevo)."""
    import os

    rutas = {k: os.path.join(carpeta, v) for k, v in ARCHIVOS_MODELO.items()}
    faltantes = [v for v in rutas.values() if not os.path.exists(v)]
    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos del modelo entrenado en Colab: " + ", ".join(faltantes)
        )

    scaler = joblib.load(rutas["scaler"])
    kmeans = joblib.load(rutas["kmeans"])
    columnas = joblib.load(rutas["columnas"])
    mapeo = joblib.load(rutas["mapeo"])
    arbol = joblib.load(rutas["arbol"])
    return scaler, kmeans, columnas, mapeo, arbol


def aplicar_modelo(rfm: pd.DataFrame, scaler, kmeans, columnas, mapeo) -> pd.DataFrame:
    """Aplica el scaler y el K-Means YA ENTRENADOS (transform/predict, no fit)."""
    rfm = rfm.copy()
    rfm["Recencia_log"] = np.log1p(rfm["Recencia"])
    rfm["Frecuencia_log"] = np.log1p(rfm["Frecuencia"])
    rfm["Monetario_log"] = np.log1p(rfm["Monetario"])

    X_log = rfm[columnas]
    X_scaled_arr = scaler.transform(X_log)

    # El K-Means de Colab se entreno con un DataFrame cuyas columnas se llamaban
    # "*_scaled" (no "*_log"), asi que reconstruimos esos mismos nombres antes de
    # predecir -- si no, sklearn rechaza la prediccion por nombres de columna distintos.
    nombres_escalados = getattr(kmeans, "feature_names_in_", None)
    if nombres_escalados is not None:
        X_scaled = pd.DataFrame(X_scaled_arr, columns=nombres_escalados, index=rfm.index)
    else:
        X_scaled = X_scaled_arr

    rfm["Cluster"] = kmeans.predict(X_scaled)
    rfm["Segmento"] = rfm["Cluster"].map(mapeo)

    # Nombres de negocio usados en el resto de la app y el dashboard.
    rfm = rfm.rename(columns={"Recencia": "Dias_Ultima_Compra", "Monetario": "Gasto_Total"})
    return rfm


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


def explicar_con_arbol(rfm: pd.DataFrame, arbol):
    """Usa el arbol YA ENTRENADO en Colab para generar reglas y medir que tan bien
    explica los segmentos del archivo cargado actualmente (no se reentrena nada)."""
    X = rfm[["Dias_Ultima_Compra", "Frecuencia", "Gasto_Total"]]
    y = rfm["Segmento"]

    y_pred = arbol.predict(X)
    accuracy = accuracy_score(y, y_pred)
    reglas_texto = export_text(arbol, feature_names=list(X.columns))
    importancias = (
        pd.DataFrame({"Variable": X.columns, "Importancia": arbol.feature_importances_})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )

    return accuracy, reglas_texto, importancias
