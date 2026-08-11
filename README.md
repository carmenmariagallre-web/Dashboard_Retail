# Segmentación Inteligente de Clientes — Retail Online (PMV)

Dashboard interactivo en Streamlit que automatiza el proceso completo: carga de datos,
limpieza, cálculo de RFM (Días desde última compra, Frecuencia, Gasto Total), clustering
con K-Means y un árbol de decisión que explica cada segmento con reglas de negocio.

## Archivos

- `app.py` — la aplicación de Streamlit (interfaz y visualizaciones).
- `rfm_utils.py` — el pipeline de datos (limpieza, cálculo de RFM, clustering, árbol explicativo).
- `requirements.txt` — librerías necesarias.

## Correrlo en tu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`. En la barra lateral puedes subir tu propio archivo
`.xlsx` (con las columnas de Online Retail) o activar el dataset de ejemplo, que se
descarga automáticamente del repositorio UCI.

## Subirlo a GitHub

1. Crea un repositorio nuevo en GitHub (vacío, sin README ni .gitignore — ya los trae esta carpeta).
2. Desde esta carpeta (`pmv_dashboard`), en tu terminal:

```bash
git init
git add .
git commit -m "PMV: dashboard de segmentación de clientes"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

(Reemplaza `TU_USUARIO/TU_REPOSITORIO` por los datos de tu repositorio.)

## Publicarlo online gratis (Streamlit Community Cloud)

Así cualquiera puede abrir el dashboard desde un link, sin instalar nada:

1. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
2. Clic en "New app".
3. Elige el repositorio que acabas de subir, la rama `main`, y como archivo principal `app.py`.
4. Clic en "Deploy". En 1-2 minutos tendrás una URL pública del dashboard.

## Notas

- El dataset original (`Online Retail.xlsx`) no se incluye en el repositorio (ver `.gitignore`)
  porque pesa ~23 MB. Usa el botón "Usar dataset de ejemplo (UCI)" en la app, o sube tu propio
  archivo cada vez.
- Los nombres de segmento (Oro, Plata, Bronce, Inactivo) se asignan automáticamente según el
  gasto promedio de cada cluster — no dependen del orden interno que le da K-Means.
