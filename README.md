# 🚦 Dashboard — Siniestralidad Vial Colombia

Dashboard interactivo construido con Streamlit para visualizar los resultados
del análisis de siniestralidad vial en Colombia (RUNT 2022–2025).

---

## 🚀 Pasos para publicar en Streamlit Cloud (gratis, URL permanente para QR)

### PASO 2 — Crear la carpeta del proyecto en tu computador

```
mi-dashboard/
├── app.py               ← el archivo principal (ya te lo damos)
├── requirements.txt     ← dependencias (ya te lo damos)
└── README.md            ← este archivo (opcional)
```

---

### PASO 3 — Subir el proyecto a GitHub

1. Entra a https://github.com y crea una cuenta si no tienes.
2. Haz clic en **"New repository"** (botón verde arriba a la derecha).
3. Nómbralo: `siniestralidad-vial-colombia`
4. Selecciona **Public** y haz clic en **"Create repository"**.
5. En la página del repo recién creado, haz clic en **"uploading an existing file"**.
6. Arrastra y suelta los 2 archivos: `app.py` y `requirements.txt`.
7. Haz clic en **"Commit changes"** (botón verde abajo).

✅ Ya tienes el proyecto en GitHub.

---

### PASO 4 — Desplegar en Streamlit Community Cloud

1. Entra a https://share.streamlit.io
2. Haz clic en **"Sign in with GitHub"** (usa la misma cuenta de GitHub).
3. Haz clic en **"New app"** (botón azul arriba a la derecha).
4. En el formulario:
   - **Repository:** `tu-usuario/siniestralidad-vial-colombia`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Haz clic en **"Deploy!"**
6. En 2–3 minutos tendrás una URL permanente como:
   `https://siniestralidad-vial-colombia.streamlit.app`

✅ Ya tienes el dashboard publicado.

---

### PASO 5 — Generar el QR

1. Ve a https://www.qr-code-generator.com
2. Pega la URL de tu dashboard de Streamlit.
3. Personaliza el color (usa #065A82 para que combine con el diseño).
4. Descarga en PNG o SVG e insértalo en el pendón.

---

## 🖥️ Correr localmente (para probar antes de publicar)

```bash
# En la carpeta del proyecto:
streamlit run app.py
```

Se abre automáticamente en http://localhost:8501

---

## 📱 Secciones del dashboard

| Sección | Contenido |
|---|---|
| 🏠 Inicio | Contexto, problema, objetivos, marco teórico |
| 🚗 Objetivo 1 | Tipo y marca de vehículo vs. gravedad |
| 📅 Objetivo 2 | Antigüedad del vehículo vs. severidad |
| 🗺️ Objetivo 3 | Geografía y temporalidad |
| 🤖 Modelos ML | Ranking de 9 modelos, métricas, pipeline |
| ✅ Conclusiones | Hallazgos, limitaciones, referencias |
