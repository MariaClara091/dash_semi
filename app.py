import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Siniestralidad Vial Colombia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def cargar_datos():
    tipo_vehiculo = pd.DataFrame({
        "Tipo": ["TRACTOCAMION","VOLQUETA","CAMION","BUS","BUSETA",
                 "CAMPERO","CAMIONETA","AUTOMOVIL","MOTOCICLETA","MICROBUS"],
        "Tasa_Mortalidad": [28.4,24.1,19.7,12.3,10.8,7.2,5.9,4.8,4.6,4.1],
        "Volumen": [3200,4100,8700,5400,3800,12000,28000,95000,210000,4200]
    })
    antiguedad = pd.DataFrame({
        "Categoria": ["0-5 años","6-10 años","11-20 años",">20 años"],
        "Tasa_Mortalidad": [3.2,4.8,6.7,9.4],
        "IC_inf": [3.0,4.6,6.5,9.0],
        "IC_sup": [3.4,5.0,6.9,9.8],
        "N": [142000,128000,89000,42000]
    })
    departamentos = pd.DataFrame({
        "Departamento": ["ANTIOQUIA","BOGOTA D.C.","VALLE DEL CAUCA",
                         "CUNDINAMARCA","SANTANDER","ATLANTICO",
                         "BOLIVAR","CORDOBA","CASANARE","META",
                         "HUILA","NARIÑO","CAUCA","TOLIMA","RISARALDA"],
        "Total_Accidentes": [68000,62000,48000,22000,18000,
                              15000,12000,8000,6000,7500,
                              9000,7200,6800,8500,11000],
        "Tasa_Mortalidad": [4.2,3.8,4.5,5.8,6.2,
                             4.9,7.1,8.4,9.8,8.1,
                             7.3,8.9,8.2,6.9,4.8]
    })
    meses = pd.DataFrame({
        "Mes": ["Ene","Feb","Mar","Abr","May","Jun",
                "Jul","Ago","Sep","Oct","Nov","Dic"],
        "Accidentes": [28000,24000,26500,25000,27000,29000,
                       32000,27500,26000,28000,25500,30000],
        "Tasa_Mortalidad": [5.1,4.8,5.4,4.9,5.0,5.2,
                             5.6,5.1,5.5,5.0,4.7,5.3]
    })
    dias = pd.DataFrame({
        "Dia": ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"],
        "Accidentes": [12000,11500,11800,12200,14000,16000,13500],
        "Tasa_Mortalidad": [4.6,4.4,4.5,4.7,5.1,5.8,6.2]
    })
    modelos = pd.DataFrame({
        "Modelo": ["Balanced Random Forest","XGBoost","Balanced Bagging + LightGBM",
                   "Random Forest","Logistic Regression","Isolation Forest",
                   "RUSBoost","LightGBM + SMOTE","IMBoost"],
        "F1_Macro": [0.6405,0.6262,0.6251,0.6244,0.5961,0.5894,0.5847,0.5518,0.5236],
        "F1_Muertos": [0.3365,0.3235,0.3216,0.2980,0.2903,0.2493,0.2778,0.1321,0.2007],
        "Recall_Muertos": [0.4889,0.5764,0.5764,0.3673,0.6633,0.4020,0.6812,0.0790,0.6074],
        "MCC": [0.3043,0.3021,0.3021,0.2559,0.2861,0.2081,0.2767,0.1600,0.1810],
        "ROC_AUC": [0.8269,0.8251,0.8250,0.7894,0.8190,0.7432,0.8246,0.8034,0.7456],
        "PR_AUC": [0.2490,0.2600,0.2539,0.2110,0.2410,0.1548,0.2430,0.2287,0.1622]
    })
    return tipo_vehiculo, antiguedad, departamentos, meses, dias, modelos

tipo_vehiculo, antiguedad, departamentos, meses, dias, modelos = cargar_datos()

# SIDEBAR
with st.sidebar:
    st.title(" Siniestralidad Vial")
    st.caption("Colombia — RUNT 2022–2025")
    st.divider()
    seccion = st.radio("Navegar a:", [
        " Inicio",
        " Objetivo 1: Características Vehículos",
        " Objetivo 2: Antigüedad",
        " Objetivo 3: Geografía & Tiempo",
        " Modelos",
        " Conclusiones"
    ])
    st.divider()
    st.metric("Registros", "406,540")
    st.metric("Accidentes fatales", "5%")
    st.metric("Mejor F1-Macro", "0.64")
    st.divider()
    st.caption("M. C. Ávila · M. J. Giraldo · A. D. Moya")
    st.caption("Seminario de Investigación 2026")

# ── INICIO ──────────────────────────────────────────────────────────────────
if seccion == "Inicio":
    st.title("Análisis de Siniestralidad Vial en Colombia")
    st.subheader("Técnicas de Machine Learning aplicadas al RUNT · 2022–2025")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros analizados", "406,540")
    c2.metric("Accidentes fatales", "5%")
    c3.metric("Modelos evaluados", "9")
    c4.metric("Mejor F1-Macro", "0.6405")

    st.divider()
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader(" Planteamiento del Problema")
        st.write("""
        La siniestralidad vial en Colombia representa una de las principales causas de mortalidad evitable.
        A pesar de los miles de registros disponibles en el RUNT, sin un análisis sistemático es imposible
        identificar qué características hacen que un accidente sea fatal.
        """)
        st.info("**¿Qué características del vehículo y del contexto determinan la probabilidad de un desenlace fatal?**")

        st.subheader(" Objetivo General")
        st.write("Detectar los patrones y características de los vehículos que pueden causar un accidente de mayor severidad, mediante técnicas de Machine Learning aplicadas a datos del RUNT Colombia.")

        st.subheader("Objetivos específicos")
        st.write("1.  Analizar la relación entre tipo/marca del vehículo y la gravedad del accidente")
        st.write("2.  Evaluar el impacto de la antigüedad del vehículo en la severidad del siniestro")
        st.write("3.  Identificar patrones espaciales y temporales asociados a la accidentalidad vial")

    with col_b:
        st.subheader(" Desbalanceo de Clases")
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Con Heridos (95%)", "Con Muertos (5%)"],
            values=[95, 5],
            hole=0.55,
            marker_colors=["#AED6EF", "#065A82"],
            textinfo="label+percent",
            textfont_size=13
        )])
        fig_pie.update_layout(showlegend=False, height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.warning("Un modelo que prediga siempre 'Con Heridos' tendría 95% de accuracy pero detectaría **cero muertes**.")

    st.divider()
    st.subheader(" Marco Teórico")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**OMS (2004) — Sistema Seguro**\n\nLos accidentes fatales son evitables mediante el diseño adecuado del sistema de movilidad.")
    with c2:
        st.info("**Fernández et al. (2018)**\n\nEl desbalanceo de clases genera modelos sesgados hacia la clase mayoritaria.")
    with c3:
        st.info("**Mannering et al. (2016)**\n\nLa predicción de severidad permite anticipar escenarios de alto riesgo y orientar políticas públicas.")


# ── OBJETIVO 1 ───────────────────────────────────────────────────────────────
elif seccion == " Objetivo 1: Características Vehículos":
    st.title(" Objetivo 1: Tipo y Marca de Vehículo")
    st.write("Relación entre características del vehículo y la gravedad del accidente")
    st.divider()

    col1, col2 = st.columns([1.3, 1])
    with col1:
        df_sorted = tipo_vehiculo.sort_values("Tasa_Mortalidad", ascending=True)
        colores = ["#AED6EF" if t < 7.5 else "#065A82" for t in df_sorted["Tasa_Mortalidad"]]
        fig = go.Figure(go.Bar(
            x=df_sorted["Tasa_Mortalidad"], y=df_sorted["Tipo"],
            orientation="h", marker_color=colores,
            text=[f"{v:.1f}%" for v in df_sorted["Tasa_Mortalidad"]],
            textposition="outside"
        ))
        fig.add_vline(x=5.0, line_dash="dash", line_color="red", annotation_text="Promedio global (5%)")
        fig.update_layout(title="Tasa de mortalidad (%) por tipo de vehículo", height=420, margin=dict(r=80))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(" Hallazgos clave")
        st.error(" **Carga pesada lidera la mortalidad**\nTractocamión, volqueta y camión registran tasas 3–6× el promedio global.")
        st.warning(" **Paradoja volumen vs. letalidad**\nLas motocicletas tienen el mayor volumen pero tasa cercana al promedio.")
        st.success(" **Automóviles son más seguros**\nLas tasas más bajas corresponden a automóviles y camionetas.")
        st.info(" **V de Cramér**\nEl TIPO tiene asociación moderada-fuerte. La MARCA tiene asociación débil.")

    st.divider()
    st.subheader("Volumen de accidentes vs. Tasa de mortalidad")
    fig2 = px.scatter(tipo_vehiculo, x="Volumen", y="Tasa_Mortalidad",
                      size="Volumen", color="Tasa_Mortalidad", text="Tipo",
                      color_continuous_scale="Blues", size_max=60)
    fig2.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="Promedio global")
    fig2.update_traces(textposition="top center")
    fig2.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader(" Marca de Vehículo")
    st.info("La marca del vehículo tiene asociación **débil** con la gravedad. El riesgo lo determina el **tipo**, no el fabricante.")
    marcas = pd.DataFrame({
        "Marca": ["BAJAJ","YAMAHA","AKT","HONDA","CHEVROLET","RENAULT","HYUNDAI","SUZUKI","KIA","OTROS"],
        "Tasa_Mortalidad": [4.8,4.6,4.7,4.5,4.1,3.9,3.8,4.9,3.7,5.8]
    }).sort_values("Tasa_Mortalidad", ascending=True)
    fig3 = go.Figure(go.Bar(
        x=marcas["Tasa_Mortalidad"], y=marcas["Marca"],
        orientation="h", marker_color="#AED6EF",
        text=[f"{v:.1f}%" for v in marcas["Tasa_Mortalidad"]], textposition="outside"
    ))
    fig3.add_vline(x=5.0, line_dash="dash", line_color="red", annotation_text="Promedio global")
    fig3.update_layout(title="Tasa de mortalidad por marca (Top 10)", height=380, margin=dict(r=80))
    st.plotly_chart(fig3, use_container_width=True)


# ── OBJETIVO 2 ───────────────────────────────────────────────────────────────
elif seccion == " Objetivo 2: Antigüedad":
    st.title(" Objetivo 2: Antigüedad del Vehículo")
    st.write("Impacto de la edad del vehículo en la severidad del accidente")
    st.divider()

    col1, col2 = st.columns([1.3, 1])
    with col1:
        colores_ant = ["#AED6EF","#7EC8E3","#065A82","#032D60"]
        fig = go.Figure(go.Bar(
            x=antiguedad["Categoria"], y=antiguedad["Tasa_Mortalidad"],
            marker_color=colores_ant,
            error_y=dict(type="data", symmetric=False,
                         array=(antiguedad["IC_sup"]-antiguedad["Tasa_Mortalidad"]).tolist(),
                         arrayminus=(antiguedad["Tasa_Mortalidad"]-antiguedad["IC_inf"]).tolist(),
                         color="black"),
            text=[f"{v:.1f}%" for v in antiguedad["Tasa_Mortalidad"]], textposition="outside"
        ))
        fig.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="Promedio global (5%)")
        fig.update_layout(title="Tasa de mortalidad por antigüedad del vehículo<br><sup>Con intervalos de confianza Wilson 95%</sup>",
                          yaxis_title="Tasa de mortalidad (%)", height=420)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Bar(
            x=antiguedad["Categoria"], y=antiguedad["N"],
            marker_color=colores_ant,
            text=[f"{v:,}" for v in antiguedad["N"]], textposition="outside"
        ))
        fig2.update_layout(title="Volumen de registros por antigüedad", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader(" Hallazgos clave")
        st.write("** Gradiente de riesgo monótono**")
        st.write("A mayor antigüedad del vehículo, mayor tasa de mortalidad.")
        st.write("** Vehículos >20 años: máximo riesgo**")
        st.write("Tasa de 9.4% — casi el doble del promedio. Sus IC no se solapan con los de vehículos nuevos.")
        st.write("** Vehículos 0-5 años: protección real**")
        st.write("Tasa por debajo del promedio gracias a ABS, airbags y estructuras de deformación programada.")
        st.divider()
        st.write("** Prueba Mann-Whitney U**")
        st.code("p = 1.13e-65\n→ Diferencia significativa")
        st.divider()
        st.subheader("Estadística por gravedad")
        df_stats = pd.DataFrame({
            "": ["Media edad", "Mediana edad"],
            "Con Heridos": ["9.8 años", "9 años"],
            "Con Muertos": ["11.6 años", "11 años"]
        })
        st.dataframe(df_stats, hide_index=True, use_container_width=True)


# ── OBJETIVO 3 ───────────────────────────────────────────────────────────────
elif seccion == " Objetivo 3: Geografía & Tiempo":
    st.title(" Objetivo 3: Geografía y Temporalidad")
    st.write("Concentración espacial y patrones temporales de los accidentes")
    st.divider()

    st.subheader(" Análisis Geográfico")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        dep_sorted = departamentos.sort_values("Tasa_Mortalidad", ascending=True)
        colores_dep = ["#AED6EF" if t < 6 else "#065A82" for t in dep_sorted["Tasa_Mortalidad"]]
        fig_dep = go.Figure(go.Bar(
            x=dep_sorted["Tasa_Mortalidad"], y=dep_sorted["Departamento"],
            orientation="h", marker_color=colores_dep,
            text=[f"{v:.1f}%" for v in dep_sorted["Tasa_Mortalidad"]], textposition="outside"
        ))
        fig_dep.add_vline(x=5.0, line_dash="dash", line_color="red", annotation_text="Promedio global")
        fig_dep.update_layout(title="Tasa de mortalidad (%) por departamento", height=500, margin=dict(r=80))
        st.plotly_chart(fig_dep, use_container_width=True)

    with col2:
        st.subheader("Paradoja volumen vs. letalidad")
        fig_b = px.scatter(departamentos, x="Total_Accidentes", y="Tasa_Mortalidad",
                           size="Total_Accidentes", color="Tasa_Mortalidad",
                           text="Departamento", color_continuous_scale="Blues", size_max=50)
        fig_b.add_hline(y=5.0, line_dash="dash", line_color="red")
        fig_b.update_traces(textposition="top center", textfont_size=9)
        fig_b.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_b, use_container_width=True)
        st.info("Los departamentos con **mayor volumen** (Antioquia, Bogotá, Valle) tienen tasas **por debajo** del promedio. Los **periféricos** concentran la mayor mortalidad.")

    st.divider()
    st.subheader(" Análisis Temporal")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_mes = make_subplots(specs=[[{"secondary_y": True}]])
        fig_mes.add_trace(go.Bar(x=meses["Mes"], y=meses["Accidentes"], name="Accidentes", marker_color="#AED6EF"), secondary_y=False)
        fig_mes.add_trace(go.Scatter(x=meses["Mes"], y=meses["Tasa_Mortalidad"], name="Tasa (%)", mode="lines+markers", marker_color="#065A82"), secondary_y=True)
        fig_mes.update_layout(title="Accidentalidad mensual", height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_mes, use_container_width=True)

    with col_b:
        colores_dias = ["#065A82" if d in ["Viernes","Sábado","Domingo"] else "#AED6EF" for d in dias["Dia"]]
        fig_dia = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dia.add_trace(go.Bar(x=dias["Dia"], y=dias["Accidentes"], name="Accidentes", marker_color=colores_dias), secondary_y=False)
        fig_dia.add_trace(go.Scatter(x=dias["Dia"], y=dias["Tasa_Mortalidad"], name="Tasa (%)", mode="lines+markers", marker_color="#E63946"), secondary_y=True)
        fig_dia.update_layout(title="Accidentalidad por día<br><sup>Azul oscuro = fin de semana</sup>", height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_dia, use_container_width=True)

    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.success("** Julio: mayor volumen**\nVacaciones de mitad de año concentran el mayor número de accidentes.")
    col_c2.warning("** Madrugada: mayor mortalidad**\nMenor tráfico → mayor velocidad → mayor probabilidad fatal.")
    col_c3.error("** Anomalía julio 2024**\n22,667 registros inválidos de STRIA Medellín fueron excluidos.")


# ── MODELOS ML ───────────────────────────────────────────────────────────────
elif seccion == "Modelos":
    st.title(" Modelos de Machine Learning y Deep Learning")
    st.write("Comparación de 9 modelos bajo desbalanceo severo (95/5)")
    st.divider()

    st.subheader(" Ranking de modelos")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        mod_sorted = modelos.sort_values("F1_Macro", ascending=True)
        n = len(mod_sorted)
        colores_mod = []
        for i in range(n):
            if i == n-1: colores_mod.append("#FAC775")
            elif i == n-2: colores_mod.append("#D3D1C7")
            elif i == n-3: colores_mod.append("#F0997B")
            else: colores_mod.append("#AED6EF")
        fig_rank = go.Figure(go.Bar(
            x=mod_sorted["F1_Macro"], y=mod_sorted["Modelo"],
            orientation="h", marker_color=colores_mod,
            text=[f"{v:.4f}" for v in mod_sorted["F1_Macro"]], textposition="outside"
        ))
        fig_rank.update_layout(title="F1-Macro por modelo", xaxis=dict(range=[0.45, 0.70]), height=420, margin=dict(r=80))
        st.plotly_chart(fig_rank, use_container_width=True)

    with col2:
        st.subheader(" Top 3")
        st.success("🥇 **Balanced Random Forest** — F1-Macro: 0.6405")
        st.info("🥈 **XGBoost** — F1-Macro: 0.6262")
        st.warning("🥉 **Balanced Bagging + LightGBM** — F1-Macro: 0.6251")
        st.divider()
        st.write("** ¿Por qué no usamos Accuracy?**")
        st.write("Un modelo que prediga siempre 'Con Heridos' tendría 95% de accuracy pero detectaría 0% de muertes. El F1-Macro penaliza por igual los errores en ambas clases.")

    st.divider()
    st.subheader(" Tabla completa de resultados")
    df_display = modelos.rename(columns={
        "F1_Macro":"F1-Macro","F1_Muertos":"F1-MUE",
        "Recall_Muertos":"Rec-MUE","ROC_AUC":"ROC-AUC","PR_AUC":"PR-AUC"
    })
    st.dataframe(df_display.style.highlight_max(subset=["F1-Macro","F1-MUE","Rec-MUE","MCC","ROC-AUC","PR-AUC"],
                 color="#E1F5EE").format("{:.4f}", subset=["F1-Macro","F1-MUE","Rec-MUE","MCC","ROC-AUC","PR-AUC"]),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(" Comparación por métrica")
    metrica = st.selectbox("Selecciona métrica:", ["F1_Macro","F1_Muertos","Recall_Muertos","MCC","ROC_AUC","PR_AUC"],
                           format_func=lambda x: x.replace("_","-"))
    fig_met = px.bar(modelos.sort_values(metrica, ascending=False), x="Modelo", y=metrica,
                     color=metrica, color_continuous_scale="Blues",
                     text=modelos.sort_values(metrica, ascending=False)[metrica].apply(lambda x: f"{x:.4f}"))
    fig_met.update_traces(textposition="outside")
    fig_met.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False, height=430)
    st.plotly_chart(fig_met, use_container_width=True)

    st.divider()
    st.subheader(" Pipeline Metodológico (CRISP-DM)")
    with st.expander("1. Ingeniería de variables"):
        st.write("- ANTIGÜEDAD_VEHICULO (tramos de riesgo)")
        st.write("- MES_SIN / MES_COS (codificación cíclica)")
        st.write("- PERFIL_MASA (5 categorías físicas)")
        st.write("- ES_FIN_SEMANA / ES_VACACIONES")
        st.write("- Target encoding con suavizado bayesiano (municipio, marca, autoridad)")
    with st.expander("2. Partición estratificada"):
        st.write("- 80% entrenamiento — 20% prueba")
        st.write("- stratify=y → preserva proporción 95/5")
        st.write("- Anti-leakage: features post-split calculadas SOLO sobre y_train")
    with st.expander("3. Preprocesador"):
        st.write("- Numéricas (12): imputación mediana + StandardScaler")
        st.write("- Categóricas (2): imputación moda + OrdinalEncoder")
        st.write("- Se descartó OHE por explosión de dimensiones (~2,914 columnas)")
    with st.expander("4. Ajuste de umbral (Dicotomía)"):
        st.write("- Búsqueda del τ* que maximiza F1-Muertos sobre la curva PR")
        st.write("- El umbral estándar (0.5) detecta ~0% de muertes")
        st.write("- El umbral óptimo detecta hasta el 45% de muertes reales")


# ── CONCLUSIONES ─────────────────────────────────────────────────────────────
elif seccion == "Conclusiones":
    st.title("Conclusiones")
    st.divider()

    conclusiones = [
        ("Predictibilidad estadística", "La severidad de los accidentes en Colombia es estadísticamente predecible a partir de variables vehiculares, geográficas y temporales del RUNT."),
        ("Tipo de vehículo: predictor más fuerte", "Los vehículos de carga pesada registran tasas 3–6× el promedio. Las motocicletas tienen alto volumen pero tasa cercana al promedio."),
        ("Antigüedad amplifica la mortalidad", "Vehículos >20 años tienen el doble del riesgo que vehículos nuevos. Los sistemas de seguridad pasiva son determinantes."),
        ("Paradoja geográfica volumen/letalidad", "Los departamentos con más accidentes tienen tasas más bajas. Los periféricos concentran la mayor mortalidad por déficit hospitalario."),
        ("Patrones temporales predecibles", "Julio, diciembre y fines de semana concentran los picos. Las madrugadas tienen menor tráfico pero mayor velocidad y mortalidad."),
        ("Balanced Random Forest: mejor modelo", "F1-Macro: 0.64. Los métodos de ensamble con manejo explícito del desbalanceo superan a los modelos tradicionales."),
        ("El umbral de decisión es política pública", "El ajuste del umbral determina cuántas falsas alarmas se acepta a cambio de no perder muertes — decisión del tomador de decisiones, no del modelo."),
    ]

    for i in range(0, len(conclusiones), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(conclusiones):
                titulo, desc = conclusiones[i+j]
                col.info(f"**{titulo}**\n\n{desc}")

    st.divider()
    col_lim, col_fut = st.columns(2)

    st.divider()
    st.subheader(" Referencias principales")
    refs = [
        "OMS (2004). *World Report on Road Traffic Injury Prevention.*",
        "Fernández et al. (2018). *Learning from Imbalanced Data Sets.* Springer.",
        "Mannering et al. (2016). *Unobserved heterogeneity and highway accident data.*",
        "Castellanos et al. (2024). *Predictive modelling of traffic accidents in Bogotá.*",
        "Mora Chacón et al. (2025). *Analysis of road accidents in Colombia.*",
        "Cunha et al. (2025). *Predicting traffic accident severity in Portugal.*",
    ]
    for r in refs:
        st.write(f"- {r}")
