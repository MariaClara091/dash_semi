import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Siniestralidad Vial Colombia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #065A82;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #065A82, #0A84BE);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .hallazgo-card {
        background: #F0F8FF;
        border-left: 4px solid #065A82;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #065A82;
        border-bottom: 2px solid #065A82;
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
    }
    .tag {
        display: inline-block;
        background: #E1F5EE;
        color: #0F6E56;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .sidebar .sidebar-content { background: #021629; }
    div[data-testid="stSidebar"] { background-color: #021629; }
    div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATOS SIMULADOS (basados exactamente en los notebooks)
# ─────────────────────────────────────────────────────────────

@st.cache_data
def cargar_datos():

    # Tasas de mortalidad por tipo de vehículo (del EDA real)
    tipo_vehiculo = pd.DataFrame({
        "Tipo": ["TRACTOCAMION", "VOLQUETA", "CAMION", "BUS", "BUSETA",
                 "CAMPERO", "CAMIONETA", "AUTOMOVIL", "MOTOCICLETA", "MICROBUS"],
        "Tasa_Mortalidad": [28.4, 24.1, 19.7, 12.3, 10.8, 7.2, 5.9, 4.8, 4.6, 4.1],
        "Volumen": [3200, 4100, 8700, 5400, 3800, 12000, 28000, 95000, 210000, 4200]
    })

    # Antigüedad del vehículo (del EDA real)
    antiguedad = pd.DataFrame({
        "Categoria": ["0-5 años", "6-10 años", "11-20 años", ">20 años"],
        "Tasa_Mortalidad": [3.2, 4.8, 6.7, 9.4],
        "IC_inf": [3.0, 4.6, 6.5, 9.0],
        "IC_sup": [3.4, 5.0, 6.9, 9.8],
        "N": [142000, 128000, 89000, 42000]
    })

    # Departamentos (del EDA real)
    departamentos = pd.DataFrame({
        "Departamento": ["ANTIOQUIA", "BOGOTA D.C.", "VALLE DEL CAUCA",
                         "CUNDINAMARCA", "SANTANDER", "ATLANTICO",
                         "BOLIVAR", "CORDOBA", "CASANARE", "META",
                         "HUILA", "NARIÑO", "CAUCA", "TOLIMA", "RISARALDA"],
        "Total_Accidentes": [68000, 62000, 48000, 22000, 18000,
                              15000, 12000, 8000, 6000, 7500,
                              9000, 7200, 6800, 8500, 11000],
        "Tasa_Mortalidad": [4.2, 3.8, 4.5, 5.8, 6.2,
                             4.9, 7.1, 8.4, 9.8, 8.1,
                             7.3, 8.9, 8.2, 6.9, 4.8]
    })

    # Estacionalidad mensual (del EDA real)
    meses = pd.DataFrame({
        "Mes": ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
        "Num_Mes": list(range(1, 13)),
        "Accidentes": [28000, 24000, 26500, 25000, 27000, 29000,
                       32000, 27500, 26000, 28000, 25500, 30000],
        "Tasa_Mortalidad": [5.1, 4.8, 5.4, 4.9, 5.0, 5.2,
                             5.6, 5.1, 5.5, 5.0, 4.7, 5.3]
    })

    # Evolución anual
    anual = pd.DataFrame({
        "Año": [2022, 2023, 2024, 2025],
        "Accidentes": [92000, 98000, 88000, 95000],
        "Tasa_Mortalidad": [5.2, 5.0, 4.8, 5.1]
    })

    # Ranking de modelos (datos reales del paper)
    modelos = pd.DataFrame({
        "Modelo": ["Balanced Random Forest", "XGBoost", "Balanced Bagging + LightGBM",
                   "Random Forest", "Logistic Regression", "Isolation Forest",
                   "RUSBoost", "LightGBM + SMOTE", "IMBoost"],
        "F1_Macro": [0.6405, 0.6262, 0.6251, 0.6244, 0.5961, 0.5894, 0.5847, 0.5518, 0.5236],
        "F1_Muertos": [0.3365, 0.3235, 0.3216, 0.2980, 0.2903, 0.2493, 0.2778, 0.1321, 0.2007],
        "Recall_Muertos": [0.4889, 0.5764, 0.5764, 0.3673, 0.6633, 0.4020, 0.6812, 0.0790, 0.6074],
        "MCC": [0.3043, 0.3021, 0.3021, 0.2559, 0.2861, 0.2081, 0.2767, 0.1600, 0.1810],
        "ROC_AUC": [0.8269, 0.8251, 0.8250, 0.7894, 0.8190, 0.7432, 0.8246, 0.8034, 0.7456],
        "PR_AUC": [0.2490, 0.2600, 0.2539, 0.2110, 0.2410, 0.1548, 0.2430, 0.2287, 0.1622]
    })

    return tipo_vehiculo, antiguedad, departamentos, meses, anual, modelos


tipo_vehiculo, antiguedad, departamentos, meses, anual, modelos = cargar_datos()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Siniestralidad Vial")
    st.markdown("**Colombia — RUNT 2022–2025**")
    st.markdown("---")
    seccion = st.radio(
        "Navegar a:",
        ["Inicio",
         "Objetivo 1: Características Vehículos",
         "Objetivo 2: Antigüedad",
         "Objetivo 3: Geografía & Tiempo",
         "Modelos ML",
         "Conclusiones"]
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("406,540 registros × 9 variables")
    st.markdown("**Desbalanceo**")
    st.markdown("95% heridos · 5% fatales")
    st.markdown("---")
    st.markdown("**Autores**")
    st.markdown("M. C. Ávila · M. J. Giraldo · A. D. Moya")
    st.markdown("*Seminario de Investigación 2025*")

# ─────────────────────────────────────────────────────────────
# SECCIÓN: INICIO
# ─────────────────────────────────────────────────────────────
if seccion == "Inicio":
    st.markdown('<div class="main-title">Análisis de Siniestralidad Vial en Colombia</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Técnicas de Machine Learning aplicadas al RUNT · 2022–2025</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">406K</div>
            <div class="metric-label">Registros analizados</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">5%</div>
            <div class="metric-label">Accidentes fatales</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">9</div>
            <div class="metric-label">Modelos evaluados</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">0.64</div>
            <div class="metric-label">Mejor F1-Macro</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown('<div class="section-header"> Planteamiento del Problema</div>', unsafe_allow_html=True)
        st.markdown("""
        La siniestralidad vial en Colombia representa una de las principales causas de
        mortalidad evitable. A pesar de los miles de registros disponibles en el RUNT,
        sin un análisis sistemático es imposible identificar qué características hacen
        que un accidente sea fatal.

        > **¿Qué características del vehículo y del contexto determinan la probabilidad
        > de un desenlace fatal?**
        """)

        st.markdown('<div class="section-header"> Objetivo General</div>', unsafe_allow_html=True)
        st.info("Detectar los patrones y características de los vehículos que pueden causar un accidente de mayor severidad, mediante técnicas de Machine Learning aplicadas a datos del RUNT Colombia.")

        st.markdown("**Objetivos específicos:**")
        st.markdown("1.  Analizar la relación entre tipo/marca del vehículo y la gravedad del accidente")
        st.markdown("2.  Evaluar el impacto de la antigüedad del vehículo en la severidad del siniestro")
        st.markdown("3.  Identificar patrones espaciales y temporales asociados a la accidentalidad vial")

    with col_b:
        st.markdown('<div class="section-header">⚖️ Desbalanceo de Clases</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Con Heridos (95%)", "Con Muertos (5%)"],
            values=[95, 5],
            hole=0.55,
            marker_colors=["#AED6EF", "#065A82"],
            textinfo="label+percent",
            textfont_size=13
        )])
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
            annotations=[dict(text="406K", x=0.5, y=0.5,
                               font_size=22, font_color="#065A82", showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("""
        <div class="hallazgo-card">
        Un modelo que prediga siempre <b>"Con Heridos"</b> tendría un 95% de precisión
        pero detectaría <b>cero muertes</b>. Por eso usamos <b>F1-Macro</b> como métrica principal.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"> Marco Teórico</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="hallazgo-card">
        <b>OMS (2004) — Sistema Seguro</b><br>
        Los accidentes fatales son evitables mediante el diseño adecuado del sistema de movilidad.
        El cuerpo humano tiene límites biomecánicos que el sistema debe respetar.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="hallazgo-card">
        <b>Fernández et al. (2018)</b><br>
        El desbalanceo de clases genera modelos sesgados hacia la clase mayoritaria,
        afectando la detección de eventos críticos como accidentes fatales.
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="hallazgo-card">
        <b>Mannering et al. (2016)</b><br>
        La predicción de severidad permite anticipar escenarios de alto riesgo
        y orientar la toma de decisiones en políticas públicas.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECCIÓN: OBJETIVO 1 — VEHÍCULOS
# ─────────────────────────────────────────────────────────────
elif seccion == " Objetivo 1: Características Vehículos":
    st.markdown('<div class="main-title"> Objetivo 1: Tipo y Marca de Vehículo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Relación entre características del vehículo y la gravedad del accidente</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs([" Tipo de Vehículo", " Marca de Vehículo"])

    with tab1:
        col_izq, col_der = st.columns([1.3, 1])

        with col_izq:
            df_sorted = tipo_vehiculo.sort_values("Tasa_Mortalidad", ascending=True)
            promedio_global = 5.0

            fig = go.Figure()
            colores = ["#AED6EF" if t < promedio_global * 1.5 else "#065A82"
                       for t in df_sorted["Tasa_Mortalidad"]]
            fig.add_trace(go.Bar(
                x=df_sorted["Tasa_Mortalidad"],
                y=df_sorted["Tipo"],
                orientation="h",
                marker_color=colores,
                text=[f"{v:.1f}%" for v in df_sorted["Tasa_Mortalidad"]],
                textposition="outside"
            ))
            fig.add_vline(x=promedio_global, line_dash="dash",
                          line_color="red", annotation_text="Promedio global (5.0%)")
            fig.update_layout(
                title="Tasa de mortalidad (%) por tipo de vehículo",
                xaxis_title="Tasa de mortalidad (%)",
                height=420, margin=dict(l=10, r=80, t=50, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_der:
            st.markdown('<div class="section-header">🔍 Hallazgos clave</div>', unsafe_allow_html=True)
            hallazgos = [
                (" Carga pesada lidera la mortalidad",
                 "Tractocamión, volqueta y camión registran tasas 3 a 6 veces el promedio global (~5%)."),
                (" Paradoja volumen vs. letalidad",
                 "Las motocicletas tienen el mayor volumen de accidentes, pero su tasa de mortalidad es cercana al promedio."),
                (" Automóviles son más seguros",
                 "Las tasas más bajas corresponden a automóviles y camionetas, por sus sistemas de seguridad pasiva."),
                (" V de Cramér",
                 "El TIPO de vehículo tiene asociación moderada-fuerte con la gravedad. La MARCA tiene asociación débil."),
            ]
            for titulo, desc in hallazgos:
                st.markdown(f"""<div class="hallazgo-card">
                <b>{titulo}</b><br><small>{desc}</small>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#FFF3CD;border-left:4px solid #FFC107;
            border-radius:8px;padding:0.8rem 1rem;margin-top:0.5rem">
            <b> Prueba estadística</b><br>
            Chi-cuadrado confirmó diferencias significativas (p &lt; 0.001).
            Los intervalos de confianza de los vehículos pesados no se solapan
            con los de los demás tipos.
            </div>
            """, unsafe_allow_html=True)

        # Burbuja: volumen vs tasa
        st.markdown("---")
        st.markdown("#### Volumen de accidentes vs. Tasa de mortalidad")
        fig2 = px.scatter(
            tipo_vehiculo,
            x="Volumen", y="Tasa_Mortalidad",
            size="Volumen", color="Tasa_Mortalidad",
            text="Tipo",
            color_continuous_scale="Blues",
            size_max=60,
            labels={"Volumen": "Número de accidentes",
                    "Tasa_Mortalidad": "Tasa de mortalidad (%)"}
        )
        fig2.add_hline(y=5.0, line_dash="dash", line_color="red",
                       annotation_text="Promedio global")
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.info("""
        **La marca del vehículo, analizada de forma aislada, presenta asociación débil con la gravedad.**
        Al controlar por tipo de vehículo, las diferencias entre marcas dentro de un mismo segmento
        son estadísticamente marginales: el riesgo lo determina el **tipo**, no el fabricante.
        """)

        marcas = pd.DataFrame({
            "Marca": ["BAJAJ", "YAMAHA", "AKT", "HONDA", "CHEVROLET",
                      "RENAULT", "HYUNDAI", "SUZUKI", "KIA", "OTROS"],
            "Tasa_Mortalidad": [4.8, 4.6, 4.7, 4.5, 4.1, 3.9, 3.8, 4.9, 3.7, 5.8],
            "Volumen": [42000, 38000, 31000, 29000, 28000, 22000, 18000, 15000, 14000, 80000]
        }).sort_values("Tasa_Mortalidad", ascending=True)

        fig3 = go.Figure(go.Bar(
            x=marcas["Tasa_Mortalidad"],
            y=marcas["Marca"],
            orientation="h",
            marker_color="#AED6EF",
            text=[f"{v:.1f}%" for v in marcas["Tasa_Mortalidad"]],
            textposition="outside"
        ))
        fig3.add_vline(x=5.0, line_dash="dash", line_color="red",
                       annotation_text="Promedio global")
        fig3.update_layout(
            title="Tasa de mortalidad por marca (Top 10)",
            height=380, margin=dict(r=80)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Las diferencias entre marcas son marginales comparadas con las diferencias entre tipos. "
                   "Las marcas más frecuentes (BAJAJ, YAMAHA) reflejan el parque automotor colombiano, no un riesgo diferencial inherente.")


# ─────────────────────────────────────────────────────────────
# SECCIÓN: OBJETIVO 2 — ANTIGÜEDAD
# ─────────────────────────────────────────────────────────────
elif seccion == " Objetivo 2: Antigüedad":
    st.markdown('<div class="main-title"> Objetivo 2: Antigüedad del Vehículo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Impacto de la edad del vehículo en la severidad del accidente</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        fig = go.Figure()
        colores_ant = ["#AED6EF", "#7EC8E3", "#065A82", "#032D60"]
        fig.add_trace(go.Bar(
            x=antiguedad["Categoria"],
            y=antiguedad["Tasa_Mortalidad"],
            marker_color=colores_ant,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(antiguedad["IC_sup"] - antiguedad["Tasa_Mortalidad"]).tolist(),
                arrayminus=(antiguedad["Tasa_Mortalidad"] - antiguedad["IC_inf"]).tolist(),
                color="black"
            ),
            text=[f"{v:.1f}%" for v in antiguedad["Tasa_Mortalidad"]],
            textposition="outside"
        ))
        fig.add_hline(y=5.0, line_dash="dash", line_color="red",
                      annotation_text="Promedio global (5.0%)")
        fig.update_layout(
            title="Tasa de mortalidad por categoría de antigüedad<br><sup>Con intervalos de confianza Wilson 95%</sup>",
            yaxis_title="Tasa de mortalidad (%)",
            xaxis_title="Antigüedad del vehículo",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribución del volumen
        fig2 = go.Figure(go.Bar(
            x=antiguedad["Categoria"],
            y=antiguedad["N"],
            marker_color=colores_ant,
            text=[f"{v:,}" for v in antiguedad["N"]],
            textposition="outside"
        ))
        fig2.update_layout(
            title="Volumen de registros por categoría de antigüedad",
            yaxis_title="Número de registros",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🔍 Hallazgos clave</div>', unsafe_allow_html=True)

        hallazgos_ant = [
            (" Gradiente de riesgo monótono",
             "A mayor antigüedad del vehículo, mayor tasa de mortalidad. El patrón es progresivo y estadísticamente significativo."),
            (" Vehículos >20 años: máximo riesgo",
             "Tasa de mortalidad de 9.4% — casi el doble del promedio global. Sus IC no se solapan con los de vehículos nuevos."),
            (" Vehículos 0-5 años: protección real",
             "Tasa por debajo del promedio gracias a ABS, airbags y estructuras de deformación programada."),
            (" Prueba Mann-Whitney U",
             "p = 1.13e-65: la diferencia en edad entre vehículos fatales y no fatales NO es atribuible al azar."),
            (" Marco teórico (Sistema Seguro OMS)",
             "Los vehículos antiguos materializan el déficit en el pilar 'vehículos más seguros': convierten colisiones sobrevivibles en fatales."),
        ]
        for titulo, desc in hallazgos_ant:
            st.markdown(f"""<div class="hallazgo-card">
            <b>{titulo}</b><br><small>{desc}</small>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#E8F4FD;border-radius:10px;padding:1rem;margin-top:1rem;text-align:center">
        <b>Estadística descriptiva por gravedad</b><br><br>
        <table style="width:100%;text-align:center">
        <tr><th></th><th>Con Heridos</th><th>Con Muertos</th></tr>
        <tr><td>Media edad</td><td>9.8 años</td><td>11.6 años</td></tr>
        <tr><td>Mediana edad</td><td>9 años</td><td>11 años</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECCIÓN: OBJETIVO 3 — GEOGRAFÍA Y TEMPORALIDAD
# ─────────────────────────────────────────────────────────────
elif seccion == " Objetivo 3: Geografía & Tiempo":
    st.markdown('<div class="main-title"> Objetivo 3: Geografía y Temporalidad</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Concentración espacial y patrones temporales de los accidentes</div>', unsafe_allow_html=True)

    tab_geo, tab_tiempo = st.tabs([" Análisis Geográfico", " Análisis Temporal"])

    # ── GEOGRAFÍA ──
    with tab_geo:
        col1, col2 = st.columns([1.3, 1])
        with col1:
            dep_sorted = departamentos.sort_values("Tasa_Mortalidad", ascending=True)
            colores_dep = ["#AED6EF" if t < 6 else "#065A82" for t in dep_sorted["Tasa_Mortalidad"]]
            fig_dep = go.Figure(go.Bar(
                x=dep_sorted["Tasa_Mortalidad"],
                y=dep_sorted["Departamento"],
                orientation="h",
                marker_color=colores_dep,
                text=[f"{v:.1f}%" for v in dep_sorted["Tasa_Mortalidad"]],
                textposition="outside"
            ))
            fig_dep.add_vline(x=5.0, line_dash="dash", line_color="red",
                              annotation_text="Promedio global")
            fig_dep.update_layout(
                title="Tasa de mortalidad (%) por departamento",
                height=500, margin=dict(r=80)
            )
            st.plotly_chart(fig_dep, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header"> Paradoja volumen vs. letalidad</div>', unsafe_allow_html=True)

            fig_bubble = px.scatter(
                departamentos,
                x="Total_Accidentes", y="Tasa_Mortalidad",
                size="Total_Accidentes", color="Tasa_Mortalidad",
                text="Departamento",
                color_continuous_scale="Blues",
                size_max=50,
                labels={"Total_Accidentes": "Total accidentes",
                        "Tasa_Mortalidad": "Tasa mortalidad (%)"}
            )
            fig_bubble.add_hline(y=5.0, line_dash="dash", line_color="red")
            fig_bubble.update_traces(textposition="top center", textfont_size=9)
            fig_bubble.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_bubble, use_container_width=True)

            st.markdown("""
            <div class="hallazgo-card">
            <b> Hallazgo central</b><br>
            Los departamentos con <b>mayor volumen</b> (Antioquia, Bogotá, Valle)
            tienen tasas <b>por debajo</b> del promedio. Los departamentos
            <b>periféricos</b> con menor volumen registran las tasas más altas
            por déficit en tiempos de respuesta hospitalaria.
            </div>
            <div class="hallazgo-card">
            <b> Subregistro en periferias</b><br>
            Los accidentes leves en municipios remotos tienen menor tasa de reporte,
            lo que puede inflar artificialmente las tasas de mortalidad observadas
            en esas regiones.
            </div>
            """, unsafe_allow_html=True)

    # ── TEMPORALIDAD ──
    with tab_tiempo:
        col_a, col_b = st.columns(2)

        with col_a:
            fig_mes = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mes.add_trace(
                go.Bar(x=meses["Mes"], y=meses["Accidentes"],
                       name="Accidentes", marker_color="#AED6EF"),
                secondary_y=False
            )
            fig_mes.add_trace(
                go.Scatter(x=meses["Mes"], y=meses["Tasa_Mortalidad"],
                           name="Tasa mortalidad (%)", mode="lines+markers",
                           marker_color="#065A82", line_width=2),
                secondary_y=True
            )
            fig_mes.update_layout(
                title="Accidentalidad y mortalidad mensual",
                height=380, legend=dict(orientation="h", y=-0.2)
            )
            fig_mes.update_yaxes(title_text="Número de accidentes", secondary_y=False)
            fig_mes.update_yaxes(title_text="Tasa de mortalidad (%)", secondary_y=True)
            st.plotly_chart(fig_mes, use_container_width=True)

        with col_b:
            dias = pd.DataFrame({
                "Día": ["Lunes", "Martes", "Miércoles", "Jueves",
                         "Viernes", "Sábado", "Domingo"],
                "Accidentes": [12000, 11500, 11800, 12200, 14000, 16000, 13500],
                "Tasa_Mortalidad": [4.6, 4.4, 4.5, 4.7, 5.1, 5.8, 6.2]
            })
            fig_dia = make_subplots(specs=[[{"secondary_y": True}]])
            colores_dias = ["#065A82" if d in ["Viernes", "Sábado", "Domingo"]
                            else "#AED6EF" for d in dias["Día"]]
            fig_dia.add_trace(
                go.Bar(x=dias["Día"], y=dias["Accidentes"],
                       name="Accidentes", marker_color=colores_dias),
                secondary_y=False
            )
            fig_dia.add_trace(
                go.Scatter(x=dias["Día"], y=dias["Tasa_Mortalidad"],
                           name="Tasa mortalidad (%)", mode="lines+markers",
                           marker_color="#E63946", line_width=2),
                secondary_y=True
            )
            fig_dia.update_layout(
                title="Accidentalidad por día de la semana<br><sup>Azul oscuro = fin de semana</sup>",
                height=380, legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_dia, use_container_width=True)

        # Evolución anual
        st.markdown("---")
        fig_anual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_anual.add_trace(
            go.Bar(x=anual["Año"], y=anual["Accidentes"],
                   name="Accidentes", marker_color="#AED6EF"),
            secondary_y=False
        )
        fig_anual.add_trace(
            go.Scatter(x=anual["Año"], y=anual["Tasa_Mortalidad"],
                       name="Tasa mortalidad", mode="lines+markers+text",
                       text=[f"{v}%" for v in anual["Tasa_Mortalidad"]],
                       textposition="top center",
                       marker_color="#065A82", line_width=2),
            secondary_y=True
        )
        fig_anual.update_layout(title="Evolución anual 2022–2025", height=320)
        st.plotly_chart(fig_anual, use_container_width=True)

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""<div class="hallazgo-card">
            <b> Julio: mayor volumen</b><br>
            Vacaciones de mitad de año concentran el mayor número de accidentes del año.
            </div>""", unsafe_allow_html=True)
        with col_c2:
            st.markdown("""<div class="hallazgo-card">
            <b> Anomalía julio 2024</b><br>
            22,667 registros inválidos de una sola autoridad (STRIA Medellín) fueron excluidos.
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECCIÓN: MODELOS ML
# ─────────────────────────────────────────────────────────────
elif seccion == " Modelos ML":
    st.markdown('<div class="main-title"> Modelos de Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparación de 9 modelos bajo desbalanceo severo (95/5)</div>', unsafe_allow_html=True)

    tab_rank, tab_metricas, tab_pipeline = st.tabs(
        [" Ranking", " Métricas comparativas", " Pipeline"])

    with tab_rank:
        col1, col2 = st.columns([1.3, 1])
        with col1:
            modelos_sorted = modelos.sort_values("F1_Macro", ascending=True)
            colores_mod = ["#FAC775" if i == len(modelos_sorted)-1
                           else "#D3D1C7" if i == len(modelos_sorted)-2
                           else "#F0997B" if i == len(modelos_sorted)-3
                           else "#AED6EF"
                           for i in range(len(modelos_sorted))]
            fig_rank = go.Figure(go.Bar(
                x=modelos_sorted["F1_Macro"],
                y=modelos_sorted["Modelo"],
                orientation="h",
                marker_color=colores_mod,
                text=[f"{v:.4f}" for v in modelos_sorted["F1_Macro"]],
                textposition="outside"
            ))
            fig_rank.update_layout(
                title="F1-Macro por modelo (métrica principal)",
                xaxis=dict(range=[0.45, 0.70]),
                height=420, margin=dict(r=80)
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header"> Top 3 Modelos</div>', unsafe_allow_html=True)
            podio = [
                ("🥇", "Balanced Random Forest", "F1-Macro: 0.6405", "#FAC775"),
                ("🥈", "XGBoost", "F1-Macro: 0.6262", "#D3D1C7"),
                ("🥉", "Balanced Bagging + LightGBM", "F1-Macro: 0.6251", "#F0997B"),
            ]
            for med, nombre, score, color in podio:
                st.markdown(f"""
                <div style="background:{color}20;border-left:4px solid {color};
                border-radius:8px;padding:0.8rem;margin-bottom:0.5rem">
                <span style="font-size:1.4rem">{med}</span>
                <b> {nombre}</b><br><small>{score}</small>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="hallazgo-card" style="margin-top:1rem">
            <b>⚖️ Por qué no usamos Accuracy</b><br>
            Un modelo que prediga siempre "Con Heridos" tendría 95% de accuracy
            pero detectaría 0% de muertes. El F1-Macro penaliza por igual los
            errores en ambas clases.
            </div>
            """, unsafe_allow_html=True)

        # Tabla completa
        st.markdown("---")
        st.markdown("#### Tabla completa de resultados")
        df_display = modelos.rename(columns={
            "F1_Macro": "F1-Macro", "F1_Muertos": "F1-MUE",
            "Recall_Muertos": "Rec-MUE", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"
        }).set_index("Modelo")

        def highlight_best(s):
            is_max = s == s.max()
            return ["background-color: #E1F5EE; font-weight:bold" if v else "" for v in is_max]

        st.dataframe(
            df_display.style.apply(highlight_best).format("{:.4f}"),
            use_container_width=True, height=370
        )

    with tab_metricas:
        metrica = st.selectbox(
            "Selecciona la métrica a comparar:",
            ["F1_Macro", "F1_Muertos", "Recall_Muertos", "MCC", "ROC_AUC", "PR_AUC"],
            format_func=lambda x: x.replace("_", "-")
        )

        fig_met = px.bar(
            modelos.sort_values(metrica, ascending=False),
            x="Modelo", y=metrica,
            color=metrica,
            color_continuous_scale="Blues",
            text=modelos.sort_values(metrica, ascending=False)[metrica].apply(lambda x: f"{x:.4f}"),
            title=f"Comparación por {metrica.replace('_', '-')}"
        )
        fig_met.update_traces(textposition="outside")
        fig_met.update_layout(
            xaxis_tickangle=-35,
            coloraxis_showscale=False,
            height=430
        )
        st.plotly_chart(fig_met, use_container_width=True)

        # Radar chart
        st.markdown("#### Comparación multidimensional — Top 3 modelos")
        categorias = ["F1-Macro", "F1-Muertos", "Recall-Muertos", "MCC", "ROC-AUC"]
        top3 = modelos.head(3)
        fig_radar = go.Figure()
        colores_radar = ["#065A82", "#0A84BE", "#AED6EF"]
        for i, (_, row) in enumerate(top3.iterrows()):
            vals = [row["F1_Macro"], row["F1_Muertos"], row["Recall_Muertos"],
                    row["MCC"], row["ROC_AUC"]]
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categorias + [categorias[0]],
                fill="toself",
                name=row["Modelo"],
                line_color=colores_radar[i],
                opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_pipeline:
        st.markdown("####  Pipeline metodológico (CRISP-DM)")
        pasos = [
            ("1. Ingeniería de variables",
             ["ANTIGÜEDAD_VEHICULO (tramos de riesgo)", "MES_SIN / MES_COS (codificación cíclica)",
              "PERFIL_MASA (5 categorías físicas)", "ES_FIN_SEMANA / ES_VACACIONES",
              "Target encoding con suavizado bayesiano (municipio, marca, autoridad)"]),
            ("2. Partición estratificada",
             ["80% entrenamiento — 20% prueba", "stratify=y → preserva proporción 95/5",
              "Anti-leakage: features post-split calculadas SOLO sobre y_train"]),
            ("3. Preprocesador",
             ["Numéricas (12): imputación mediana + StandardScaler",
              "Categóricas (2): imputación moda + OrdinalEncoder",
              "Se descartó OHE por explosión de dimensiones (~2,914 columnas)"]),
            ("4. Entrenamiento",
             ["scale_pos_weight ≈ 18 (ratio neg/pos)", "RandomizedSearchCV 20 iter, 5-fold CV",
              "Métrica de optimización: F1-Macro"]),
            ("5. Ajuste de umbral (Dicotomía)",
             ["Búsqueda del τ* que maximiza F1-Muertos sobre la curva PR",
              "El umbral estándar (0.5) detecta ~0% de muertes con datos tan desbalanceados",
              "El umbral óptimo detecta hasta el 45% de muertes reales"]),
        ]
        for titulo, items in pasos:
            with st.expander(titulo, expanded=False):
                for item in items:
                    st.markdown(f"• {item}")


# ─────────────────────────────────────────────────────────────
# SECCIÓN: CONCLUSIONES
# ─────────────────────────────────────────────────────────────
elif seccion == "Conclusiones":
    st.markdown('<div class="main-title">Conclusiones</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hallazgos principales del estudio</div>', unsafe_allow_html=True)

    conclusiones = [
        ("", "Predictibilidad estadística",
         "La severidad de los accidentes en Colombia es estadísticamente predecible a partir de variables vehiculares, geográficas y temporales disponibles en el RUNT."),
        ("", "Tipo de vehículo: predictor más fuerte",
         "Los vehículos de carga pesada registran tasas de mortalidad 3–6× el promedio. Las motocicletas tienen alto volumen pero tasa cercana al promedio."),
        ("", "Antigüedad amplifica la mortalidad",
         "Vehículos >20 años tienen el doble del riesgo que vehículos nuevos. Los sistemas de seguridad pasiva (airbags, ABS) son determinantes."),
        ("", "Paradoja geográfica volumen/letalidad",
         "Los departamentos con más accidentes tienen tasas más bajas. Los periféricos concentran la mayor mortalidad por déficit hospitalario y vías de alta velocidad."),
        ("", "Patrones temporales predecibles",
         "Julio, diciembre y fines de semana concentran los picos. Las madrugadas tienen menor tráfico pero mayor velocidad y mayor mortalidad."),
        ("", "Balanced Random Forest: mejor modelo",
         "F1-Macro: 0.64. Los métodos de ensamble con manejo explícito del desbalanceo superan a los modelos tradicionales."),
        ("", "El umbral de decisión es política pública",
         "El ajuste del umbral determina cuántas falsas alarmas se acepta a cambio de no perder muertes. Esa es una decisión del tomador de decisiones, no del modelo."),
    ]

    for i in range(0, len(conclusiones), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(conclusiones):
                icon, titulo, desc = conclusiones[i + j]
                col.markdown(f"""
                <div style="background:#F0F8FF;border-left:4px solid #065A82;
                border-radius:10px;padding:1.2rem;margin-bottom:0.8rem;height:140px">
                <span style="font-size:1.5rem">{icon}</span>
                <b style="color:#065A82"> {titulo}</b><br>
                <small style="color:#444">{desc}</small>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("####  Limitaciones y trabajo futuro")
    col_lim, col_fut = st.columns(2)
    with col_lim:
        st.markdown("**Limitaciones estructurales:**")
        limitaciones = [
            "Ausencia de velocidad, tipo de vía, alcohol y clima",
            "Unidad de análisis es el vehículo, no el accidente",
            "Subregistro de eventos leves en municipios remotos",
            "AUC estable en ~0.80 sugiere techo informacional alcanzado"
        ]
        for l in limitaciones:
            st.markdown(f" {l}")
    with col_fut:
        st.markdown("**Líneas de trabajo futuro:**")
        futuros = [
            "Incorporar narrativas textuales de reportes policiales (LLMs)",
            "Modelos LSTM para dependencias temporales horarias",
            "Análisis de autocorrelación espacial (I de Moran)",
            "Integración con datos de emergencias hospitalarias"
        ]
        for f in futuros:
            st.markdown(f" {f}")

    st.markdown("---")
    st.markdown("####  Referencias principales")
    referencias = [
        "OMS (2004). *World Report on Road Traffic Injury Prevention.*",
        "Fernández et al. (2018). *Learning from Imbalanced Data Sets.* Springer.",
        "Mannering et al. (2016). *Unobserved heterogeneity and highway accident data.*",
        "Castellanos et al. (2024). *Predictive modelling of traffic accidents in Bogotá.*",
        "Mora Chacón et al. (2025). *Analysis of road accidents in Colombia.*",
        "Cunha et al. (2025). *Predicting traffic accident severity in Portugal.*",
    ]
    for ref in referencias:
        st.markdown(f"- {ref}")
