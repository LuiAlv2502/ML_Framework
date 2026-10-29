"""Tema visual inspirado en Cyberpunk 2077 para Streamlit y Plotly."""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Paleta Cyberpunk 2077: amarillo de la marca, cian/rojo del HUD y neones de Night City
AMARILLO = "#FCEE0A"
CIAN = "#00F0FF"
TURQUESA = "#55EAD4"
ROJO = "#FF003C"
ROJO_OSCURO = "#C5003C"
MAGENTA = "#FF2A6D"
ROSA_NEON = "#EA00D9"
VIOLETA = "#711C91"
VERDE = "#00FF9F"
NARANJA = "#FF9E00"

FONDO = "#0A0A12"
PANEL = "#12121F"
REJILLA = "#262640"
TEXTO = "#E6E6F0"
TEXTO_SUAVE = "#9A9AB8"

SECUENCIA = [CIAN, AMARILLO, MAGENTA, VERDE, ROSA_NEON, NARANJA, TURQUESA, ROJO, VIOLETA]
ESCALA_CONTINUA = [
    [0.0, "#0D0221"],
    [0.25, VIOLETA],
    [0.5, ROSA_NEON],
    [0.75, MAGENTA],
    [1.0, AMARILLO],
]
ESCALA_DIVERGENTE = [
    [0.0, CIAN],
    [0.5, PANEL],
    [1.0, ROJO],
]

FUENTE = "Rajdhani, 'Segoe UI', sans-serif"
FUENTE_TITULO = "Orbitron, Rajdhani, sans-serif"


def _ejes():
    return dict(
        gridcolor=REJILLA,
        linecolor=CIAN,
        zerolinecolor=AMARILLO,
        zerolinewidth=1,
        tickcolor=REJILLA,
        title=dict(font=dict(color=CIAN)),
        showline=True,
    )


def _ejes_3d():
    return dict(
        backgroundcolor=PANEL,
        gridcolor=REJILLA,
        zerolinecolor=AMARILLO,
        linecolor=CIAN,
        showbackground=True,
        title=dict(font=dict(color=CIAN)),
    )


_plantilla = go.layout.Template(
    layout=go.Layout(
        colorway=SECUENCIA,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(family=FUENTE, color=TEXTO, size=14),
        title=dict(font=dict(family=FUENTE_TITULO, color=AMARILLO, size=18), x=0.01),
        xaxis=_ejes(),
        yaxis=_ejes(),
        scene=dict(xaxis=_ejes_3d(), yaxis=_ejes_3d(), zaxis=_ejes_3d()),
        legend=dict(
            bgcolor="rgba(18,18,31,0.85)",
            bordercolor=REJILLA,
            borderwidth=1,
            font=dict(color=TEXTO),
        ),
        hoverlabel=dict(
            bgcolor=FONDO,
            bordercolor=AMARILLO,
            font=dict(family=FUENTE, color=TEXTO),
        ),
        coloraxis=dict(
            colorscale=ESCALA_CONTINUA,
            colorbar=dict(outlinecolor=REJILLA, tickfont=dict(color=TEXTO)),
        ),
        colorscale=dict(sequential=ESCALA_CONTINUA, diverging=ESCALA_DIVERGENTE),
        bargap=0.25,
        margin=dict(l=40, r=20, t=60, b=40),
    ),
    data=dict(
        bar=[go.Bar(marker=dict(line=dict(color=FONDO, width=1)))],
        histogram=[go.Histogram(marker=dict(line=dict(color=FONDO, width=1)))],
        scatter=[go.Scatter(marker=dict(line=dict(color=FONDO, width=0.5)))],
    ),
)

pio.templates["cyberpunk"] = _plantilla
pio.templates.default = "cyberpunk"


def grafico(figura, key=None, height=None):
    """Muestra una figura Plotly con el tema neón (sin el tema de Streamlit)."""
    if height is not None:
        figura.update_layout(height=height)
    st.plotly_chart(figura, width="stretch", theme=None, key=key)


def brillo(figura, color=CIAN):
    """Añade un halo neón detrás de las trazas de línea de la figura."""
    originales = len(figura.data)
    for traza in list(figura.data):
        if getattr(traza, "mode", None) and "lines" in traza.mode:
            ancho = (traza.line.width or 2) * 4
            figura.add_trace(
                go.Scatter(
                    x=traza.x,
                    y=traza.y,
                    mode="lines",
                    line=dict(color=traza.line.color or color, width=ancho),
                    opacity=0.15,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=traza.legendgroup or traza.name,
                )
            )
    for traza in figura.data[:originales]:
        traza.legendgroup = traza.legendgroup or traza.name
    figura.data = figura.data[originales:] + figura.data[:originales]
    return figura


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, label, p, li {{
    font-family: {FUENTE};
}}

.stApp {{
    background:
        radial-gradient(circle at 15% 0%, rgba(0, 240, 255, 0.07), transparent 40%),
        radial-gradient(circle at 100% 100%, rgba(234, 0, 217, 0.08), transparent 45%),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0 1px, transparent 1px 3px),
        {FONDO};
}}

h1 {{
    font-family: {FUENTE_TITULO} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {AMARILLO} !important;
    text-shadow: 0 0 6px rgba(252, 238, 10, 0.55), 3px 3px 0 {ROJO_OSCURO};
    border-left: 6px solid {CIAN};
    padding-left: 0.6rem !important;
}}

h2 {{
    font-family: {FUENTE_TITULO} !important;
    color: {CIAN} !important;
    letter-spacing: 0.05em;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.45);
    border-bottom: 1px solid {REJILLA};
}}

h3, h4 {{
    font-family: {FUENTE_TITULO} !important;
    color: {MAGENTA} !important;
    letter-spacing: 0.04em;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D0221 0%, {FONDO} 70%);
    border-right: 2px solid {ROSA_NEON};
    box-shadow: 4px 0 18px rgba(234, 0, 217, 0.25);
}}

[data-testid="stMetric"] {{
    background: linear-gradient(135deg, rgba(0,240,255,0.08), rgba(18,18,31,0.9));
    border: 1px solid {REJILLA};
    border-left: 4px solid {AMARILLO};
    padding: 0.6rem 0.9rem;
    clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
}}

[data-testid="stMetricValue"] {{
    font-family: {FUENTE_TITULO};
    color: {CIAN};
    text-shadow: 0 0 6px rgba(0, 240, 255, 0.5);
}}

[data-testid="stMetricLabel"] p {{
    color: {TEXTO_SUAVE};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.stButton > button, .stDownloadButton > button {{
    font-family: {FUENTE_TITULO};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border: 1px solid {CIAN};
    color: {CIAN};
    background: transparent;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
    transition: all 0.15s ease-in-out;
}}

.stButton > button:hover, .stDownloadButton > button:hover {{
    background: rgba(0, 240, 255, 0.12);
    color: {AMARILLO};
    border-color: {AMARILLO};
    box-shadow: 0 0 12px rgba(252, 238, 10, 0.5);
}}

.stButton > button[kind="primary"] {{
    background: {AMARILLO};
    color: #000 !important;
    border: none;
    font-weight: 700;
}}

.stButton > button[kind="primary"]:hover {{
    background: {CIAN};
    box-shadow: 0 0 16px rgba(0, 240, 255, 0.7);
}}

[data-baseweb="tab-list"] {{
    gap: 0.3rem;
    border-bottom: 1px solid {REJILLA};
}}

[data-baseweb="tab"] {{
    font-family: {FUENTE_TITULO};
    letter-spacing: 0.04em;
    background: rgba(18, 18, 31, 0.7);
    padding: 0.4rem 0.9rem !important;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
}}

[data-baseweb="tab"][aria-selected="true"] {{
    background: rgba(252, 238, 10, 0.12);
    color: {AMARILLO} !important;
}}

[data-baseweb="tab-highlight"] {{
    background-color: {AMARILLO} !important;
    box-shadow: 0 0 10px {AMARILLO};
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: {REJILLA} !important;
}}

[data-testid="stExpander"] details {{
    border-color: {REJILLA};
}}

[data-testid="stPlotlyChart"] {{
    border: 1px solid {REJILLA};
    border-top: 2px solid {CIAN};
    box-shadow: 0 0 14px rgba(0, 240, 255, 0.08);
    padding: 0.3rem;
}}

code {{
    color: {VERDE} !important;
}}
</style>
"""


def aplicar_estilo():
    st.markdown(_CSS, unsafe_allow_html=True)
