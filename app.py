"""
app.py — Dashboard Streamlit de análisis técnico price-action.

Correr con:
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import analysis as ta

# ─────────────────────────── Config ───────────────────────────
st.set_page_config(
    page_title="Analista Técnico",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stMetric { background-color: #15151f; padding: 12px; border-radius: 8px; }
    h1, h2, h3 { color: #fafafa; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Data ───────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return df
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


# ─────────────────────────── Sidebar ───────────────────────────
st.sidebar.title("⚙️ Configuración")

ticker = st.sidebar.text_input(
    "Ticker",
    value="AAPL",
    help="Ej: AAPL, TSLA, NVDA, BTC-USD, ETH-USD, YPF, GGAL, GGAL.BA, MELI"
).strip().upper()

with st.sidebar.expander("📌 Tickers populares"):
    st.markdown("""
- **US**: AAPL, TSLA, NVDA, MSFT, GOOGL, META, AMZN
- **Cripto**: BTC-USD, ETH-USD, SOL-USD
- **ADRs Argentinos**: YPF, GGAL, PAM, BMA, TEO
- **BCBA**: YPFD.BA, GGAL.BA, ALUA.BA, PAMP.BA
- **Índices**: ^GSPC (S&P500), ^IXIC (Nasdaq), ^MERV (Merval)
""")

period_opts = {
    "1 mes": "1mo", "3 meses": "3mo", "6 meses": "6mo",
    "1 año": "1y", "2 años": "2y", "5 años": "5y", "Máximo": "max",
}
period = period_opts[st.sidebar.selectbox(
    "Período", list(period_opts.keys()), index=2
)]

interval_opts = {
    "Diario": "1d", "Semanal": "1wk", "Mensual": "1mo",
    "1 hora": "1h", "30 min": "30m", "15 min": "15m",
}
interval = interval_opts[st.sidebar.selectbox(
    "Timeframe", list(interval_opts.keys()), index=0
)]

st.sidebar.markdown("### 🎚️ Sensibilidad")
swing_distance = st.sidebar.slider(
    "Distancia mín. entre swings (velas)", 2, 30, 5,
    help="Más alto = menos swings, sólo los más significativos."
)
swing_prom = st.sidebar.slider(
    "Prominencia mínima (%)", 0.5, 5.0, 1.5, 0.1,
    help="Qué tan grande tiene que ser un movimiento para contar como swing."
) / 100

st.sidebar.markdown("### 👁️ Mostrar")
show_swings = st.sidebar.checkbox("Swings ▲▼", True)
show_levels = st.sidebar.checkbox("Soportes / Resistencias", True)
show_trendlines = st.sidebar.checkbox("Trendlines", True)
show_channel = st.sidebar.checkbox("Canal (si lo hay)", True)
show_fib = st.sidebar.checkbox("Fibonacci", True)
show_patterns = st.sidebar.checkbox("Marcar patrones", True)


# ─────────────────────────── Carga ───────────────────────────
try:
    df = load_data(ticker, period, interval)
except Exception as e:
    st.error(f"❌ Error cargando **{ticker}**: {e}")
    st.stop()

if df.empty:
    st.error(f"❌ No encontré datos para **{ticker}**. Verificá el ticker.")
    st.stop()

if len(df) < 10:
    st.warning(f"⚠️ Sólo {len(df)} velas disponibles. Probá un período más largo.")


# ─────────────────────────── Header ───────────────────────────
last = float(df['Close'].iloc[-1])
prev = float(df['Close'].iloc[-2]) if len(df) > 1 else last
change_pct = (last / prev - 1) * 100 if prev else 0
hi = float(df['High'].max())
lo = float(df['Low'].min())
vol = float(df['Volume'].iloc[-1])

st.title(f"📈 {ticker}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Último", f"${last:,.2f}", f"{change_pct:+.2f}%")
c2.metric("Máximo período", f"${hi:,.2f}", f"{(last/hi - 1)*100:+.1f}% del máx")
c3.metric("Mínimo período", f"${lo:,.2f}", f"{(last/lo - 1)*100:+.1f}% del mín")
c4.metric("Volumen últ.", f"{vol/1e6:.1f}M")


# ─────────────────────────── Análisis ───────────────────────────
swings = ta.detect_swings(df, distance=swing_distance, prominence_pct=swing_prom)
levels = ta.cluster_levels(swings)
channel = ta.detect_channel(swings)
fib = ta.fibonacci_retracements(swings)
patterns = ta.find_patterns(swings)


# ─────────────────────────── Plot ───────────────────────────
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.78, 0.22], vertical_spacing=0.03,
)

# --- Velas ---
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
    name='Precio', showlegend=False,
), row=1, col=1)

# --- Volumen ---
vol_colors = ['#26a69a' if c >= o else '#ef5350'
              for o, c in zip(df['Open'], df['Close'])]
fig.add_trace(go.Bar(
    x=df.index, y=df['Volume'], marker_color=vol_colors,
    opacity=0.6, name='Vol', showlegend=False,
), row=2, col=1)

n = len(df)
x_idx = np.arange(n)

# --- Swings ---
if show_swings and len(swings.high_idx):
    fig.add_trace(go.Scatter(
        x=df.index[swings.high_idx],
        y=swings.high_prices * 1.005,
        mode='markers',
        marker=dict(symbol='triangle-down', size=11, color='#ef5350'),
        name='Máximos', showlegend=False,
        hovertemplate='Máximo: $%{y:.2f}<extra></extra>',
    ), row=1, col=1)
if show_swings and len(swings.low_idx):
    fig.add_trace(go.Scatter(
        x=df.index[swings.low_idx],
        y=swings.low_prices * 0.995,
        mode='markers',
        marker=dict(symbol='triangle-up', size=11, color='#26a69a'),
        name='Mínimos', showlegend=False,
        hovertemplate='Mínimo: $%{y:.2f}<extra></extra>',
    ), row=1, col=1)

# --- S/R levels ---
if show_levels:
    for lvl in levels[:6]:
        fig.add_hline(
            y=lvl.price, line_dash="solid", line_color="#f9a825",
            opacity=0.55, line_width=1, row=1, col=1,
        )
        fig.add_annotation(
            x=df.index[-1], y=lvl.price,
            xref="x", yref="y",
            text=f" ${lvl.price:.2f} · {lvl.touches}x",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(color="#f9a825", size=10),
            row=1, col=1,
        )

# --- Trendlines (sólo si no hay canal o si el usuario las pidió sin canal) ---
draw_separate_trendlines = show_trendlines and not (show_channel and channel)
if draw_separate_trendlines:
    tl_low = ta.fit_trendline(swings.low_idx, swings.low_prices)
    if tl_low:
        y = tl_low.slope * x_idx + tl_low.intercept
        fig.add_trace(go.Scatter(
            x=df.index[tl_low.start_idx:], y=y[tl_low.start_idx:],
            mode='lines',
            line=dict(color='#26a69a', width=2, dash='dash'),
            name='Soporte din.', showlegend=False,
            hovertemplate='Soporte din.: $%{y:.2f}<extra></extra>',
        ), row=1, col=1)
    tl_high = ta.fit_trendline(swings.high_idx, swings.high_prices)
    if tl_high:
        y = tl_high.slope * x_idx + tl_high.intercept
        fig.add_trace(go.Scatter(
            x=df.index[tl_high.start_idx:], y=y[tl_high.start_idx:],
            mode='lines',
            line=dict(color='#ef5350', width=2, dash='dash'),
            name='Resist. din.', showlegend=False,
            hovertemplate='Resistencia din.: $%{y:.2f}<extra></extra>',
        ), row=1, col=1)

# --- Canal ---
if show_channel and channel:
    yl = channel.lower.slope * x_idx + channel.lower.intercept
    yu = channel.upper.slope * x_idx + channel.upper.intercept
    start = max(channel.lower.start_idx, channel.upper.start_idx)
    fig.add_trace(go.Scatter(
        x=df.index[start:], y=yu[start:],
        mode='lines', line=dict(color='#ef5350', width=2),
        name='Canal sup.', showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index[start:], y=yl[start:],
        mode='lines', line=dict(color='#26a69a', width=2),
        fill='tonexty', fillcolor='rgba(100,150,200,0.08)',
        name='Canal inf.', showlegend=False,
    ), row=1, col=1)

# --- Fibonacci ---
if show_fib and fib:
    fib_colors = {
        0.0: '#888888', 0.236: '#ce93d8', 0.382: '#90caf9',
        0.5:  '#ffb74d', 0.618: '#80cbc4', 0.786: '#a5d6a7', 1.0: '#888888',
    }
    fib_start = min(fib.swing_high_idx, fib.swing_low_idx)
    for ratio, price in fib.levels.items():
        color = fib_colors.get(ratio, '#888')
        fig.add_shape(
            type="line",
            x0=df.index[fib_start], x1=df.index[-1],
            y0=price, y1=price,
            line=dict(color=color, width=1, dash='dot'),
            opacity=0.7, row=1, col=1,
        )
        fig.add_annotation(
            x=df.index[fib_start], y=price,
            text=f"Fib {ratio*100:.1f}% — ${price:.2f}",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(color=color, size=9),
            row=1, col=1,
        )

# --- Marcar patrones ---
if show_patterns:
    for p in patterns:
        if not p.bar_indices:
            continue
        for bi, pr in zip(p.bar_indices, p.prices or [df['Close'].iloc[i] for i in p.bar_indices]):
            fig.add_annotation(
                x=df.index[bi], y=pr,
                text=f"⭐ {p.name}",
                showarrow=True, arrowhead=2, arrowcolor="#ffd54f",
                ax=0, ay=-30,
                font=dict(color="#ffd54f", size=10),
                bgcolor="rgba(0,0,0,0.55)",
                row=1, col=1,
            )

# --- Layout ---
fig.update_layout(
    height=720, template='plotly_dark',
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=110, t=10, b=10),
    paper_bgcolor='#15151f', plot_bgcolor='#15151f',
    hovermode='x unified',
    dragmode='pan',
    font=dict(color='#cccccc'),
)
fig.update_xaxes(gridcolor='#222233', showgrid=True)
fig.update_yaxes(gridcolor='#222233', showgrid=True, row=1, col=1, side='right')
fig.update_yaxes(showgrid=False, row=2, col=1, side='right')
fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
# Saltar fines de semana / no-trading hours
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


# ─────────────────────────── Tabla / Análisis ───────────────────────────
cl, cr = st.columns([1, 1])

with cl:
    st.subheader("📊 Niveles clave")
    if levels:
        rows = []
        for l in levels[:8]:
            dist = (l.price / last - 1) * 100
            rows.append({
                "Precio": f"${l.price:,.2f}",
                "Toques": l.touches,
                "Distancia": f"{dist:+.2f}%",
                "Tipo": "Resistencia" if dist > 0 else "Soporte",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No se detectaron niveles claros. Bajá la sensibilidad o ampliá el período.")

    if fib:
        st.subheader("🔢 Fibonacci")
        st.caption(f"Move {fib.direction.upper()}: "
                   f"${fib.low_price:.2f} → ${fib.high_price:.2f}")
        rows = []
        for r, p in fib.levels.items():
            dist = (p / last - 1) * 100
            rows.append({"Nivel": f"{r*100:.1f}%", "Precio": f"${p:,.2f}",
                         "Distancia": f"{dist:+.2f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with cr:
    st.subheader("🔎 Patrones detectados")
    if patterns:
        for p in patterns:
            with st.container(border=True):
                st.markdown(f"**{p.name}**")
                st.caption(p.description)
    else:
        st.info("Ningún patrón identificado. Probá ampliar el período o "
                "ajustar la sensibilidad.")

    if channel:
        st.subheader("📐 Canal")
        slope_per_bar_lo = channel.lower.slope
        slope_per_bar_up = channel.upper.slope
        st.markdown(
            f"- Pendiente inferior: **{slope_per_bar_lo:+.4f}** USD/vela  \n"
            f"- Pendiente superior: **{slope_per_bar_up:+.4f}** USD/vela  \n"
            f"- Dirección: "
            f"{'📈 Alcista' if (slope_per_bar_lo + slope_per_bar_up)/2 > 0 else '📉 Bajista'}"
        )

    st.subheader("📍 Resumen")
    nearest_above = min(
        ([l for l in levels if l.price > last] or [None]),
        key=lambda l: l.price - last if l else float('inf')
    )
    nearest_below = max(
        ([l for l in levels if l.price < last] or [None]),
        key=lambda l: l.price - last if l else float('-inf')
    )
    summary = []
    if nearest_above:
        d = (nearest_above.price / last - 1) * 100
        summary.append(f"- Resistencia más cercana: **${nearest_above.price:.2f}** ({d:+.2f}%)")
    if nearest_below:
        d = (nearest_below.price / last - 1) * 100
        summary.append(f"- Soporte más cercano: **${nearest_below.price:.2f}** ({d:+.2f}%)")
    summary.append(f"- {len(swings.high_idx)} máximos y {len(swings.low_idx)} mínimos detectados.")
    st.markdown("\n".join(summary))

st.caption("Datos vía Yahoo Finance · No es asesoramiento financiero · "
           "Los algoritmos son un punto de partida, no reemplazan tu análisis.")
