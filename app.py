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

# curl_cffi permite hacer requests "impersonando" un navegador real,
# lo cual evita que Yahoo bloquee las IPs de Streamlit Cloud.
try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

import analysis as ta

# ─────────────────────────── Config ───────────────────────────
st.set_page_config(
    page_title="Analista Técnico",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0a0e1a;
    --bg-tint: #0d1220;
    --surface: #131722;
    --surface-2: #1a1f2e;
    --border: #1e222d;
    --border-light: #2a2e39;
    --text: #d1d4dc;
    --text-mid: #9ca3af;
    --text-muted: #6b7280;
    --text-faint: #4a4d57;
    --green: #00d4aa;
    --green-dim: #00d4aa22;
    --red: #ff5252;
    --red-dim: #ff525222;
    --yellow: #ffb800;
    --magenta: #e879f9;
    --blue: #5b8def;
}

/* — Global reset & base — */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(at 0% 0%, rgba(0,212,170,0.04) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(91,141,239,0.03) 0px, transparent 50%);
}

/* — Hide Streamlit chrome — */
#MainMenu, footer { visibility: hidden; height: 0; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] {
    background: transparent;
}
.stApp > header { background: transparent; }

/* Asegurar que el botón para reabrir el sidebar siempre sea visible */
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999 !important;
}

/* Bloquear el botón "<" que colapsa el sidebar — el sidebar queda siempre fijo */
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[kind="header"],
section[data-testid="stSidebar"] button[kind="headerNoPadding"],
[data-testid="stSidebar"] [data-testid="baseButton-header"] {
    display: none !important;
}

/* — Block container — */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    max-width: none;
}

/* — Branded header — */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 1.25rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.app-brand { display: flex; align-items: center; gap: 0.75rem; }
.app-mark {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--green) 0%, var(--blue) 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono'; font-weight: 600;
    color: var(--bg); font-size: 16px;
    border-radius: 3px;
}
.app-title-block { display: flex; flex-direction: column; gap: 2px; }
.app-title {
    font-family: 'IBM Plex Sans'; font-weight: 600;
    font-size: 16px; letter-spacing: -0.01em; color: var(--text);
    line-height: 1;
}
.app-tagline {
    font-family: 'IBM Plex Sans Condensed'; font-weight: 500;
    font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-muted); line-height: 1;
}

/* — Status pill — */
.status-row { display: flex; gap: 0.5rem; align-items: center; }
.pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.35rem 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: 100px;
    font-family: 'IBM Plex Mono'; font-weight: 500;
    font-size: 10px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-mid);
}
.pill-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 0 var(--green);
    animation: pulse 2.5s ease-out infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0,212,170,0.6); }
    70% { box-shadow: 0 0 0 6px rgba(0,212,170,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,212,170,0); }
}

/* — Ticker display (the big symbol/price row) — */
.tkr-row {
    display: flex; align-items: baseline; gap: 1.5rem;
    padding: 0.5rem 0 1.5rem 0;
    flex-wrap: wrap;
}
.tkr-symbol {
    font-family: 'IBM Plex Mono'; font-weight: 600;
    font-size: 30px; letter-spacing: -0.02em;
    color: var(--text);
}
.tkr-price {
    font-family: 'IBM Plex Mono'; font-weight: 500;
    font-size: 30px; font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}
.tkr-change {
    font-family: 'IBM Plex Mono'; font-weight: 500;
    font-size: 14px; font-variant-numeric: tabular-nums;
    padding: 0.25rem 0.6rem; border-radius: 3px;
}
.tkr-up { color: var(--green); background: var(--green-dim); }
.tkr-down { color: var(--red); background: var(--red-dim); }

/* — Metric strip — */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-bottom: 2rem;
}
.metric {
    background: var(--surface);
    padding: 0.85rem 1.15rem;
}
.metric-label {
    font-family: 'IBM Plex Sans Condensed'; font-weight: 600;
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'IBM Plex Mono'; font-weight: 500;
    font-size: 18px; font-variant-numeric: tabular-nums;
    color: var(--text); letter-spacing: -0.01em;
}
.metric-delta {
    font-family: 'IBM Plex Mono';
    font-size: 11px; font-variant-numeric: tabular-nums;
    margin-top: 0.3rem; color: var(--text-muted);
}
.metric-delta.up { color: var(--green); }
.metric-delta.down { color: var(--red); }

/* — Section labels — */
.section {
    font-family: 'IBM Plex Sans Condensed'; font-weight: 600;
    font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-mid);
    margin: 2.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
}
.section-tag {
    font-family: 'IBM Plex Mono'; font-weight: 400;
    font-size: 10px; color: var(--text-faint);
    letter-spacing: 0.08em;
}

/* — Cards — */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1rem 1.15rem;
    margin-bottom: 0.6rem;
}
.card-pattern {
    border-left: 2px solid var(--yellow);
}
.card-title {
    font-family: 'IBM Plex Sans'; font-weight: 600;
    font-size: 13px; color: var(--text);
    margin-bottom: 0.3rem;
}
.card-desc {
    font-family: 'IBM Plex Sans';
    font-size: 12px; color: var(--text-muted);
    line-height: 1.5;
}

/* — Data table (custom) — */
.dtable {
    width: 100%;
    border: 1px solid var(--border);
    border-collapse: collapse;
    font-family: 'IBM Plex Mono';
    font-size: 12px;
    background: var(--surface);
}
.dtable thead th {
    background: var(--bg-tint);
    color: var(--text-muted);
    font-family: 'IBM Plex Sans Condensed';
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.6rem 0.85rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.dtable tbody td {
    padding: 0.55rem 0.85rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.dtable tbody tr:last-child td { border-bottom: none; }
.dtable tbody tr:hover { background: var(--surface-2); }
.dtable .pos { color: var(--green); }
.dtable .neg { color: var(--red); }
.dtable .muted { color: var(--text-muted); }

/* — Sidebar — */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Sans Condensed' !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-mid) !important;
    margin-top: 1.75rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}
[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Sans' !important;
    font-size: 12px !important;
    color: var(--text-mid) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--bg) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono' !important;
    font-size: 13px !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 1px var(--green) !important;
}

/* Sidebar expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: var(--bg);
    border: 1px solid var(--border-light);
    border-radius: 2px;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-family: 'IBM Plex Sans' !important;
    font-size: 12px !important;
    color: var(--text-mid) !important;
}

/* — Plotly chart frame — */
[data-testid="stPlotlyChart"] {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0.5rem;
}

/* — Streamlit dataframe restyling — */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

/* — Alert boxes — */
.stAlert { border-radius: 2px !important; border: 1px solid var(--border-light) !important; }

/* — Footer — */
.app-footer {
    margin-top: 3.5rem;
    padding: 1.25rem 0 0.5rem 0;
    border-top: 1px solid var(--border);
    font-family: 'IBM Plex Sans Condensed';
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-faint);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
}
.app-footer-source { color: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Data ───────────────────────────
@st.cache_resource
def _get_session():
    """Sesión HTTP que se hace pasar por Chrome para evitar bloqueos en cloud."""
    if HAS_CURL_CFFI:
        return curl_requests.Session(impersonate="chrome")
    return None


def _yf_to_stooq(ticker: str) -> str | None:
    """Convierte un ticker estilo Yahoo Finance al formato de Stooq."""
    t = ticker.upper().strip()
    # Índices Yahoo (prefijo ^) → Stooq tiene formatos propios
    if t.startswith('^'):
        mapping = {
            '^GSPC': '^spx', '^IXIC': '^ndx', '^DJI': '^dji',
            '^RUT': '^rut', '^VIX': '^vix', '^FTSE': '^ftm',
            '^N225': '^nkx', '^GDAXI': '^dax', '^MERV': '^mrv',
        }
        return mapping.get(t, t.lower())
    # Cripto Yahoo (sufijo -USD) → Stooq usa minúsculas pegadas
    if t.endswith('-USD'):
        return t.replace('-USD', 'usd').lower()
    # Forex Yahoo (sufijo =X) → Stooq sin sufijo
    if t.endswith('=X'):
        return t.replace('=X', '').lower()
    # Acciones BCBA argentinas (.BA) — Stooq no las tiene
    if t.endswith('.BA'):
        return None
    # Default: asumo acción US → agregar sufijo .us
    return f"{t.lower()}.us"


def _fetch_stooq(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga data de Stooq como CSV. No requiere API key."""
    stooq_symbol = _yf_to_stooq(ticker)
    if stooq_symbol is None:
        return pd.DataFrame()  # ticker no soportado por Stooq

    interval_map = {'1d': 'd', '1wk': 'w', '1mo': 'm'}
    stooq_interval = interval_map.get(interval, 'd')
    # Stooq sólo da diario/semanal/mensual — intervalos intradía no soportados

    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i={stooq_interval}"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

    if df.empty or 'Date' not in df.columns:
        return pd.DataFrame()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    # Mantener sólo columnas estándar y renombrar si hace falta
    cols_needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df.columns for c in cols_needed):
        # Stooq a veces no devuelve Volume para ciertos activos
        if 'Volume' not in df.columns:
            df['Volume'] = 0
    df = df[cols_needed].dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Filtrar al período pedido
    period_days = {
        '1mo': 30, '3mo': 95, '6mo': 185,
        '1y': 370, '2y': 740, '5y': 1830, 'max': 99999,
    }
    days = period_days.get(period, 185)
    if not df.empty:
        cutoff = df.index[-1] - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    return df


@st.cache_data(ttl=300, show_spinner="Descargando data...")
def load_data(ticker: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    """Devuelve (df, fuente). Intenta yfinance primero, cae a Stooq si falla."""

    # Intento 1: yfinance con curl_cffi
    try:
        session = _get_session()
        kwargs = dict(period=period, interval=interval,
                      progress=False, auto_adjust=False)
        if session is not None:
            kwargs['session'] = session
        df = yf.download(ticker, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(), "Yahoo Finance"
    except Exception:
        pass

    # Intento 2: Stooq fallback (sólo intervals d/w/m)
    if interval in ('1d', '1wk', '1mo'):
        df = _fetch_stooq(ticker, period, interval)
        if not df.empty:
            return df, "Stooq"

    return pd.DataFrame(), "none"


# ─────────────────────────── Sidebar ───────────────────────────
st.sidebar.markdown(
    '<div style="font-family:\'IBM Plex Sans Condensed\';font-weight:700;'
    'font-size:13px;letter-spacing:0.18em;text-transform:uppercase;'
    'color:var(--text);padding:0 0 0.5rem 0;">Configuración</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("### Activo")
ticker = st.sidebar.text_input(
    "Ticker", value="AAPL", label_visibility="collapsed",
    placeholder="AAPL · BTC-USD · YPF · ^GSPC",
).strip().upper()

st.sidebar.markdown("### Marco temporal")
period_opts = {
    "1 mes": "1mo", "3 meses": "3mo", "6 meses": "6mo",
    "1 año": "1y", "2 años": "2y", "5 años": "5y", "Máximo": "max",
}
period = period_opts[st.sidebar.selectbox(
    "Período", list(period_opts.keys()), index=2, label_visibility="collapsed"
)]

interval_opts = {
    "Diario": "1d", "Semanal": "1wk", "Mensual": "1mo",
    "1 hora": "1h", "30 min": "30m", "15 min": "15m",
}
interval = interval_opts[st.sidebar.selectbox(
    "Timeframe", list(interval_opts.keys()), index=0, label_visibility="collapsed"
)]

st.sidebar.markdown("### Sensibilidad")
swing_distance = st.sidebar.slider(
    "Distancia entre swings", 2, 30, 5,
    help="Velas mínimas entre dos swings. Subí para filtrar movimientos menores."
)
swing_prom = st.sidebar.slider(
    "Prominencia mínima (%)", 0.5, 5.0, 1.5, 0.1,
    help="Tamaño mínimo del movimiento para contar como swing significativo."
) / 100

st.sidebar.markdown("### Capas visibles")
show_swings = st.sidebar.checkbox("Swings (▲ ▼)", True)
show_levels = st.sidebar.checkbox("Soportes / Resistencias", True)
show_trendlines = st.sidebar.checkbox("Trendlines", True)
show_channel = st.sidebar.checkbox("Canal", True)
show_fib = st.sidebar.checkbox("Fibonacci", True)
show_patterns = st.sidebar.checkbox("Patrones", True)


# ─────────────────────────── Carga ───────────────────────────
try:
    df, data_source = load_data(ticker, period, interval)
except Exception as e:
    st.error(f"❌ Error cargando **{ticker}**: {e}")
    st.stop()

if df.empty:
    st.error(
        f"❌ No encontré datos para **{ticker}** con período *{period}* / timeframe *{interval}*.\n\n"
        "Posibles causas:\n"
        "- El ticker está mal escrito (probá `AAPL`, `BTC-USD`, `YPF`)\n"
        "- Stooq no cubre ese activo (las acciones BCBA con `.BA` no funcionan, "
        "usá el ADR: `YPF`, `GGAL`, `PAM`, etc.)\n"
        "- Intervalos intradía (1h, 30m, 15m) sólo funcionan vía Yahoo, "
        "que ahora bloquea Streamlit Cloud. Probá *Diario* o *Semanal*.\n"
    )
    st.stop()

if len(df) < 10:
    st.warning(f"⚠️ Sólo {len(df)} velas disponibles. Probá un período más largo.")


# ─────────────────────────── Header ───────────────────────────
last = float(df['Close'].iloc[-1])
prev = float(df['Close'].iloc[-2]) if len(df) > 1 else last
change_pct = (last / prev - 1) * 100 if prev else 0
change_abs = last - prev
hi = float(df['High'].max())
lo = float(df['Low'].min())
vol = float(df['Volume'].iloc[-1])
vol_avg = float(df['Volume'].tail(20).mean())
rng_pct = (hi / lo - 1) * 100

up = change_pct >= 0
mark_letter = ticker[0] if ticker else "•"

# Header con brand + status pill
st.markdown(f"""
<div class="app-header">
    <div class="app-brand">
        <div class="app-mark">{mark_letter}</div>
        <div class="app-title-block">
            <div class="app-title">ANALISTA TÉCNICO</div>
            <div class="app-tagline">PRICE ACTION · AUTO-DETECTION</div>
        </div>
    </div>
    <div class="status-row">
        <div class="pill"><div class="pill-dot"></div>LIVE · {data_source.upper()}</div>
        <div class="pill">{interval.upper()} · {period.upper()}</div>
    </div>
</div>

<div class="tkr-row">
    <div class="tkr-symbol">{ticker}</div>
    <div class="tkr-price">${last:,.2f}</div>
    <div class="tkr-change {'tkr-up' if up else 'tkr-down'}">
        {'▲' if up else '▼'} {change_abs:+,.2f} · {change_pct:+.2f}%
    </div>
</div>

<div class="metric-strip">
    <div class="metric">
        <div class="metric-label">Máximo · período</div>
        <div class="metric-value">${hi:,.2f}</div>
        <div class="metric-delta">{(last/hi - 1)*100:+.2f}% del techo</div>
    </div>
    <div class="metric">
        <div class="metric-label">Mínimo · período</div>
        <div class="metric-value">${lo:,.2f}</div>
        <div class="metric-delta up">{(last/lo - 1)*100:+.2f}% del piso</div>
    </div>
    <div class="metric">
        <div class="metric-label">Rango total</div>
        <div class="metric-value">{rng_pct:.1f}%</div>
        <div class="metric-delta">${lo:,.0f} → ${hi:,.0f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Volumen último</div>
        <div class="metric-value">{vol/1e6:.1f}M</div>
        <div class="metric-delta {'up' if vol >= vol_avg else 'down'}">
            {(vol/vol_avg - 1)*100:+.0f}% vs prom 20
        </div>
    </div>
    <div class="metric">
        <div class="metric-label">Velas · período</div>
        <div class="metric-value">{len(df)}</div>
        <div class="metric-delta">{df.index[0].strftime('%d-%b-%y').upper()} →</div>
    </div>
</div>
""", unsafe_allow_html=True)


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
    height=640, template='plotly_dark',
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=110, t=15, b=10),
    paper_bgcolor='#131722', plot_bgcolor='#131722',
    hovermode='x unified',
    dragmode='pan',
    font=dict(
        family='IBM Plex Mono, monospace',
        color='#9ca3af',
        size=11,
    ),
    hoverlabel=dict(
        bgcolor='#0a0e1a',
        bordercolor='#2a2e39',
        font=dict(family='IBM Plex Mono, monospace', size=11, color='#d1d4dc'),
    ),
)
fig.update_xaxes(
    gridcolor='#1a1f2e', showgrid=True, gridwidth=1,
    linecolor='#1e222d', zerolinecolor='#1e222d',
    tickfont=dict(family='IBM Plex Mono', size=10, color='#6b7280'),
)
fig.update_yaxes(
    gridcolor='#1a1f2e', showgrid=True, gridwidth=1,
    linecolor='#1e222d', zerolinecolor='#1e222d',
    tickfont=dict(family='IBM Plex Mono', size=10, color='#9ca3af'),
    row=1, col=1, side='right',
    tickprefix='$',
)
fig.update_yaxes(
    showgrid=False, row=2, col=1, side='right',
    tickfont=dict(family='IBM Plex Mono', size=9, color='#6b7280'),
)
fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
# Saltar fines de semana / no-trading hours
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

st.markdown('<div class="section">Gráfico <span class="section-tag">CANDLES · '
            f'{interval.upper()}</span></div>', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={
    'scrollZoom': True,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines',
        'hoverClosestCartesian', 'hoverCompareCartesian',
    ],
})


# ─────────────────────────── Tabla / Análisis ───────────────────────────
def _level_table_html(levels_list, last_price):
    if not levels_list:
        return ('<div class="card"><div class="card-desc">'
                'No se detectaron niveles claros. Bajá la sensibilidad o '
                'ampliá el período.</div></div>')
    rows = ""
    for l in levels_list[:8]:
        dist = (l.price / last_price - 1) * 100
        tipo = "Resistencia" if dist > 0 else "Soporte"
        cls = "neg" if dist < 0 else "pos"
        rows += (
            f'<tr><td>${l.price:,.2f}</td>'
            f'<td>{"●" * min(l.touches, 5)} <span class="muted">{l.touches}</span></td>'
            f'<td class="{cls}">{dist:+.2f}%</td>'
            f'<td class="muted">{tipo}</td></tr>'
        )
    return f"""
    <table class="dtable">
        <thead><tr>
            <th>Precio</th><th>Toques</th><th>Distancia</th><th>Tipo</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


def _fib_table_html(fib_obj, last_price):
    if not fib_obj:
        return ""
    rows = ""
    for r, p in fib_obj.levels.items():
        dist = (p / last_price - 1) * 100
        cls = "pos" if dist > 0 else "neg"
        emphasis = "" if r not in (0.382, 0.5, 0.618) else ' style="color:var(--yellow)"'
        rows += (
            f'<tr><td{emphasis}>{r*100:.1f}%</td>'
            f'<td>${p:,.2f}</td>'
            f'<td class="{cls}">{dist:+.2f}%</td></tr>'
        )
    return f"""
    <div style="font-family:'IBM Plex Mono';font-size:11px;color:var(--text-muted);
                margin-bottom:0.6rem;letter-spacing:0.05em;">
        MOVE {fib_obj.direction.upper()} · ${fib_obj.low_price:,.2f} → ${fib_obj.high_price:,.2f}
    </div>
    <table class="dtable">
        <thead><tr><th>Nivel</th><th>Precio</th><th>Distancia</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


cl, cr = st.columns([1, 1], gap="large")

with cl:
    st.markdown(
        '<div class="section">Niveles clave '
        f'<span class="section-tag">{len(levels)} DETECTADOS</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(_level_table_html(levels, last), unsafe_allow_html=True)

    if fib:
        st.markdown(
            '<div class="section">Retrocesos Fibonacci '
            '<span class="section-tag">ÚLTIMO MOVE</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(_fib_table_html(fib, last), unsafe_allow_html=True)

with cr:
    st.markdown(
        '<div class="section">Patrones '
        f'<span class="section-tag">{len(patterns)} DETECTADOS</span></div>',
        unsafe_allow_html=True
    )
    if patterns:
        for p in patterns:
            st.markdown(
                f'<div class="card card-pattern">'
                f'<div class="card-title">{p.name}</div>'
                f'<div class="card-desc">{p.description}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="card"><div class="card-desc">'
            'Ningún patrón clásico identificado. Probá ampliar el período '
            'o ajustar la sensibilidad.</div></div>',
            unsafe_allow_html=True
        )

    # Canal
    if channel:
        avg_slope = (channel.lower.slope + channel.upper.slope) / 2
        direction_txt = "ALCISTA ▲" if avg_slope > 0 else "BAJISTA ▼"
        direction_cls = "pos" if avg_slope > 0 else "neg"
        st.markdown(
            '<div class="section">Canal detectado</div>'
            f'<div class="card">'
            f'<div class="card-title" style="display:flex;justify-content:space-between;">'
            f'<span>Canal {direction_txt[0:8].lower().capitalize()}</span>'
            f'<span class="{direction_cls}" style="font-family:IBM Plex Mono;font-size:11px;">'
            f'{direction_txt}</span></div>'
            f'<div class="card-desc">'
            f'Pendiente inferior: <span style="color:var(--green);font-family:IBM Plex Mono;">'
            f'{channel.lower.slope:+.4f}</span>$/vela · '
            f'Pendiente superior: <span style="color:var(--red);font-family:IBM Plex Mono;">'
            f'{channel.upper.slope:+.4f}</span>$/vela'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # Resumen
    nearest_above = min(
        ([l for l in levels if l.price > last] or [None]),
        key=lambda l: l.price - last if l else float('inf')
    )
    nearest_below = max(
        ([l for l in levels if l.price < last] or [None]),
        key=lambda l: l.price - last if l else float('-inf')
    )
    st.markdown('<div class="section">Resumen ejecutivo</div>', unsafe_allow_html=True)
    summary_html = '<div class="card"><div class="card-desc">'
    if nearest_above:
        d = (nearest_above.price / last - 1) * 100
        summary_html += (
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
            f'<span>Resistencia más cercana</span>'
            f'<span style="font-family:IBM Plex Mono;color:var(--text);">'
            f'${nearest_above.price:,.2f} <span class="neg">({d:+.2f}%)</span></span></div>'
        )
    if nearest_below:
        d = (nearest_below.price / last - 1) * 100
        summary_html += (
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
            f'<span>Soporte más cercano</span>'
            f'<span style="font-family:IBM Plex Mono;color:var(--text);">'
            f'${nearest_below.price:,.2f} <span class="pos">({d:+.2f}%)</span></span></div>'
        )
    summary_html += (
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
        f'<span>Swings detectados</span>'
        f'<span style="font-family:IBM Plex Mono;color:var(--text);">'
        f'{len(swings.high_idx)} ▼ máximos · {len(swings.low_idx)} ▲ mínimos</span></div>'
    )
    summary_html += '</div></div>'
    st.markdown(summary_html, unsafe_allow_html=True)

st.markdown(f"""
<div class="app-footer">
    <div>NOT FINANCIAL ADVICE · ALGORITHMIC DETECTION IS A STARTING POINT</div>
    <div class="app-footer-source">DATA · {data_source.upper()}</div>
</div>
""", unsafe_allow_html=True)
